import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold

from models.stackdili_fixed.ga.base import BaseGA
from models.stackdili_fixed.ga.modules.atom_features import smiles_to_graph
from models.stackdili_fixed.ga.modules.gcn import TwoLayerGCN
from models.stackdili_fixed.ga.modules.attention import (
    NodeFPCrossAttention,
    GraphAttentionReadout,
)


class _DualPathModel(nn.Module):
    """Path A(fp_vec) + Path B(GCN+CrossAttention) → 256-dim embedding.

    배치 입력:
        fp_vec    : (B, 128)
        atom_feat : (B, max_N, 25)
        adj       : (B, max_N, max_N)
        mask      : (B, max_N)  True=실제 원자
    출력:
        embedding : (B, 256)
        logits    : (B, 2)
    """

    def __init__(self, attn_dim: int = 24, lambda_diff: float = 0.5, dropout: float = 0.3):
        super().__init__()
        self.gcn         = TwoLayerGCN()
        self.cross_pass2 = NodeFPCrossAttention(node_dim=128)
        self.cross_pass1 = GraphAttentionReadout(
            node_dim=128, attn_dim=attn_dim, lambda_diff=lambda_diff
        )
        self.aux_mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(
        self,
        fp_vec:    torch.Tensor,
        atom_feat: torch.Tensor,
        adj:       torch.Tensor,
        mask:      torch.Tensor = None,
    ):
        node_mat    = self.gcn(atom_feat, adj)                    # (B, N, 128)
        node_mat_fp = self.cross_pass2(node_mat, fp_vec, mask)   # (B, N, 128)
        z_readout   = self.cross_pass1(fp_vec, node_mat_fp, mask)  # (B, 128)
        embedding   = torch.cat([fp_vec, z_readout], dim=-1)     # (B, 256)
        logits      = self.aux_mlp(embedding)                     # (B, 2)
        return embedding, logits


class GAv5(BaseGA):
    """이미지 아키텍처 기반 피처 선택 (v5) — 미니배치 + 임베딩 스태킹 연결.

    Path A: VT → RF Top-128 → fp_vec (128-dim, Projection Block 없음)
    Path B: SMILES → TwoLayerGCN → NodeFPCrossAttention → GraphAttentionReadout
            Q/K/V: 24-dim, 패딩 기반 미니배치 학습

    학습 후 전체 샘플(train+test)의 256-dim 임베딩을 Feature.csv에 추가합니다.
    select_features()는 RF Top-128 컬럼명 + emb_000..emb_255 (384개)를 반환합니다.
    """

    def __init__(
        self,
        n_top_features: int   = 128,
        var_threshold:  float = 0.01,
        attn_dim:       int   = 24,
        lambda_diff:    float = 0.5,
        epochs:         int   = 50,
        batch_size:     int   = 32,
        lr:             float = 1e-3,
        dropout:        float = 0.3,
        random_seed:    int   = 42,
    ):
        self.n_top_features = n_top_features
        self.var_threshold  = var_threshold
        self.attn_dim       = attn_dim
        self.lambda_diff    = lambda_diff
        self.epochs         = epochs
        self.batch_size     = batch_size
        self.lr             = lr
        self.dropout        = dropout
        self.random_seed    = random_seed

    # ------------------------------------------------------------------
    # 경로 유틸
    # ------------------------------------------------------------------

    def _resolve_paths(self):
        module_dir   = os.path.dirname(os.path.abspath(__file__))
        features_dir = os.path.normpath(
            os.path.join(module_dir, "..", "..", "..", "features")
        )
        return (
            os.path.join(features_dir, "Feature_raw.csv"),
            os.path.join(features_dir, "Feature.csv"),
        )

    # ------------------------------------------------------------------
    # Path A
    # ------------------------------------------------------------------

    def _select_path_a(self, X: pd.DataFrame, y: pd.Series):
        vt = VarianceThreshold(threshold=self.var_threshold)
        vt.fit(X)
        kept   = X.columns[vt.get_support()].tolist()
        X_vt   = X[kept]
        print(f"[Path A] VT: {X.shape[1]} → {len(kept)}개")

        k  = min(self.n_top_features, X_vt.shape[1])
        rf = RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            random_state=self.random_seed, n_jobs=-1,
        )
        rf.fit(X_vt.values, y.values)
        top_idx = np.argsort(rf.feature_importances_)[::-1][:k]
        fp_cols = [kept[i] for i in top_idx]
        print(f"[Path A] RF Top-{k}: {len(fp_cols)}개 (Projection Block 없음)")
        return fp_cols

    # ------------------------------------------------------------------
    # Path B — 그래프 생성 + 패딩 배치
    # ------------------------------------------------------------------

    def _build_graphs(self, smiles_list: list):
        graphs, fail = [], 0
        for smi in smiles_list:
            af, adj = smiles_to_graph(smi)
            if af is None:
                graphs.append(None)
                fail += 1
            else:
                graphs.append((af, adj))
        if fail:
            print(f"[Path B] SMILES 파싱 실패: {fail}개 (해당 샘플 패딩 0으로 처리)")
        return graphs

    def _build_padded_tensors(self, graphs, fp_mat: np.ndarray):
        """그래프 리스트 → 패딩된 배치 텐서.

        Returns:
            atom_t : (M, max_N, 25)
            adj_t  : (M, max_N, max_N)
            mask_t : (M, max_N)  bool
            fp_t   : (M, 128)
            valid  : (M,) bool — 파싱 성공 여부
        """
        valid_graphs = [g for g in graphs if g is not None]
        max_n = max(g[0].shape[0] for g in valid_graphs) if valid_graphs else 1
        m     = len(graphs)

        atom_t = torch.zeros(m, max_n, 25,     dtype=torch.float32)
        adj_t  = torch.zeros(m, max_n, max_n,  dtype=torch.float32)
        mask_t = torch.zeros(m, max_n,          dtype=torch.bool)
        valid  = torch.zeros(m,                 dtype=torch.bool)

        for i, g in enumerate(graphs):
            if g is None:
                continue
            af, adj = g
            n = af.shape[0]
            atom_t[i, :n, :] = torch.tensor(af,  dtype=torch.float32)
            adj_t [i, :n, :n] = torch.tensor(adj, dtype=torch.float32)
            mask_t[i, :n]    = True
            valid [i]         = True

        fp_t = torch.tensor(fp_mat, dtype=torch.float32)
        return atom_t, adj_t, mask_t, fp_t, valid

    # ------------------------------------------------------------------
    # 학습
    # ------------------------------------------------------------------

    def _get_fp_mat(self, raw_df: pd.DataFrame, fp_cols: list) -> np.ndarray:
        """전체 DataFrame에서 fp_cols 추출 + min-max 정규화."""
        mat = raw_df[fp_cols].values.astype(np.float32)
        col_min = mat.min(axis=0, keepdims=True)
        col_max = mat.max(axis=0, keepdims=True)
        return (mat - col_min) / (col_max - col_min + 1e-8)

    def _train(
        self,
        model:   _DualPathModel,
        atom_t:  torch.Tensor,
        adj_t:   torch.Tensor,
        mask_t:  torch.Tensor,
        fp_t:    torch.Tensor,
        valid:   torch.Tensor,
        y:       np.ndarray,
    ) -> None:
        # 유효 샘플만 학습
        idx    = valid.nonzero(as_tuple=True)[0]
        y_t    = torch.tensor(y[idx.numpy()], dtype=torch.long)
        ds     = TensorDataset(
            atom_t[idx], adj_t[idx], mask_t[idx], fp_t[idx], y_t
        )
        loader = DataLoader(
            ds, batch_size=self.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(self.random_seed),
        )

        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        model.train()

        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            for atom_b, adj_b, mask_b, fp_b, label_b in loader:
                optimizer.zero_grad()
                _, logits = model(fp_b, atom_b, adj_b, mask_b)
                loss = criterion(logits, label_b)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 10 == 0 or epoch == 1:
                n_steps = len(loader)
                print(
                    f"[학습] epoch {epoch:3d}/{self.epochs}"
                    f" | loss={total_loss/n_steps:.4f}"
                    f" | steps={n_steps}"
                )

    # ------------------------------------------------------------------
    # 임베딩 추출
    # ------------------------------------------------------------------

    def _get_embeddings(
        self,
        model:  _DualPathModel,
        atom_t: torch.Tensor,
        adj_t:  torch.Tensor,
        mask_t: torch.Tensor,
        fp_t:   torch.Tensor,
    ) -> np.ndarray:
        model.eval()
        all_emb = []
        loader  = DataLoader(
            TensorDataset(atom_t, adj_t, mask_t, fp_t),
            batch_size=self.batch_size, shuffle=False,
        )
        with torch.no_grad():
            for atom_b, adj_b, mask_b, fp_b in loader:
                emb, _ = model(fp_b, atom_b, adj_b, mask_b)
                all_emb.append(emb.numpy())
        return np.concatenate(all_emb, axis=0)  # (M, 256)

    # ------------------------------------------------------------------
    # BaseGA 인터페이스
    # ------------------------------------------------------------------

    def select_features(self, X: pd.DataFrame, y: pd.Series) -> list:
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        _, feat_path = self._resolve_paths()

        # 1. Path A: VT → RF Top-128
        fp_cols = self._select_path_a(X, y)

        # 2. Feature.csv 전체 로드 (train + test) — clean 적용 여부에 맞게 현재 파일 사용
        raw_df     = pd.read_csv(feat_path)
        train_mask = raw_df["ref"] != "DILIrank"
        print(f"[Path B] 전체 샘플: {len(raw_df)}개 (train={train_mask.sum()}, test={(~train_mask).sum()})")

        # 3. fp_mat: 전체 샘플용 (정규화)
        all_fp_mat = self._get_fp_mat(raw_df, fp_cols)

        # train fp_mat (X와 동일 행 순서)
        train_fp_mat = all_fp_mat[train_mask.values]

        # 4. 전체 SMILES → 분자 그래프
        print("[Path B] 분자 그래프 생성 중...")
        all_graphs = self._build_graphs(raw_df["SMILES"].tolist())

        # 5. 패딩 배치 텐서 생성 (전체)
        atom_t, adj_t, mask_t, fp_t, valid = self._build_padded_tensors(
            all_graphs, all_fp_mat
        )
        print(f"[Path B] 패딩 완료: max_atoms={atom_t.shape[1]}, 배치 크기={self.batch_size}")

        # train 전용 텐서
        train_idx  = train_mask.values
        atom_tr    = atom_t[train_idx]
        adj_tr     = adj_t [train_idx]
        mask_tr    = mask_t[train_idx]
        fp_tr      = fp_t  [train_idx]
        valid_tr   = valid [train_idx]

        # 6. 미니배치 학습 (train only)
        model = _DualPathModel(
            attn_dim=self.attn_dim,
            lambda_diff=self.lambda_diff,
            dropout=self.dropout,
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[학습] DualPathModel | params={n_params:,} | epochs={self.epochs} | batch={self.batch_size}")
        self._train(model, atom_tr, adj_tr, mask_tr, fp_tr, valid_tr, y.values)

        # 7. 전체 샘플 256-dim 임베딩 추출
        print("[임베딩] 전체 샘플 추출 중...")
        embeddings = self._get_embeddings(model, atom_t, adj_t, mask_t, fp_t)
        print(f"[임베딩] 완료: {embeddings.shape}")

        # 8. Feature.csv에 임베딩 컬럼 추가 후 저장
        emb_cols = [f"emb_{i:03d}" for i in range(256)]
        emb_df   = pd.DataFrame(embeddings, columns=emb_cols, index=raw_df.index)
        out_df   = pd.concat(
            [raw_df[["SMILES", "Label", "ref"] + fp_cols], emb_df], axis=1
        )
        out_df.to_csv(feat_path, index=False)
        print(f"[Feature.csv] 저장 완료: {len(fp_cols) + 256}개 피처 (RF {len(fp_cols)} + emb 256)")

        # 9. BaseGA 계약: 컬럼명 리스트 반환
        selected = fp_cols + emb_cols
        print(f"\n[GA v5] 최종 선택: {len(selected)}개 피처")
        return selected
