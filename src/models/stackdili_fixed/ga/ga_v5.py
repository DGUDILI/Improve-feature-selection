import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold

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
        epochs:         int   = 100,
        batch_size:     int   = 64,
        lr:             float = 1e-3,
        dropout:        float = 0.3,
        random_seed:    int   = 42,
        n_splits:       int   = 5,
        early_stop_patience: int   = 12,
        early_stop_delta:    float = 0.003,
    ):
        self.n_top_features      = n_top_features
        self.var_threshold       = var_threshold
        self.attn_dim            = attn_dim
        self.lambda_diff         = lambda_diff
        self.epochs              = epochs
        self.batch_size          = batch_size
        self.lr                  = lr
        self.dropout             = dropout
        self.random_seed         = random_seed
        self.n_splits            = n_splits
        self.early_stop_patience = early_stop_patience
        self.early_stop_delta    = early_stop_delta

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
        top_idx        = np.argsort(rf.feature_importances_)[::-1][:k]
        fp_cols        = [kept[i] for i in top_idx]
        fp_importances = rf.feature_importances_[top_idx].tolist()
        print(f"[Path A] RF Top-{k}: {len(fp_cols)}개 (Projection Block 없음)")
        return fp_cols, fp_importances

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

    def _get_fp_mat(
        self,
        raw_df: pd.DataFrame,
        fp_cols: list,
        train_mask: np.ndarray = None,
    ) -> np.ndarray:
        """fp_cols 추출 + min-max 정규화.

        train_mask가 주어지면 train 행 기준으로 통계를 계산하고
        전체(train+test)에 동일한 스케일을 적용합니다 (누수 방지).
        train_mask가 None이면 전체 기준으로 정규화합니다.
        """
        mat = raw_df[fp_cols].values.astype(np.float32)
        fit_mat = mat[train_mask] if train_mask is not None else mat
        col_min = fit_mat.min(axis=0, keepdims=True)
        col_max = fit_mat.max(axis=0, keepdims=True)
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

        # 클래스 불균형 보정: pos_weight = neg/pos 비율
        y_arr    = y[idx.numpy()]
        n_pos    = (y_arr == 1).sum()
        n_neg    = (y_arr == 0).sum()
        pos_w    = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0
        weight   = torch.tensor([1.0, pos_w], dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=weight)

        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.lr * 0.1
        )
        model.train()

        best_loss      = float('inf')
        no_improve_cnt = 0

        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            for atom_b, adj_b, mask_b, fp_b, label_b in loader:
                optimizer.zero_grad()
                _, logits = model(fp_b, atom_b, adj_b, mask_b)
                loss = criterion(logits, label_b)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()

            avg_loss = total_loss / len(loader)

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"[학습] epoch {epoch:3d}/{self.epochs}"
                    f" | loss={avg_loss:.4f}"
                    f" | lr={scheduler.get_last_lr()[0]:.5f}"
                    f" | pos_w={pos_w:.2f}"
                )

            # Early stopping: 손실이 delta 이상 개선되지 않으면 카운트
            if avg_loss < best_loss - self.early_stop_delta:
                best_loss      = avg_loss
                no_improve_cnt = 0
            else:
                no_improve_cnt += 1
                if no_improve_cnt >= self.early_stop_patience:
                    print(
                        f"[Early Stop] epoch {epoch} | "
                        f"best_loss={best_loss:.4f} | patience={self.early_stop_patience}"
                    )
                    break

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
    # OOF 임베딩
    # ------------------------------------------------------------------

    def _oof_embeddings(
        self,
        atom_tr_all: torch.Tensor,
        adj_tr_all:  torch.Tensor,
        mask_tr_all: torch.Tensor,
        fp_tr_all:   torch.Tensor,
        valid_tr_all: torch.Tensor,
        y_train:     np.ndarray,
        atom_te:     torch.Tensor,
        adj_te:      torch.Tensor,
        mask_te:     torch.Tensor,
        fp_te:       torch.Tensor,
    ) -> tuple:
        """Fold별 독립 GCN 학습으로 OOF 임베딩 생성.

        Returns:
            oof_emb_train : (N_train, 256)  — 각 샘플이 val fold일 때 추출된 임베딩
            test_emb_avg  : (N_test,  256)  — K개 fold 모델의 test 임베딩 평균
        """
        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_seed
        )

        n_train = atom_tr_all.shape[0]
        n_test  = atom_te.shape[0]
        oof_emb_train  = np.zeros((n_train, 256), dtype=np.float32)
        test_emb_accum = np.zeros((n_test,  256), dtype=np.float32)

        for fold_i, (tr_idx, val_idx) in enumerate(
            skf.split(np.zeros(n_train), y_train)
        ):
            print(
                f"\n[GCN OOF] ── Fold {fold_i + 1}/{self.n_splits} "
                f"(train={len(tr_idx)}, val={len(val_idx)}) ──"
            )
            torch.manual_seed(self.random_seed + fold_i)

            model = _DualPathModel(
                attn_dim=self.attn_dim,
                lambda_diff=self.lambda_diff,
                dropout=self.dropout,
            )

            self._train(
                model,
                atom_tr_all[tr_idx], adj_tr_all[tr_idx],
                mask_tr_all[tr_idx], fp_tr_all[tr_idx],
                valid_tr_all[tr_idx], y_train[tr_idx],
            )

            # OOF: val 샘플 임베딩 (이 모델이 한 번도 본 적 없는 샘플)
            oof_emb_train[val_idx] = self._get_embeddings(
                model,
                atom_tr_all[val_idx], adj_tr_all[val_idx],
                mask_tr_all[val_idx], fp_tr_all[val_idx],
            )

            # test 임베딩 누적
            test_emb_accum += self._get_embeddings(
                model, atom_te, adj_te, mask_te, fp_te,
            )

        test_emb_avg = test_emb_accum / self.n_splits
        print(
            f"\n[GCN OOF] 완료 | "
            f"train OOF emb: {oof_emb_train.shape}, "
            f"test avg emb: {test_emb_avg.shape}"
        )
        return oof_emb_train, test_emb_avg

    # ------------------------------------------------------------------
    # BaseGA 인터페이스
    # ------------------------------------------------------------------

    def select_features(self, X: pd.DataFrame, y: pd.Series) -> list:
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        _, feat_path = self._resolve_paths()

        # 1. Path A: VT → RF Top-128
        fp_cols, fp_importances = self._select_path_a(X, y)

        # 2. Feature.csv 전체 로드 (train + test) — clean 적용 여부에 맞게 현재 파일 사용
        raw_df     = pd.read_csv(feat_path)
        train_mask = raw_df["ref"] != "DILIrank"
        print(f"[Path B] 전체 샘플: {len(raw_df)}개 (train={train_mask.sum()}, test={(~train_mask).sum()})")

        # 3. fp_mat: train 기준 min-max 통계로 전체 정규화 (누수 방지)
        all_fp_mat = self._get_fp_mat(raw_df, fp_cols, train_mask=train_mask.values)

        # 4. 전체 SMILES → 분자 그래프
        print("[Path B] 분자 그래프 생성 중...")
        all_graphs = self._build_graphs(raw_df["SMILES"].tolist())

        # 5. 패딩 배치 텐서 생성 (전체)
        atom_t, adj_t, mask_t, fp_t, valid = self._build_padded_tensors(
            all_graphs, all_fp_mat
        )
        print(f"[Path B] 패딩 완료: max_atoms={atom_t.shape[1]}, 배치 크기={self.batch_size}")

        # train / test 인덱스
        train_idx_arr = train_mask.values
        test_idx_arr  = ~train_idx_arr

        # train 전용 텐서
        atom_tr    = atom_t[train_idx_arr]
        adj_tr     = adj_t [train_idx_arr]
        mask_tr    = mask_t[train_idx_arr]
        fp_tr      = fp_t  [train_idx_arr]
        valid_tr   = valid [train_idx_arr]

        # test 전용 텐서
        atom_te    = atom_t[test_idx_arr]
        adj_te     = adj_t [test_idx_arr]
        mask_te    = mask_t[test_idx_arr]
        fp_te      = fp_t  [test_idx_arr]

        # 6-7. Fold별 독립 GCN 학습 → OOF 임베딩 생성
        _tmp_model = _DualPathModel()
        n_params   = sum(p.numel() for p in _tmp_model.parameters())
        del _tmp_model
        print(
            f"[GCN OOF] DualPathModel | params={n_params:,} | "
            f"epochs={self.epochs} | batch={self.batch_size} | "
            f"n_splits={self.n_splits}"
        )

        oof_tr_emb, avg_te_emb = self._oof_embeddings(
            atom_tr, adj_tr, mask_tr, fp_tr, valid_tr,
            y.values,
            atom_te, adj_te, mask_te, fp_te,
        )

        # 전체 임베딩 조립: (N_all, 256)
        embeddings = np.zeros((len(raw_df), 256), dtype=np.float32)
        embeddings[train_idx_arr] = oof_tr_emb
        embeddings[test_idx_arr]  = avg_te_emb
        print(f"[임베딩] 조립 완료: {embeddings.shape} "
              f"(train OOF + test avg)")

        # 8. Feature.csv에 임베딩 컬럼 추가 후 저장
        emb_cols = [f"emb_{i:03d}" for i in range(256)]
        emb_df   = pd.DataFrame(embeddings, columns=emb_cols, index=raw_df.index)
        out_df   = pd.concat(
            [raw_df[["SMILES", "Label", "ref"] + fp_cols], emb_df], axis=1
        )
        out_df.to_csv(feat_path, index=False)
        print(f"[Feature.csv] 저장 완료: {len(fp_cols) + 256}개 피처 (RF {len(fp_cols)} + emb 256)")

        # 9. 피처 메타 정보 JSON 저장 (stacking에서 동적 TOP_FEATURES 활용)
        features_dir  = os.path.dirname(feat_path)
        meta_path     = os.path.join(features_dir, "ga_v5_feature_meta.json")
        meta = {
            "fp_cols":        fp_cols,
            "fp_importances": fp_importances,
            "emb_cols":       emb_cols,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[GA v5] 피처 메타 저장: {meta_path}")

        # 10. BaseGA 계약: 컬럼명 리스트 반환
        selected = fp_cols + emb_cols
        print(f"\n[GA v5] 최종 선택: {len(selected)}개 피처")
        return selected
