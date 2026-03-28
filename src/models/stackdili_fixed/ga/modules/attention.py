import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeFPCrossAttention(nn.Module):
    """Cross Pass 2 — NodeFPCrossAttention.

    각 노드를 fp_vec(글로벌 지문)으로 게이팅합니다.
        gate_i = σ(Wq(node_i) · Wk(fp_vec) / √d)
        node_mat_fp = gate_i * node_mat

    단일: node_mat (N, 128), fp_vec (128,)
    배치: node_mat (B, N, 128), fp_vec (B, 128), mask (B, N)
    """

    def __init__(self, node_dim: int = 128):
        super().__init__()
        self.Wq    = nn.Linear(node_dim, node_dim, bias=False)
        self.Wk    = nn.Linear(node_dim, node_dim, bias=False)
        self.scale = node_dim ** 0.5

    def forward(
        self,
        node_mat: torch.Tensor,          # (N,128) or (B,N,128)
        fp_vec:   torch.Tensor,          # (128,)  or (B,128)
        mask:     torch.Tensor = None,   # (B,N) bool — True=실제 원자
    ) -> torch.Tensor:
        batched = node_mat.dim() == 3

        if batched:
            # (B, N, 128) → gate: (B, N, 1)
            k = self.Wk(fp_vec).unsqueeze(-1)            # (B, 128, 1)
            q = self.Wq(node_mat)                        # (B, N, 128)
            gate = torch.sigmoid(
                torch.bmm(q, k) / self.scale             # (B, N, 1)
            )
            if mask is not None:
                gate = gate * mask.unsqueeze(-1).float()
        else:
            # 단일 샘플
            k    = self.Wk(fp_vec).unsqueeze(-1)         # (128, 1)
            q    = self.Wq(node_mat)                     # (N, 128)
            gate = torch.sigmoid((q @ k) / self.scale)   # (N, 1)

        return gate * node_mat


class GraphAttentionReadout(nn.Module):
    """Cross Pass 1 — GraphAttentionReadout.

    Q = fp_vec, K/V = node_mat_fp.
    Q/K/V를 attn_dim(24-dim)으로 축소 후 어텐션 계산.
    A_diff = A1 - λ·A2.
    패딩 위치는 마스크로 -inf 처리하여 softmax에서 제외.

    단일: fp_vec (128,),  node_mat_fp (N,128)
    배치: fp_vec (B,128), node_mat_fp (B,N,128), mask (B,N)
    """

    def __init__(self, node_dim: int = 128, attn_dim: int = 24, lambda_diff: float = 0.5):
        super().__init__()
        self.attn_dim    = attn_dim
        self.lambda_diff = lambda_diff
        self.scale       = attn_dim ** 0.5

        self.W_Q   = nn.Linear(node_dim, attn_dim, bias=False)
        self.W_K   = nn.Linear(node_dim, attn_dim, bias=False)
        self.W_V   = nn.Linear(node_dim, attn_dim, bias=False)
        self.W_out = nn.Linear(attn_dim, node_dim, bias=True)

    def forward(
        self,
        fp_vec:      torch.Tensor,         # (128,)  or (B,128)
        node_mat_fp: torch.Tensor,         # (N,128) or (B,N,128)
        mask:        torch.Tensor = None,  # (B,N) bool
    ) -> torch.Tensor:
        batched = fp_vec.dim() == 2

        if batched:
            Q = self.W_Q(fp_vec).unsqueeze(1)    # (B, 1, 24)
            K = self.W_K(node_mat_fp)            # (B, N, 24)
            V = self.W_V(node_mat_fp)            # (B, N, 24)

            scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (B, 1, N)

            if mask is not None:
                pad = ~mask.unsqueeze(1)  # (B, 1, N) — True=패딩 위치
                # A1: 패딩 → -inf (softmax 후 0)
                scores_A1 = scores.masked_fill(pad, float("-inf"))
                # A2: 부호 반전 후 패딩 → -inf (softmax 후 0)
                scores_A2 = (-scores).masked_fill(pad, float("-inf"))
            else:
                scores_A1 = scores
                scores_A2 = -scores

            A1 = torch.nan_to_num(F.softmax(scores_A1, dim=-1), nan=0.0)
            A2 = torch.nan_to_num(F.softmax(scores_A2, dim=-1), nan=0.0)

            A_diff  = A1 - self.lambda_diff * A2      # (B, 1, N)
            z_24    = torch.bmm(A_diff, V).squeeze(1) # (B, 24)
            return self.W_out(z_24)                   # (B, 128)
        else:
            # 단일 샘플
            Q = self.W_Q(fp_vec).unsqueeze(0)         # (1, 24)
            K = self.W_K(node_mat_fp)                 # (N, 24)
            V = self.W_V(node_mat_fp)                 # (N, 24)

            scores = (Q @ K.T) / self.scale               # (1, N)
            A1 = torch.nan_to_num(F.softmax(scores,  dim=-1), nan=0.0)
            A2 = torch.nan_to_num(F.softmax(-scores, dim=-1), nan=0.0)
            A_diff = A1 - self.lambda_diff * A2
            z_24   = (A_diff @ V).squeeze(0)             # (24,)
            return self.W_out(z_24)                       # (128,)
