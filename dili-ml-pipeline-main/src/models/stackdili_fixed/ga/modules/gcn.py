import torch
import torch.nn as nn


class TwoLayerGCN(nn.Module):
    """2-hop GCN with JK(Jumping Knowledge) concat + residual.

    단일 샘플: atom_feat (N, 25), adj (N, N)
    배치      : atom_feat (B, N, 25), adj (B, N, N)

    Layer 1: Linear(25→64),  D⁻AW₁, ReLU  → H1
    Layer 2: Linear(64→128), D⁻AW₂, ReLU  → H2
    W_res:   Linear(25→128)                → skip connection
    JK-Cat:  cat(H1, H2) → Linear(192→128) → node_mat
    """

    def __init__(self):
        super().__init__()
        self.W1    = nn.Linear(25,  64,  bias=False)
        self.W2    = nn.Linear(64,  128, bias=False)
        self.W_res = nn.Linear(25,  128, bias=False)
        self.W_jk  = nn.Linear(64 + 128, 128, bias=True)
        self.act   = nn.ReLU()

    def forward(self, atom_feat: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            atom_feat : (N, 25) 또는 (B, N, 25)
            adj       : (N, N)  또는 (B, N, N)  — D⁻¹A 정규화 인접 행렬
        Returns:
            node_mat  : (N, 128) 또는 (B, N, 128)
        """
        # @ 연산자가 배치 차원을 자동 처리 (2-D, 3-D 모두 동작)
        H1 = self.act(adj @ self.W1(atom_feat))                          # (..., N, 64)
        H2 = self.act(adj @ self.W2(H1) + self.W_res(atom_feat))        # (..., N, 128)
        node_mat = self.W_jk(torch.cat([H1, H2], dim=-1))               # (..., N, 128)
        return node_mat
