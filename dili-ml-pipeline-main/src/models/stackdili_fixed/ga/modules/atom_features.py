import numpy as np
from rdkit import Chem

# 원자번호 OHE 대상 (22종)
ATOM_LIST = [
    6, 7, 8, 9, 15, 16, 17, 35, 53,   # C N O F P S Cl Br I
    11, 12, 14, 19, 20, 26, 27, 28,    # Na Mg Si K Ca Fe Co Ni
    29, 30, 33, 34, 0,                  # Cu Zn As Se Unknown(0)
]
ATOM_TO_IDX = {a: i for i, a in enumerate(ATOM_LIST)}


def _atom_features(atom) -> np.ndarray:
    """단일 원자 → 25-dim 피처 벡터.

    OHE 원자번호 (22) + formal_charge (1) + is_aromatic (1) + num_Hs (1)
    """
    ohe = np.zeros(len(ATOM_LIST), dtype=np.float32)
    idx = ATOM_TO_IDX.get(atom.GetAtomicNum(), ATOM_TO_IDX[0])
    ohe[idx] = 1.0

    extra = np.array([
        float(atom.GetFormalCharge()),
        float(atom.GetIsAromatic()),
        float(atom.GetTotalNumHs()),
    ], dtype=np.float32)

    return np.concatenate([ohe, extra])  # (25,)


def smiles_to_graph(smiles: str):
    """SMILES → (atom_feat, adj_norm).

    Returns:
        atom_feat : np.ndarray (N, 25)
        adj_norm  : np.ndarray (N, N)  D⁻¹A 정규화 인접 행렬
        None, None: 파싱 실패 시
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    n = mol.GetNumAtoms()
    atom_feat = np.stack([_atom_features(a) for a in mol.GetAtoms()])  # (N, 25)

    adj = np.zeros((n, n), dtype=np.float32)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    np.fill_diagonal(adj, 1.0)  # self-loop

    # D⁻¹A 정규화
    degree = adj.sum(axis=1, keepdims=True).clip(min=1.0)
    adj_norm = adj / degree

    return atom_feat, adj_norm
