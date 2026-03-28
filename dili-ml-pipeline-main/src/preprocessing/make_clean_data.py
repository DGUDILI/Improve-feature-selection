import os
import pandas as pd
from rdkit import Chem

# 경로 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGINAL_DATA_PATH = os.path.join(SCRIPT_DIR, "../features/Feature.csv")
CLEANED_DATA_PATH = os.path.join(SCRIPT_DIR, "../features/Feature_cleaned.csv")

print("[ 🧹 중복 데이터(치팅) 청소 작업 시작 ]")
print("-" * 60)

# 1. 데이터 로드
data = pd.read_csv(ORIGINAL_DATA_PATH)
train_data = data[data['ref'] != 'DILIrank'].copy()
test_data = data[data['ref'] == 'DILIrank'].copy()

# 2. Canonical SMILES 변환 함수
def get_canonical(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except:
        return None

# 3. 임시로 정규화된 SMILES 열 추가
print("SMILES 정규화 중... (잠시만 기다려주세요)")
train_data['Canonical_SMILES'] = train_data['SMILES'].apply(get_canonical)
test_data['Canonical_SMILES'] = test_data['SMILES'].apply(get_canonical)

# 4. Test 셋의 정규화된 SMILES 목록(정답지) 추출
test_canonical_set = set(test_data['Canonical_SMILES'].dropna())

# 5. Train 셋에서 Test 셋과 겹치는(치팅) 데이터 필터링 (제거!)
initial_train_len = len(train_data)
train_cleaned = train_data[~train_data['Canonical_SMILES'].isin(test_canonical_set)].copy()
removed_count = initial_train_len - len(train_cleaned)

print(f"✔️ 원본 Train 데이터 개수: {initial_train_len}개")
print(f"🚨 제거된 중복(치팅) 데이터 개수: {removed_count}개")
print(f"✨ 청소 완료된 Train 데이터 개수: {len(train_cleaned)}개")

# 6. 임시 열 삭제 및 데이터 병합
train_cleaned = train_cleaned.drop(columns=['Canonical_SMILES'])
test_data = test_data.drop(columns=['Canonical_SMILES'])

# 다시 하나의 데이터프레임으로 합치기
cleaned_data = pd.concat([train_cleaned, test_data], axis=0, ignore_index=True)

# 7. 새로운 CSV 파일로 저장
cleaned_data.to_csv(CLEANED_DATA_PATH, index=False)
print("-" * 60)
print(f"💾 깨끗한 데이터셋이 저장되었습니다: {CLEANED_DATA_PATH}")