"""제공 코드: split을 다시 만들지 않고 train·valid만 Tensor로 반환합니다."""
# 파일 경로 처리를 위한 라이브러리
from pathlib import Path
# CSV 데이터 처리
import pandas as pd
# Tensor 생성
import torch

# CSV 파일을 읽어 train / valid 데이터를 Tensor로 변환
def load_data(path):
    # CSV 파일 로드
    df = pd.read_csv(path)
    # split 컬럼이 train인 데이터만 선택
    tr = df[df["split"] == "train"]
    # split 컬럼이 valid인 데이터만 선택
    va = df[df["split"] == "valid"]
    # train 입력 데이터(x1, x2)를 float Tensor로 변환
    X = torch.tensor(tr[["x1", "x2"]].values, dtype=torch.float32)
    # train 정답(label)을 long Tensor로 변환
    y = torch.tensor(tr["label"].values, dtype=torch.long)
    # validation 입력 데이터
    Xv = torch.tensor(va[["x1", "x2"]].values, dtype=torch.float32)
    # validation 정답 데이터
    yv = torch.tensor(va["label"].values, dtype=torch.long)
    # validation 데이터의 id 목록 반환 //나중에 예측 결과 저장용 id 목록
    return X, y, Xv, yv, va["id"].tolist()
