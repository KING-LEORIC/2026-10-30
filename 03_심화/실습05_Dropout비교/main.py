from pathlib import Path
# CSV 파일을 읽고 DataFrame 형태로 다루기 위한 라이브러리
import pandas as pd
# PyTorch 핵심 라이브러리
import torch
# 신경망 레이어 및 손실함수 제공
from torch import nn

torch.set_num_threads(1)
# 현재 실행 중인 .py 파일 위치
# 상대경로로 data/, outputs/ 폴더를 찾기 위해 사용
base = Path(__file__).resolve().parent
# Dataset과 DataLoader 사용
# - TensorDataset : 텐서를 데이터셋 형태로 묶음
# - DataLoader : 미니배치 생성 및 셔플 제공
from torch.utils.data import TensorDataset, DataLoader
# 모델 가중치 복사 등에 사용하는 표준 라이브러리
import copy
# GUI 창 없이 이미지 파일 저장용 백엔드 설정
import matplotlib
matplotlib.use("Agg")  # 창을 띄우지 않고 PNG에 저장합니다.
# 학습 결과 그래프 시각화
import matplotlib.pyplot as plt

# 결과 저장 폴더 생성
out = base / "outputs"
# 폴더가 없으면 생성, 있으면 그대로 사용
out.mkdir(exist_ok=True)
# 재현용 기본값 CPU. GPU 실습은 강사 확인 뒤 device 문자열만 cuda:0으로 바꿉니다.
# 학습 장치(Device) 설정
# - cpu : 모든 연산을 CPU에서 수행
# - cuda:0 : 첫 번째 NVIDIA GPU 사용
device = torch.device("cpu")
# CSV 파일 읽기
df = pd.read_csv(base / "data" / "points.csv")
# 학습용 데이터 추출
train = df[df["split"] == "train"]
# 검증용 데이터 추출
valid = df[df["split"] == "valid"]
# 입력 특성(x1, x2)을 float32 텐서로 변환 //신경망 입력으로 사용
X_train = torch.tensor(train[["x1", "x2"]].values, dtype=torch.float32)
# 정답 레이블 변환 //CrossEntropyLoss 사용을 위해 long 타입 필요
y_train = torch.tensor(train["label"].values, dtype=torch.long)
# 검증 데이터도 동일하게 변환 후 device로 이동
X_valid = torch.tensor(valid[["x1", "x2"]].values, dtype=torch.float32).to(device)
y_valid = torch.tensor(valid["label"].values, dtype=torch.long).to(device)
# 다중 클래스 분류용 손실함수
# - 내부적으로 Softmax + Negative Log Likelihood 수행
# - 출력(logits)과 실제 클래스(label)의 차이를 계산
loss_fn = nn.CrossEntropyLoss()
# 학습 시작~종료까지 걸린 시간을 측정 //CPU/GPU 성능 비교 및 실험 기록에 활용
import time
# p=0.1, 0.3, 0.5의 세 후보를 만드세요.
# 첫 10 epochs의 감소, best valid loss, 마지막 train_eval loss를 같이 확인하세요.
# >>> S05
# TODO S05: 위 요구사항에 맞게 이 블록만 완성합니다.

# 드롭아웃 비율(p) 실험 설정
# - dropout=0.1 : 뉴런 10%를 무작위 비활성화
# - dropout=0.3 : 뉴런 30%를 무작위 비활성화
# - dropout=0.5 : 뉴런 50%를 무작위 비활성화
# - 드롭아웃이 커질수록 과적합(overfitting)은 줄어들 수 있지만 지나치면 학습 자체가 어려워질 수 있음
# - 동일한 모델 구조에서 dropout 값만 바꿔 성능을 비교하는 실험
configs = [{'name': 'dropout0.1', 'dropout': 0.1}, {'name': 'dropout0.3', 'dropout': 0.3}, {'name': 'dropout0.5', 'dropout': 0.5}]
# <<< S05
summary = []

for config in configs:
    # 실험 재현성 확보
    # - 동일한 초기 가중치로 시작하여
    #   dropout 값만 성능 차이에 영향을 주도록 설정
    torch.manual_seed(42)
    width = config.get("width", 32)
    # 신경망 구조 생성
    # - 입력(2차원 좌표) → 은닉층(32) → 은닉층(32) → 출력(2개 클래스)
    layers = [nn.Linear(2, width)]
    if config.get("bn", False):
        # BatchNorm 사용 시 배치 단위 평균/분산으로 정규화
        # 학습 안정화 및 수렴 속도 향상 효과
        layers.append(nn.BatchNorm1d(width))
    # - nn.ReLU() : 비선형 활성화 함수
    # - nn.Dropout(config.get("dropout", 0.0)) : 과적합 방지용 Dropout
    layers.extend([nn.ReLU(), nn.Dropout(config.get("dropout", 0.0)), nn.Linear(width, width)])
    if config.get("bn", False):
        layers.append(nn.BatchNorm1d(width))
    # - nn.Linear(width, 2) : 최종 2개 클래스 출력
    layers.extend([nn.ReLU(), nn.Dropout(config.get("dropout", 0.0)), nn.Linear(width, 2)])
    model = nn.Sequential(*layers).to(device)
    batch_size = config.get("batch_size", 32)
    # 미니배치 데이터 로더
    # - batch_size : 한 번의 gradient update에 사용할 샘플 수
    # - shuffle=True : 매 epoch마다 데이터를 섞어 학습 순서에 따른 편향 감소
    # - generator.manual_seed(43) : 셔플 결과를 고정하여 실험 재현성 확보
    # - num_workers=0 : 별도 프로세스 없이 메인 프로세스에서 데이터 로딩
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(43), num_workers=0)
    # AdamW 옵티마이저
    # - Adam의 적응형 학습률 특징 유지
    # - weight decay를 분리 적용하여 일반화 성능 개선
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01,
                                   weight_decay=config.get("weight_decay", 0.0))
    # 학습률 스케줄러
    # - validation loss가 개선되지 않으면 학습률 감소
    # - factor=0.5 : 기존 학습률의 절반으로 감소
    # - patience=2 : 2 epoch 동안 개선이 없을 때 적용
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.5, patience=2, threshold=.001, threshold_mode="abs")
    history = []
    # GPU 사용 시 정확한 시간 측정을 위해 동기화
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    # 전체 학습 반복
    # - 60 epoch 동안 모델 학습
    for epoch in range(1, 60 + 1):
        # 현재 epoch에서 사용할 학습률 기록
        lr_used = optimizer.param_groups[0]["lr"]
        model.train()
        # 현재 epoch의 배치별 loss 누적
        sample_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            # 이전 gradient 초기화
            optimizer.zero_grad()
            # 순전파(forward)
            loss = loss_fn(model(xb), yb)
            # 역전파(backpropagation)
            loss.backward()
            # 가중치 업데이트
            optimizer.step()
            # 샘플 수를 반영한 총 loss 누적
            sample_loss += loss.item() * len(yb)
        # 평가 모드 전환
        # - Dropout 비활성화
        # - BatchNorm은 이동평균 사용
        model.eval()
        with torch.no_grad():
            # 전체 train 데이터 기준 loss 측정 //드롭아웃 없이 순수 성능 평가
            train_loss = loss_fn(model(X_train.to(device)), y_train.to(device)).item()
            # validation 데이터 예측
            logits = model(X_valid)
            # validation loss 계산
            valid_loss = loss_fn(logits, y_valid).item()
            # validation accuracy 계산
            valid_acc = (logits.argmax(1) == y_valid).float().mean().item()
        # validation loss 기준으로 학습률 조정
        if config.get("scheduler", False):
            scheduler.step(valid_loss)
        # epoch별 결과 저장
        history.append([epoch, sample_loss / len(train), train_loss, valid_loss, valid_acc,
                        lr_used, optimizer.param_groups[0]["lr"], len(loader)])
    # GPU 시간 측정 종료 전 동기화
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    # 학습 이력을 DataFrame으로 변환 //이후 CSV 저장 및 분석에 사용
    table = pd.DataFrame(history, columns=["epoch", "train_loss_step", "train_loss_eval", "valid_loss",
                                           "valid_acc", "lr_used", "lr_next", "iterations"])
    # 가장 낮은 validation loss를 기록한 epoch 추출 //일반적으로 가장 좋은 일반화 성능을 의미
    best = table.loc[table["valid_loss"].idxmin()]
    name = config["name"]
    # 각 실험의 학습 기록 저장
    table.to_csv(out / f"history_{name}.csv", index=False)
    # 최종 비교용 요약 정보 저장
    summary.append({**config, "best_epoch": int(best["epoch"]), "best_valid_loss": float(best["valid_loss"]),
                    "last_train_loss_eval": train_loss, "last_valid_loss": valid_loss,
                    "iterations_per_epoch": len(loader), "total_steps": len(loader)*60,
                    "elapsed_seconds": elapsed})
    # 학습률이 감소한 epoch 확인 //scheduler 동작 시점을 분석할 수 있음
    changes = table[table["lr_next"] < table["lr_used"]]["epoch"].tolist()
    print(name, "best epoch:", int(best["epoch"]), "best valid loss:", round(best["valid_loss"], 6),
          "iterations:", len(loader), "lr 감소 epoch:", changes)
    # Loss 곡선 시각화
    # - train_loss_eval : 학습 데이터 성능
    # - valid_loss : 검증 데이터 성능
    # - best epoch : 최적 모델 위치 표시
    plt.figure(figsize=(8, 5))
    plt.plot(table["epoch"], table["train_loss_eval"], label="train (eval)")
    plt.plot(table["epoch"], table["valid_loss"], label="valid")
    plt.axvline(best["epoch"], linestyle="--", label="best epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    # 학습 곡선 이미지 저장
    plt.savefig(out / f"curves_{name}.png", dpi=150)
    plt.close()
# 실험별 핵심 결과를 하나의 표로 저장 //dropout 비율에 따른 성능 비교에 활용
pd.DataFrame(summary).to_csv(out / "experiment_table.csv", index=False)
print("시간은 experiment_table.csv에 기록하며 CPU/GPU·실행 부하에 따라 달라집니다.")
