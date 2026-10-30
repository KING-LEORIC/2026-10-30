from pathlib import Path
# CSV 데이터 처리를 위한 pandas
import pandas as pd
# PyTorch 핵심 라이브러리
import torch
from torch import nn

# CPU 스레드 개수 고정(실험 재현성 및 환경 차이 최소화)
torch.set_num_threads(1)
# 현재 파이썬 파일이 위치한 디렉터리
base = Path(__file__).resolve().parent
# Dataset, DataLoader 사용
from torch.utils.data import TensorDataset, DataLoader
# 객체 복사용
import copy
# 그래프 저장용 matplotlib
import matplotlib
matplotlib.use("Agg")  # 창을 띄우지 않고 PNG에 저장합니다.
import matplotlib.pyplot as plt

# 결과 저장 폴더 생성
out = base / "outputs"
out.mkdir(exist_ok=True)
# 재현용 기본값 CPU. GPU 실습은 강사 확인 뒤 device 문자열만 cuda:0으로 바꿉니다.
device = torch.device("cpu")
# CSV 파일 로드
df = pd.read_csv(base / "data" / "points.csv")
# train/valid 데이터 분리
train = df[df["split"] == "train"]
valid = df[df["split"] == "valid"]
# 입력 특성(x1, x2)을 텐서로 변환
X_train = torch.tensor(train[["x1", "x2"]].values, dtype=torch.float32)
# 정답 레이블
y_train = torch.tensor(train["label"].values, dtype=torch.long)
# 검증 입력을 device(CPU 또는 GPU)로 이동
X_valid = torch.tensor(valid[["x1", "x2"]].values, dtype=torch.float32).to(device)
# 검증 레이블을 device로 이동
y_valid = torch.tensor(valid["label"].values, dtype=torch.long).to(device)
# 다중 분류용 손실 함수
loss_fn = nn.CrossEntropyLoss()
# 시간 측정용
import time
# batch_size 8, 32, 전체 240의 세 후보를 만드세요.
# 동일 epoch 비교는 동일 갱신 횟수 비교가 아닙니다. total_steps와 elapsed_seconds를 함께 보고합니다.
# >>> S03
# TODO S03: 위 요구사항에 맞게 이 블록만 완성합니다.

# batch size 비교 실험 설정
# - 8  : 작은 배치(가중치 갱신 빈도 높음)
# - 32 : 일반적으로 많이 사용하는 배치 크기
# - 240: 전체 학습 데이터를 한 번에 사용하는 Full Batch Gradient Descent
# - 동일한 30 epoch를 수행하더라도 batch size에 따라 epoch당 iteration 수와 total_steps는 서로 달라짐
# 참고:
# batch size가 작을수록 epoch당 parameter update 횟수가 많아진다.
# 따라서 같은 30 epoch라도 실제 학습량(total_steps)은 서로 다르므로
# experiment_table.csv의 total_steps, elapsed_seconds를 함께 확인한다.
configs = [{'name': 'batch8', 'batch_size': 8}, {'name': 'batch32', 'batch_size': 32}, {'name': 'batch240', 'batch_size': 240}]
# <<< S03
# 실험 결과 요약 저장용
summary = []

# 각 설정(batch size)에 대해 반복 실험
for config in configs:
    # 모델 초기화가 동일하게 되도록 시드 고정
    torch.manual_seed(42)
    # 은닉층 너비(기본 32)
    width = config.get("width", 32)
    # 첫 번째 선형 계층
    layers = [nn.Linear(2, width)]
    # BatchNorm 사용 옵션
    if config.get("bn", False):
        layers.append(nn.BatchNorm1d(width))
    # 첫 번째 은닉 블록
    layers.extend([nn.ReLU(), nn.Dropout(config.get("dropout", 0.0)), nn.Linear(width, width)])
    # 두 번째 BatchNorm 사용 옵션
    if config.get("bn", False):
        layers.append(nn.BatchNorm1d(width))
    # 두 번째 은닉 블록 + 출력층
    layers.extend([nn.ReLU(), nn.Dropout(config.get("dropout", 0.0)), nn.Linear(width, 2)])
    # Sequential 모델 생성 및 device 이동
    model = nn.Sequential(*layers).to(device)
    # 현재 설정의 batch size
    batch_size = config.get("batch_size", 32)
    # 학습 데이터를 mini-batch 단위로 공급하는 DataLoader 생성
    # TensorDataset(X_train, y_train)
    #   -> 입력(x1, x2)과 정답(label)을 하나의 데이터셋으로 묶음
    # batch_size=batch_size
    #   -> config에서 지정한 크기(8, 32, 240)만큼 데이터를 잘라서 제공
    #   -> batch_size가 작을수록 가중치 업데이트 횟수는 많아짐
    # shuffle=True
    #   -> 매 epoch마다 데이터 순서를 섞음
    #   -> 특정 순서에 과적합되는 것을 방지
    # generator=torch.Generator().manual_seed(43)
    #   -> shuffle 결과를 재현 가능하도록 시드 고정
    #   -> 여러 번 실행해도 동일한 데이터 섞기 순서 사용
    # num_workers=0
    #   -> 별도 프로세스를 사용하지 않고 메인 프로세스에서 데이터 로드
    #   -> 실습 환경에서 재현성과 안정성을 높이기 위함
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(43), num_workers=0)
    # AdamW 옵티마이저
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01,
                                   weight_decay=config.get("weight_decay", 0.0))
    # Plateau 발생 시 learning rate 감소
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.5, patience=2, threshold=.001, threshold_mode="abs")
    # epoch별 기록 저장
    history = []
    # GPU 사용 시 정확한 시간 측정을 위해 동기화
    if device.type == "cuda":
        torch.cuda.synchronize()
    # 전체 학습 시작 시간
    started = time.perf_counter()
    # 30 epoch 학습
    for epoch in range(1, 30 + 1):
        # 현재 learning rate 저장
        lr_used = optimizer.param_groups[0]["lr"]
        # 학습 모드
        model.train()
        # mini-batch 손실 누적
        sample_loss = 0.0
        # batch 단위 학습
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            # 순전파
            loss = loss_fn(model(xb), yb)
            # 역전파
            loss.backward()
            # 파라미터 갱신
            optimizer.step()
            # 평균 loss 계산용 누적
            sample_loss += loss.item() * len(yb)
        # 평가 모드
        model.eval()
        with torch.no_grad():
            # 전체 train set 평가 loss
            train_loss = loss_fn(model(X_train.to(device)), y_train.to(device)).item()
            # validation 예측
            logits = model(X_valid)
            # validation loss
            valid_loss = loss_fn(logits, y_valid).item()
            # validation accuracy
            valid_acc = (logits.argmax(1) == y_valid).float().mean().item()
        # 설정된 경우 scheduler 적용
        if config.get("scheduler", False):
            scheduler.step(valid_loss)
        # epoch 결과 저장
        history.append([epoch, sample_loss / len(train), train_loss, valid_loss, valid_acc,
                        lr_used, optimizer.param_groups[0]["lr"], len(loader)])
    # GPU 사용 시 종료 전 동기화
    if device.type == "cuda":
        torch.cuda.synchronize()
    # 총 학습 시간
    elapsed = time.perf_counter() - started
    # history를 DataFrame으로 변환
    table = pd.DataFrame(history, columns=["epoch", "train_loss_step", "train_loss_eval", "valid_loss",
                                           "valid_acc", "lr_used", "lr_next", "iterations"])
    # validation loss가 가장 낮은 epoch 선택
    best = table.loc[table["valid_loss"].idxmin()]
    # 현재 실험 이름
    name = config["name"]
    # epoch 로그 CSV 저장
    table.to_csv(out / f"history_{name}.csv", index=False)
    # 실험 요약정보 저장
    summary.append({**config, "best_epoch": int(best["epoch"]), "best_valid_loss": float(best["valid_loss"]),
                    "last_train_loss_eval": train_loss, "last_valid_loss": valid_loss,
                    "iterations_per_epoch": len(loader), "total_steps": len(loader)*30,
                    "elapsed_seconds": elapsed})
    # learning rate 감소된 epoch 추출
    changes = table[table["lr_next"] < table["lr_used"]]["epoch"].tolist()
    # 콘솔 출력
    print(name, "best epoch:", int(best["epoch"]), "best valid loss:", round(best["valid_loss"], 6),
          "iterations:", len(loader), "lr 감소 epoch:", changes)
    # Loss 곡선 시각화
    plt.figure(figsize=(8, 5))
    plt.plot(table["epoch"], table["train_loss_eval"], label="train (eval)")
    plt.plot(table["epoch"], table["valid_loss"], label="valid")
    # 최적 epoch 표시
    plt.axvline(best["epoch"], linestyle="--", label="best epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    # 그래프 PNG 저장
    plt.savefig(out / f"curves_{name}.png", dpi=150)
    plt.close()
# 실험 결과 요약 테이블 저장
pd.DataFrame(summary).to_csv(out / "experiment_table.csv", index=False)
# 실행 시간 관련 안내 출력
print("시간은 experiment_table.csv에 기록하며 CPU/GPU·실행 부하에 따라 달라집니다.")
