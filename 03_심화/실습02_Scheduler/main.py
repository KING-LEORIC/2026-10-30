from pathlib import Path
import pandas as pd
import torch
from torch import nn

# PyTorch 연산에 사용할 CPU 스레드 수를 1개로 제한 : 실습 환경에 따라 성능 편차를 줄이기 위함
torch.set_num_threads(1)
# 현재 실행 중인 파이썬 파일의 디렉터리 경로
base = Path(__file__).resolve().parent
from torch.utils.data import TensorDataset, DataLoader
import copy
import matplotlib
matplotlib.use("Agg")  # 창을 띄우지 않고 PNG에 저장합니다.
import matplotlib.pyplot as plt

# 결과 저장 디렉터리 생성
out = base / "outputs"
out.mkdir(exist_ok=True)
# 재현용 기본값 CPU. GPU 실습은 강사 확인 뒤 device 문자열만 cuda:0으로 바꿉니다.
device = torch.device("cpu")
# CSV 파일 읽기
df = pd.read_csv(base / "data" / "points.csv")
# train 데이터 분리
train = df[df["split"] == "train"]
# validation 데이터 분리
valid = df[df["split"] == "valid"]
# 입력 변수(x1, x2)를 float32 Tensor로 변환
X_train = torch.tensor(train[["x1", "x2"]].values, dtype=torch.float32)
# 정답 레이블을 long Tensor로 변환
y_train = torch.tensor(train["label"].values, dtype=torch.long)
# validation 입력 데이터를 Tensor로 변환 후 device로 이동
X_valid = torch.tensor(valid[["x1", "x2"]].values, dtype=torch.float32).to(device)
# validation 정답 데이터를 device로 이동
y_valid = torch.tensor(valid["label"].values, dtype=torch.long).to(device)
# 다중 클래스 분류용 손실함수
loss_fn = nn.CrossEntropyLoss()
import time
# 실험 설정
# - fixed : LR 고정
# - plateau : ReduceLROnPlateau 사용
configs = [{'name': 'fixed'}, {'name': 'plateau', 'scheduler': True}]
# 실험 결과 요약 저장용
summary = []

for config in configs:
    # 모델 초기화 재현성을 위해 seed 고정
    torch.manual_seed(42)
    # 은닉층 노드 수
    width = config.get("width", 32)
    # 첫 번째 은닉층
    layers = [nn.Linear(2, width)]
    # BatchNorm 사용 시 추가
    if config.get("bn", False):
        layers.append(nn.BatchNorm1d(width))
    # 첫 번째 활성화층 + Dropout
    layers.extend([nn.ReLU(), nn.Dropout(config.get("dropout", 0.0)), nn.Linear(width, width)])
    # 두 번째 BatchNorm
    if config.get("bn", False):
        layers.append(nn.BatchNorm1d(width))
    # 두 번째 활성화층 + 출력층
    layers.extend([nn.ReLU(), nn.Dropout(config.get("dropout", 0.0)), nn.Linear(width, 2)])
    # 순차 모델 생성
    model = nn.Sequential(*layers).to(device)
    # 미니배치 크기
    batch_size = config.get("batch_size", 32)
    # DataLoader에서 배치 단위로 데이터를 가져올 수 있게 해줌
    # - TensorDataset으로 학습 데이터(X_train, y_train)를 Dataset 객체로 생성
    # - shuffle=True로 매 epoch마다 데이터 순서 섞기
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(43), num_workers=0)
    # AdamW 옵티마이저
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01,
                                   weight_decay=config.get("weight_decay", 0.0))
    # validation loss 개선이 없을 경우 learning rate 감소
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.5, patience=2, threshold=.001, threshold_mode="abs")
    # epoch별 학습 기록 저장
    history = []
    # GPU 사용 시 정확한 시간 측정을 위해 동기화
    if device.type == "cuda":
        torch.cuda.synchronize()
    # 실험 시작 시각
    started = time.perf_counter()
    # 총 60 epoch 학습
    for epoch in range(1, 60 + 1):
        # 현재 epoch에서 실제 사용될 learning rate
        lr_used = optimizer.param_groups[0]["lr"]
        # 학습 모드
        model.train()
        sample_loss = 0.0
        for xb, yb in loader:
            # 배치를 device로 이동
            xb, yb = xb.to(device), yb.to(device)
            # gradient 초기화
            optimizer.zero_grad()
            # 손실 계산 : 모델에 입력 데이터를 넣어 예측값을 계산하고 예측값과 정답(yb)을 이용해 CrossEntropyLoss를 계산
            loss = loss_fn(model(xb), yb)
            # 역전파
            loss.backward()
            # 가중치 업데이트
            optimizer.step()
            # 전체 데이터 기준 평균 loss 계산을 위해 누적
            sample_loss += loss.item() * len(yb)
        # 평가 모드
        model.eval()
        with torch.no_grad():
            # train 전체 데이터 평가 loss
            train_loss = loss_fn(model(X_train.to(device)), y_train.to(device)).item()
            # validation 예측
            logits = model(X_valid)
            # validation loss
            valid_loss = loss_fn(logits, y_valid).item()
            # validation accuracy
            valid_acc = (logits.argmax(1) == y_valid).float().mean().item()
        if config.get("scheduler", False):
            # 현재 epoch의 valid_loss를 scheduler.step(valid_loss)에 전달하세요.
            # optimizer.step은 batch마다, Plateau scheduler.step은 valid 계산 뒤 epoch마다 1번입니다.
            # lr_used는 이번 epoch, lr_next는 다음 epoch에 사용할 값입니다.
            # >>> S02
            # TODO S02: 위 요구사항에 맞게 이 블록만 완성합니다.
            # pass

            # 현재 epoch의 validation loss를 기준으로 학습 상태를 모니터링
            # validation loss가 충분히 감소하지 않는 상태가
            # patience(2 epoch) 이상 지속되면 learning rate를 factor(0.5)만큼 감소
            # optimizer 내부 learning rate 값이 갱신되며 다음 epoch 학습에 사용됨
            scheduler.step(valid_loss)
            # <<< S02
        history.append([epoch, sample_loss / len(train), train_loss, valid_loss, valid_acc,
                        lr_used, optimizer.param_groups[0]["lr"], len(loader)])
    # GPU 사용 시 종료 전 동기화
    if device.type == "cuda":
        torch.cuda.synchronize()
    # 총 학습 시간
    elapsed = time.perf_counter() - started
    # 기록을 DataFrame으로 변환
    table = pd.DataFrame(history, columns=["epoch", "train_loss_step", "train_loss_eval", "valid_loss",
                                           "valid_acc", "lr_used", "lr_next", "iterations"])
    # validation loss가 최소인 epoch 선택
    best = table.loc[table["valid_loss"].idxmin()]
    name = config["name"]
    # 학습 기록 CSV 저장
    table.to_csv(out / f"history_{name}.csv", index=False)
    # 실험 요약 저장
    summary.append({**config, "best_epoch": int(best["epoch"]), "best_valid_loss": float(best["valid_loss"]),
                    "last_train_loss_eval": train_loss, "last_valid_loss": valid_loss,
                    "iterations_per_epoch": len(loader), "total_steps": len(loader)*60,
                    "elapsed_seconds": elapsed})
    # learning rate가 감소한 epoch 찾기
    changes = table[table["lr_next"] < table["lr_used"]]["epoch"].tolist()
    print(name, "best epoch:", int(best["epoch"]), "best valid loss:", round(best["valid_loss"], 6),
          "iterations:", len(loader), "lr 감소 epoch:", changes)
    # loss 그래프 생성
    plt.figure(figsize=(8, 5))
    # train loss
    plt.plot(table["epoch"], table["train_loss_eval"], label="train (eval)")
    # validation loss
    plt.plot(table["epoch"], table["valid_loss"], label="valid")
    # best epoch 표시
    plt.axvline(best["epoch"], linestyle="--", label="best epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    # 그래프 PNG 저장
    plt.savefig(out / f"curves_{name}.png", dpi=150)
    plt.close()
# 모든 실험 결과 요약 CSV 저장
pd.DataFrame(summary).to_csv(out / "experiment_table.csv", index=False)
print("시간은 experiment_table.csv에 기록하며 CPU/GPU·실행 부하에 따라 달라집니다.")
