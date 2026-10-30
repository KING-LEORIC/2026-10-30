from pathlib import Path
import pandas as pd
import torch
from torch import nn

torch.set_num_threads(1)
base = Path(__file__).resolve().parent
from torch.utils.data import TensorDataset, DataLoader
import copy
import matplotlib
matplotlib.use("Agg")  # 창을 띄우지 않고 PNG에 저장합니다.
import matplotlib.pyplot as plt

out = base / "outputs"
out.mkdir(exist_ok=True)
# 재현용 기본값 CPU. GPU 실습은 강사 확인 뒤 device 문자열만 cuda:0으로 바꿉니다.
device = torch.device("cpu")
df = pd.read_csv(base / "data" / "points.csv")
train = df[df["split"] == "train"]
valid = df[df["split"] == "valid"]
X_train = torch.tensor(train[["x1", "x2"]].values, dtype=torch.float32)
y_train = torch.tensor(train["label"].values, dtype=torch.long)
X_valid = torch.tensor(valid[["x1", "x2"]].values, dtype=torch.float32).to(device)
y_valid = torch.tensor(valid["label"].values, dtype=torch.long).to(device)
loss_fn = nn.CrossEntropyLoss()
import time
# 폭 64 모델에서 Dropout만 0.0과 0.3으로 바꾸세요.
# 해당 데이터에서 실제 과적합이 나타나지 않으면 억지로 과적합이라고 쓰지 않습니다.
# >>> S04
# TODO S04: 위 요구사항에 맞게 이 블록만 완성합니다.

# Dropout 효과 비교 실험
# - width=64 : 이전 실험에서 사용한 은닉층 크기를 그대로 유지
# - dropout=0.0 : 드롭아웃을 사용하지 않는 기준(baseline) 모델
# - dropout=0.3 : 학습 시 은닉노드의 30%를 무작위로 비활성화하여 과적합 완화를 시도하는 모델
# - 다른 조건은 동일하게 유지하여 Dropout 적용 여부만 비교
configs = [{'name': 'baseline', 'width': 64, 'dropout': 0.0}, {'name': 'dropout', 'width': 64, 'dropout': 0.3}]
# <<< S04
# 현재 실험 결과를 저장할 리스트
# 각 모델별 최적 epoch, validation loss, 실행 시간 등을 기록
summary = []

for config in configs:
    # 실험 간 재현성 확보
    # - 동일한 초기 가중치로 시작하여 Dropout 효과만 비교
    torch.manual_seed(42)
    # 은닉층 폭(width)
    # - 노드 수가 많을수록 표현력은 증가하지만 과적합 위험도 증가
    width = config.get("width", 32)
    layers = [nn.Linear(2, width)]
    if config.get("bn", False):
        layers.append(nn.BatchNorm1d(width))
    # Dropout(p)
    # - p 비율만큼 뉴런 출력을 무작위 제거
    # - 학습 시 특정 뉴런에 대한 의존을 줄여 일반화 성능 향상을 기대
    layers.extend([nn.ReLU(), nn.Dropout(config.get("dropout", 0.0)), nn.Linear(width, width)])
    if config.get("bn", False):
        layers.append(nn.BatchNorm1d(width))
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
    # AdamW Optimizer
    # - lr=0.01 : 학습률(가중치 갱신 크기)
    # - weight_decay : L2 정규화 강도
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01,
                                   weight_decay=config.get("weight_decay", 0.0))
    # Validation loss 개선이 일정 기간 없으면 학습률 감소
    # - factor=0.5 : 현재 lr의 50%로 감소
    # - patience=2 : 2 epoch 동안 개선이 없을 때 적용
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.5, patience=2, threshold=.001, threshold_mode="abs")
    history = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for epoch in range(1, 80 + 1):
        lr_used = optimizer.param_groups[0]["lr"]
        model.train()
        sample_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            sample_loss += loss.item() * len(yb)
        model.eval()
        with torch.no_grad():
            train_loss = loss_fn(model(X_train.to(device)), y_train.to(device)).item()
            logits = model(X_valid)
            valid_loss = loss_fn(logits, y_valid).item()
            valid_acc = (logits.argmax(1) == y_valid).float().mean().item()
        if config.get("scheduler", False):
            scheduler.step(valid_loss)
        # Epoch별 기록
        # - train_loss_step : 미니배치 평균 학습 loss
        # - train_loss_eval : 전체 train 데이터 평가 loss
        # - valid_loss      : validation loss
        # - valid_acc       : validation accuracy
        # - lr_used         : 현재 epoch에 사용된 learning rate
        # - lr_next         : 다음 epoch에 적용될 learning rate
        # - iterations      : epoch당 배치 반복 횟수
        history.append([epoch, sample_loss / len(train), train_loss, valid_loss, valid_acc,
                        lr_used, optimizer.param_groups[0]["lr"], len(loader)])
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    table = pd.DataFrame(history, columns=["epoch", "train_loss_step", "train_loss_eval", "valid_loss",
                                           "valid_acc", "lr_used", "lr_next", "iterations"])
    best = table.loc[table["valid_loss"].idxmin()]
    name = config["name"]
    table.to_csv(out / f"history_{name}.csv", index=False)
    # 최적 성능 저장
    # - best_epoch : validation loss가 가장 낮았던 epoch
    # - best_valid_loss : 최소 validation loss
    # - last_train_loss_eval : 마지막 epoch의 train loss
    # - last_valid_loss : 마지막 epoch의 validation loss
    # - iterations_per_epoch : epoch당 미니배치 반복 횟수
    # - total_steps : 전체 파라미터 업데이트 횟수
    # - elapsed_seconds : 실험 수행 시간
    summary.append({**config, "best_epoch": int(best["epoch"]), "best_valid_loss": float(best["valid_loss"]),
                    "last_train_loss_eval": train_loss, "last_valid_loss": valid_loss,
                    "iterations_per_epoch": len(loader), "total_steps": len(loader)*80,
                    "elapsed_seconds": elapsed})
    changes = table[table["lr_next"] < table["lr_used"]]["epoch"].tolist()
    print(name, "best epoch:", int(best["epoch"]), "best valid loss:", round(best["valid_loss"], 6),
          "iterations:", len(loader), "lr 감소 epoch:", changes)
    plt.figure(figsize=(8, 5))
    plt.plot(table["epoch"], table["train_loss_eval"], label="train (eval)")
    plt.plot(table["epoch"], table["valid_loss"], label="valid")
    # Loss 곡선 저장
    # - train/valid loss 변화를 시각화하고 최적 epoch 위치를 함께 표시
    plt.axvline(best["epoch"], linestyle="--", label="best epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / f"curves_{name}.png", dpi=150)
    plt.close()
pd.DataFrame(summary).to_csv(out / "experiment_table.csv", index=False)
print("시간은 experiment_table.csv에 기록하며 CPU/GPU·실행 부하에 따라 달라집니다.")
