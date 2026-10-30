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

# 실험 결과 저장 폴더 생성
out = base / "outputs"
out.mkdir(exist_ok=True)
# 재현용 기본값 CPU. GPU 실습은 강사 확인 뒤 device 문자열만 cuda:0으로 바꿉니다.
# 재현을 위해 CPU 사용 //GPU 실습 시 device만 cuda:0으로 변경
device = torch.device("cpu")
# 데이터 로드
df = pd.read_csv(base / "data" / "points.csv")
# train / valid 분리
train = df[df["split"] == "train"]
valid = df[df["split"] == "valid"]
# 입력 특성(x1, x2)과 정답(label)을 Tensor로 변환
X_train = torch.tensor(train[["x1", "x2"]].values, dtype=torch.float32)
y_train = torch.tensor(train["label"].values, dtype=torch.long)
# 검증 데이터는 device로 이동
X_valid = torch.tensor(valid[["x1", "x2"]].values, dtype=torch.float32).to(device)
y_valid = torch.tensor(valid["label"].values, dtype=torch.long).to(device)
# 다중 분류용 손실 함수
loss_fn = nn.CrossEntropyLoss()
import time
# baseline, BatchNorm만 추가, weight_decay=0.01만 추가한 세 후보를 만드세요.
# Dropout·BatchNorm·Weight Decay는 목적과 작용 위치가 다른 기법입니다.
# >>> S06
# TODO S06: 위 요구사항에 맞게 이 블록만 완성합니다.

# 비교할 실험 설정
# - baseline : 기본 모델
# - BatchNorm : BatchNorm만 추가
# - wd0.01 : Weight Decay만 추가
configs = [{'name': 'baseline'}, {'name': 'BatchNorm', 'bn': True}, {'name': 'wd0.01', 'weight_decay': 0.01}]
# <<< S06
# 실험 결과 요약 저장
summary = []

for config in configs:
    # 가중치 초기화를 동일하게 맞춰 공정 비교
    torch.manual_seed(42)
    # 은닉층 너비
    width = config.get("width", 32)
    # 첫 번째 은닉층
    layers = [nn.Linear(2, width)]
    # BatchNorm 실험일 때만 추가
    if config.get("bn", False):
        layers.append(nn.BatchNorm1d(width))
    # 은닉층 구성
    layers.extend([nn.ReLU(), nn.Dropout(config.get("dropout", 0.0)), nn.Linear(width, width)])
    # BatchNorm 실험일 때만 추가
    if config.get("bn", False):
        layers.append(nn.BatchNorm1d(width))
    # 출력층 구성
    layers.extend([nn.ReLU(), nn.Dropout(config.get("dropout", 0.0)), nn.Linear(width, 2)])
    # 모델 생성
    model = nn.Sequential(*layers).to(device)
    # 미니배치 크기
    batch_size = config.get("batch_size", 32)
    # DataLoader 생성
    # - shuffle=True로 매 epoch마다 데이터 순서 변경
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(43), num_workers=0)
    # AdamW 옵티마이저
    # - weight_decay는 L2 정규화 역할
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01,
                                   weight_decay=config.get("weight_decay", 0.0))
    # 학습률 스케줄러
    # - validation loss가 개선되지 않으면 학습률 감소
    # - factor=0.5 : 기존 학습률의 절반으로 감소
    # - patience=2 : 2 epoch 동안 개선이 없을 때 적용
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.5, patience=2, threshold=.001, threshold_mode="abs")
    history = []
    # GPU 사용 시 정확한 시간 측정을 위한 동기화
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    # 60 epoch 학습
    for epoch in range(1, 60 + 1):
        # 현재 학습률 저장
        lr_used = optimizer.param_groups[0]["lr"]
        # 학습 모드
        model.train()
        sample_loss = 0.0
        # 미니배치 학습
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            sample_loss += loss.item() * len(yb)
        # 평가 모드
        model.eval()
        with torch.no_grad():
            # 전체 train loss 측정
            train_loss = loss_fn(model(X_train.to(device)), y_train.to(device)).item()
            # validation loss 및 accuracy 측정
            logits = model(X_valid)
            valid_loss = loss_fn(logits, y_valid).item()
            valid_acc = (logits.argmax(1) == y_valid).float().mean().item()
        # scheduler 사용 시 validation loss 기준으로 LR 조정
        if config.get("scheduler", False):
            scheduler.step(valid_loss)
        # epoch별 학습 기록 저장
        history.append([epoch, sample_loss / len(train), train_loss, valid_loss, valid_acc,
                        lr_used, optimizer.param_groups[0]["lr"], len(loader)])
    # GPU 사용 시 종료 전 동기화
    if device.type == "cuda":
        torch.cuda.synchronize()
    # 총 학습 시간
    elapsed = time.perf_counter() - started
    # 학습 이력 DataFrame 생성
    table = pd.DataFrame(history, columns=["epoch", "train_loss_step", "train_loss_eval", "valid_loss",
                                           "valid_acc", "lr_used", "lr_next", "iterations"])
    # 가장 낮은 validation loss를 기록한 epoch
    best = table.loc[table["valid_loss"].idxmin()]
    name = config["name"]
    # 실험별 학습 기록 저장
    table.to_csv(out / f"history_{name}.csv", index=False)
    # 실험 결과 요약 저장
    summary.append({**config, "best_epoch": int(best["epoch"]), "best_valid_loss": float(best["valid_loss"]),
                    "last_train_loss_eval": train_loss, "last_valid_loss": valid_loss,
                    "iterations_per_epoch": len(loader), "total_steps": len(loader)*60,
                    "elapsed_seconds": elapsed})
    # 학습률이 감소한 epoch 확인
    changes = table[table["lr_next"] < table["lr_used"]]["epoch"].tolist()
    # 결과 출력
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
    # 그래프 저장
    plt.savefig(out / f"curves_{name}.png", dpi=150)
    plt.close()
# 실험 결과 요약표 저장
pd.DataFrame(summary).to_csv(out / "experiment_table.csv", index=False)
# 실행 시간은 환경에 따라 달라짐
print("시간은 experiment_table.csv에 기록하며 CPU/GPU·실행 부하에 따라 달라집니다.")
