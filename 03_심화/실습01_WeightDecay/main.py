from pathlib import Path
import pandas as pd
import torch
from torch import nn

# PyTorch가 사용하는 CPU 스레드 수를 1개로 제한
torch.set_num_threads(1)
# 현재 실행 중인 파이썬 파일의 위치
base = Path(__file__).resolve().parent
from torch.utils.data import TensorDataset, DataLoader
import copy
import matplotlib
matplotlib.use("Agg")  # 창을 띄우지 않고 PNG에 저장합니다.
import matplotlib.pyplot as plt

# 결과 저장 폴더 생성
out = base / "outputs"
out.mkdir(exist_ok=True)
# 재현용 기본값 CPU. GPU 실습은 강사 확인 뒤 device 문자열만 cuda:0으로 바꿉니다.
device = torch.device("cpu")
# 데이터 읽기
df = pd.read_csv(base / "data" / "points.csv")
# train/valid 분리
train = df[df["split"] == "train"]
valid = df[df["split"] == "valid"]
# 입력 특성(x1,x2)을 FloatTensor로 변환
X_train = torch.tensor(train[["x1", "x2"]].values, dtype=torch.float32)
# 정답(label)을 LongTensor로 변환
y_train = torch.tensor(train["label"].values, dtype=torch.long)
# 검증 데이터 생성 후 device(CPU/GPU)로 이동
X_valid = torch.tensor(valid[["x1", "x2"]].values, dtype=torch.float32).to(device)
y_valid = torch.tensor(valid["label"].values, dtype=torch.long).to(device)
# 다중분류용 손실함수
loss_fn = nn.CrossEntropyLoss()
import time
# AdamW의 weight_decay만 0과 0.01로 바꾸세요.
# 같은 seed·split·lr·batch size·epoch에서 train_eval/valid loss를 비교합니다.
# >>> S01
# TODO S01: 위 요구사항에 맞게 이 블록만 완성합니다.

# weight decay 효과 비교를 위해 두 실험을 정의
# - 정규화 미적용
# - L2 정규화 적용
configs = [{'name': 'wd0', 'weight_decay': 0.0}, {'name': 'wd1', 'weight_decay': 0.01}]
# <<< S01
# 실험 결과 요약 저장용 리스트
summary = []

for config in configs:
    # 모델 초기화를 항상 동일하게 만들기 위한 시드 고정
    torch.manual_seed(42)
    # 은닉층 너비 (기본 32)
    width = config.get("width", 32)
    # 첫 번째 linear layer
    layers = [nn.Linear(2, width)]
    # BatchNorm 사용 시 추가
    if config.get("bn", False):
        layers.append(nn.BatchNorm1d(width))
    # 첫 번째 은닉 블록
    layers.extend([nn.ReLU(), nn.Dropout(config.get("dropout", 0.0)), nn.Linear(width, width)])
    # BatchNorm 사용 시 추가
    if config.get("bn", False):
        layers.append(nn.BatchNorm1d(width))
    # 두 번째 은닉 블록 + 출력층
    layers.extend([nn.ReLU(), nn.Dropout(config.get("dropout", 0.0)), nn.Linear(width, 2)])
    # Sequential 모델 생성
    model = nn.Sequential(*layers).to(device)
    # 배치 크기
    batch_size = config.get("batch_size", 32)
    # DataLoader 생성
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(43), num_workers=0)
    # AdamW 옵티마이저
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01,
                                   weight_decay=config.get("weight_decay", 0.0))
    # validation loss가 개선되지 않으면 lr 감소
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.5, patience=2, threshold=.001, threshold_mode="abs")
    # epoch별 기록 저장
    history = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    # 실험 시작 시간 측정
    started = time.perf_counter()
    # 60 epoch 학습
    for epoch in range(1, 60 + 1):
        # 현재 learning rate 저장
        lr_used = optimizer.param_groups[0]["lr"]
        model.train()
        # mini-batch 학습 손실 누적
        sample_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            # 순전파
            loss = loss_fn(model(xb), yb)
            # 역전파
            loss.backward()
            # 가중치 업데이트
            optimizer.step()
            sample_loss += loss.item() * len(yb)
        # 평가 모드
        model.eval()
        with torch.no_grad():
            # 전체 train 데이터 평가용 loss
            train_loss = loss_fn(model(X_train.to(device)), y_train.to(device)).item()
            # validation 예측
            logits = model(X_valid)
            # validation loss
            valid_loss = loss_fn(logits, y_valid).item()
            # validation accuracy
            valid_acc = (logits.argmax(1) == y_valid).float().mean().item()
        # scheduler 사용 시 lr 조정
        if config.get("scheduler", False):
            scheduler.step(valid_loss)
        # epoch 결과 저장
        history.append([epoch, sample_loss / len(train), train_loss, valid_loss, valid_acc,
                        lr_used, optimizer.param_groups[0]["lr"], len(loader)])
    if device.type == "cuda":
        torch.cuda.synchronize()
    # 전체 실험 시간
    elapsed = time.perf_counter() - started
    # 기록을 DataFrame으로 변환
    table = pd.DataFrame(history, columns=["epoch", "train_loss_step", "train_loss_eval", "valid_loss",
                                           "valid_acc", "lr_used", "lr_next", "iterations"])
    # valid loss가 최소인 epoch 선택
    best = table.loc[table["valid_loss"].idxmin()]
    name = config["name"]
    # 학습 이력 저장
    table.to_csv(out / f"history_{name}.csv", index=False)
    # 요약 결과 저장
    summary.append({**config, "best_epoch": int(best["epoch"]), "best_valid_loss": float(best["valid_loss"]),
                    "last_train_loss_eval": train_loss, "last_valid_loss": valid_loss,
                    "iterations_per_epoch": len(loader), "total_steps": len(loader)*60,
                    "elapsed_seconds": elapsed})
    # learning rate 감소가 발생한 epoch 추출
    changes = table[table["lr_next"] < table["lr_used"]]["epoch"].tolist()
    print(name, "best epoch:", int(best["epoch"]), "best valid loss:", round(best["valid_loss"], 6),
          "iterations:", len(loader), "lr 감소 epoch:", changes)
    # Loss 곡선 그리기
    plt.figure(figsize=(8, 5))
    plt.plot(table["epoch"], table["train_loss_eval"], label="train (eval)")
    plt.plot(table["epoch"], table["valid_loss"], label="valid")
    # 최적 epoch 위치 표시
    plt.axvline(best["epoch"], linestyle="--", label="best epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    # 그래프 저장
    plt.savefig(out / f"curves_{name}.png", dpi=150)
    plt.close()
# 모든 실험 결과 요약 저장
pd.DataFrame(summary).to_csv(out / "experiment_table.csv", index=False)
print("시간은 experiment_table.csv에 기록하며 CPU/GPU·실행 부하에 따라 달라집니다.")
