"""실습07: 같은 설정에서 규제 하나씩 바꾸고 valid loss로 후보를 선택합니다."""
from pathlib import Path
import json
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import load_data
from train import fit

torch.set_num_threads(1)
# 실험 결과 저장 경로 설정
base = Path(__file__).resolve().parent
# 출력 파일 저장 폴더
out = base / "outputs"
# 모델 체크포인트 저장 폴더
checkpoint_dir = base / "checkpoints"
# 출력 폴더 생성
out.mkdir(exist_ok=True)
# 체크포인트 폴더 생성
checkpoint_dir.mkdir(exist_ok=True)
# CPU 사용
device = torch.device("cpu")
# train 데이터와 validation 데이터 로드
X, y, Xv, yv, ids = load_data(base / "data" / "points.csv")
# 기본 실험 설정
# - width        : 은닉층 크기
# - bn           : BatchNorm 사용 여부
# - dropout      : Dropout 비율
# - weight_decay : L2 정규화 강도
# - lr           : 학습률
# - batch_size   : 배치 크기
# - epochs       : 최대 epoch 수
# - patience     : Early Stopping 허용 횟수
default = dict(width=32, bn=False, dropout=0.0, weight_decay=0.0, lr=.01,
               batch_size=32, epochs=80, seed=42, patience=8, min_delta=.001)
# 비교할 실험 설정
# - baseline     : 기본 모델
# - dropout      : Dropout(0.3)만 적용
# - batchnorm    : BatchNorm만 적용
# - weight_decay : Weight Decay(0.01)만 적용
configs = [{**default,"name":"baseline"}, {**default,"name":"dropout","dropout":.3},
           {**default,"name":"batchnorm","bn":True}, {**default,"name":"weight_decay","weight_decay":.01}]
# 실험 결과 저장 리스트
rows = []
# 현재 최고 성능 loss
winning_loss = float("inf")
# 모든 설정에 대해 실험 수행
for config in configs:
    # 모델 학습
    model, history, summary, predicted = fit(config, X, y, Xv, yv, device)
    table = pd.DataFrame(history)
    # 학습 이력 CSV 저장
    table.to_csv(out / f"history_{config['name']}.csv", index=False)
    rows.append({**config, **summary})
    # validation loss가 가장 작은 모델 선택
    if summary["best_valid_loss"] < winning_loss:
        winning_loss = summary["best_valid_loss"]
        # 현재 최고 모델 정보 저장
        # - winner = model
        # - best_config = config
        # - winning_history = table
        # - winning_prediction = predicted
        winner, best_config, winning_history, winning_prediction = model, config, table, predicted
    print(summary["name"], "best epoch:",summary["best_epoch"], "stop:",summary["stop_epoch"],
          "best valid loss:",round(summary["best_valid_loss"],6))
# 전체 실험 결과 저장
pd.DataFrame(rows).to_csv(out / "experiment_table.csv",index=False)
# 최고 성능 모델의 설정 저장
(out / "best_config.json").write_text(json.dumps(best_config,ensure_ascii=False,indent=2),encoding="utf-8")
# 선택한 모델의 state_dict를 checkpoints/best_model.pt에 저장하세요.
# 선택 config는 best_config.json에 이미 기록됩니다. 학습 재개용이 아닌 추론 복원용입니다.
# >>> S09
# TODO S09: 위 요구사항에 맞게 이 블록만 완성합니다.
# pass

# 선택된 최고 모델의 가중치(state_dict) 저장
torch.save(model.state_dict(), out / base / "checkpoints" / "best_model.pt")

# <<< S09
# validation 데이터의 실제값/예측값 저장
pd.DataFrame({"id":ids,"actual":yv.tolist(),"predicted":winning_prediction.tolist()}).to_csv(
    out / "validation_predictions.csv",index=False)
# 최적 모델의 학습 곡선 시각화
plt.figure(figsize=(8,5))
# train loss
plt.plot(winning_history["epoch"],winning_history["train_loss_eval"],label="train (eval)")
# validation loss
plt.plot(winning_history["epoch"],winning_history["valid_loss"],label="valid")
# 가장 좋은 epoch 표시
plt.axvline(winning_history.loc[winning_history.valid_loss.idxmin(),"epoch"],linestyle="--",label="best epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(out / "learning_curves.png",dpi=150)
plt.close()
print("선택:",best_config["name"],"/ test 미사용 / validation 재사용에 따른 선택 편향 주의")
