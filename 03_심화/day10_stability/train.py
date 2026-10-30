"""학습, Scheduler, Early Stopping, best snapshot을 한 실험으로 묶습니다."""
# state_dict 복사용
import copy
# PyTorch 라이브러리
import torch
# 신경망 모듈
from torch import nn
# Dataset, DataLoader 사용
from torch.utils.data import TensorDataset, DataLoader
# 모델 생성 함수
from model import build_model
# 평가 함수
from evaluate import evaluate

# 하나의 실험(config)에 대한 학습 수행
def fit(config, X, y, Xv, yv, device):
    # 실험 재현성을 위한 시드 고정
    torch.manual_seed(config["seed"])
    # 모델 생성 및 device 이동
    model = build_model(config).to(device)
    # AdamW Optimizer 사용
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    # validation loss 기반 Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.5, patience=3, threshold=1e-4, threshold_mode="abs")
    # DataLoader 셔플 재현성 확보
    generator = torch.Generator().manual_seed(config["seed"]+1)
    # 학습용 DataLoader 생성
    loader = DataLoader(TensorDataset(X, y), batch_size=config["batch_size"], shuffle=True,
                        generator=generator, num_workers=0)
    # 현재까지 가장 좋은 validation loss
    best_loss = float("inf")
    # Early Stopping 기준 loss
    patience_reference = float("inf")
    # 성능 개선이 없는 epoch 수
    bad_epochs = 0
    # 학습 기록 저장용
    history = []
    # Epoch 반복
    for epoch in range(1, config["epochs"]+1):
        # 학습 모드 활성화
        model.train()
        # epoch loss 누적
        total_loss = 0.0
        # 현재 learning rate 기록
        lr_used = optimizer.param_groups[0]["lr"]
        # Mini-Batch 학습
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            # 이전 gradient 제거
            optimizer.zero_grad()
            # Forward + Loss 계산
            loss = nn.functional.cross_entropy(model(xb), yb)
            # 배치 loss의 역전파 후 optimizer로 파라미터를 갱신하세요.
            # loss는 배치 평균입니다. 로그 집계에서는 loss.item()*배치건수를 누적해야 합니다.
            # >>> S07
            # TODO S07: 위 요구사항에 맞게 이 블록만 완성합니다.
            # pass

            # 역전파 수행
            loss.backward()
            # 파라미터 업데이트
            optimizer.step()
            # <<< S07
            # epoch 평균 loss 계산용 누적
            total_loss += loss.item()*len(yb)
        # train 전체 데이터 평가
        train_loss, _, _ = evaluate(model, X, y, device)
        # validation 평가
        valid_loss, valid_acc, _ = evaluate(model, Xv, yv, device)
        # scheduler 업데이트
        scheduler.step(valid_loss)
        # validation loss 최소값 갱신 시
        if valid_loss < best_loss:
            # 관측한 valid loss가 더 작을 때 state_dict를 copy.deepcopy로 보관하세요.
            # 단순 대입은 이후 학습에 따라 바뀌므로 best_state가 최적 epoch를 보존하지 못합니다.
            # 이 best는 관측 최소값이고, patience 재설정용 min_delta 기준은 별도입니다.
            # >>> S08
            # TODO S08: 위 요구사항에 맞게 이 블록만 완성합니다.

            # 최소 loss 기록
            best_loss = valid_loss
            # 최적 epoch 저장
            best_epoch = epoch
            # 최적 모델 가중치 저장
            best_state = copy.deepcopy(model.state_dict())
            # <<< S08
        # min_delta를 넘는 개선이 있을 때만 patience 카운터를 초기화합니다.
        # min_delta 이상 개선된 경우 patience 초기화
        if valid_loss < patience_reference-config["min_delta"]:
            patience_reference = valid_loss
            bad_epochs = 0
        # 개선이 부족한 경우 카운트 증가
        else:
            bad_epochs += 1
        # 학습 정보 기록
        history.append({"epoch":epoch,"train_loss_step":total_loss/len(y),
                        "train_loss_eval":train_loss,"valid_loss":valid_loss,"valid_acc":valid_acc,
                        "lr_used":lr_used,"lr_next":optimizer.param_groups[0]["lr"],"bad_epochs":bad_epochs})
        # Early Stopping
        if bad_epochs >= config["patience"]:
            break
    # 마지막 epoch의 파라미터와 best는 다를 수 있습니다.
    # 최적 epoch의 가중치 복원
    model.load_state_dict(best_state)
    # 복원된 모델 평가
    restored_loss, restored_acc, prediction = evaluate(model, Xv, yv, device)
    # 실험 요약 정보
    summary = {"name":config["name"],"best_epoch":best_epoch,"best_valid_loss":best_loss,
               "restored_valid_loss":restored_loss,"valid_acc_at_best":restored_acc,
               "stop_epoch":epoch,"last_valid_loss":valid_loss,"test_used":False}
    return model, history, summary, prediction
