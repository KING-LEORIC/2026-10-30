"""제공 평가 코드. train/eval과 gradient 추적은 별개의 상태입니다."""
# PyTorch 라이브러리
import torch
# 신경망 모듈
from torch import nn

# 모델의 loss와 accuracy를 계산하는 평가 함수
def evaluate(model, X, y, device):
    # 평가 전 각 모듈의 train/eval 상태 저장
    modes = [(m, m.training) for m in model.modules()]
    try:
        # 평가 모드로 전환
        model.eval()
        # gradient 계산 비활성화
        with torch.no_grad():
            # 모델 예측 결과 계산
            logits = model(X.to(device))
            # CrossEntropy Loss 계산
            loss = nn.functional.cross_entropy(logits, y.to(device)).item()
            # 가장 높은 확률의 클래스 선택
            prediction = logits.argmax(1).cpu()
        # loss, accuracy, prediction 반환
        return loss, (prediction == y.cpu()).float().mean().item(), prediction
    finally:
        # train(flag)는 자식까지 덮으므로 개별 모드를 직접 복구합니다.
        # 평가 후 원래 train/eval 상태 복구
        for module, mode in modes:
            module.training = mode
