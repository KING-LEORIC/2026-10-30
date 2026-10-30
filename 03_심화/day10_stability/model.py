"""제공 코드: 선택한 안정화 기법만 구조에 추가합니다."""
# 신경망 레이어 사용
from torch import nn

# config 설정에 따라 모델 생성
def build_model(config):
    # 은닉층 너비
    w = config["width"]
    # 입력층 생성
    layers = [nn.Linear(2, w)]
    # BatchNorm 사용 시 추가
    if config["bn"]:
        layers.append(nn.BatchNorm1d(w))
    # 첫 번째 은닉 블록
    layers.extend([nn.ReLU(), nn.Dropout(config["dropout"]), nn.Linear(w, w)])
    # 두 번째 BatchNorm
    if config["bn"]:
        layers.append(nn.BatchNorm1d(w))
    # 두 번째 은닉 블록 + 출력층
    layers.extend([nn.ReLU(), nn.Dropout(config["dropout"]), nn.Linear(w, 2)])
    # Sequential 모델 반환
    return nn.Sequential(*layers)
