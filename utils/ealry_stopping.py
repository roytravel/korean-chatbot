class EarlyStopping:
    """
        조기 종료를 위한 모듈
        1. metric(loss or accuracy)를 기준으로 patience회 연속 성능이 좋아지지 않을 경우 종료 사인 반환
        2. 학습과 검증 loss 평균이 threshold 이하면 종료 사인 반환
    """
    def __init__(self, mode='min', patience=5, min_delta=0):
        self.mode = mode # 'min' or 'max'
        self.patience = patience
        self.min_delta = min_delta
        self.loss = float("INF")
        self.patience_limit = patience
        
    def is_stop(self, loss):
        if self.loss + self.min_delta > loss:
            self.loss = loss
            self.patience = 0
        else:
            self.patience += 1
            
        return self.patience >= self.patience_limit