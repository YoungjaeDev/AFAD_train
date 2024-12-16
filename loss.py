# 1. L1Loss (MAE): MSE보다 이상치에 덜 민감
# criterion_age = nn.L1Loss()

# # 2. SmoothL1Loss (Huber Loss): MSE와 MAE의 장점 결합
# criterion_age = nn.SmoothL1Loss(beta=1.0)  # beta로 MSE/MAE 전환 지점 조절

# 3. 나이 예측에 특화된 커스텀 loss도 가능
# class OrdinalRegressionLoss(nn.Module):
#     def __init__(self, alpha=1.0):
#         super().__init__()
#         self.mse = nn.MSELoss()
#         self.l1 = nn.L1Loss()
#         self.alpha = alpha
    
#     def forward(self, pred, target):
#         return self.mse(pred, target) + self.alpha * self.l1(pred, target)