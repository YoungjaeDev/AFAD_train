"""
timm.create_model(..., pretrained=True, num_classes=0)로 백본의 마지막 Linear 레이어 제외
나이(회귀): 출력 차원 1 → age_out.shape == (batch_size,)
성별(분류): 출력 차원 2 → gender_out.shape == (batch_size, 2)
"""

import torch
import torch.nn as nn
import timm

class MultiTaskModel(nn.Module):
    def __init__(self, backbone_name='convnext_tiny'):
        super().__init__()
        # timm 모델 로드, 최종 fc 제거(num_classes=0)
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        in_features = self.backbone.num_features
        
        # 나이 회귀용 헤드
        self.age_head = nn.Linear(in_features, 1)
        # 성별 분류용 헤드 (남/여 2 클래스)
        self.gender_head = nn.Linear(in_features, 2)
        
    def forward(self, x):
        feat = self.backbone(x)          # [batch_size, in_features]
        age_out = self.age_head(feat).squeeze(1)     # [batch_size]
        gender_out = self.gender_head(feat)          # [batch_size, 2]
        return age_out, gender_out


class MultiTaskModelV2(nn.Module):
    def __init__(self, backbone_name='convnext_tiny'):
        super().__init__()
        # timm 모델 로드, 최종 fc 제거(num_classes=0)
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        in_features = self.backbone.num_features
        
        # 나이 회귀용 헤드
        self.age_head = nn.Linear(in_features, 5)
        # 성별 분류용 헤드 (남/여 2 클래스)
        self.gender_head = nn.Linear(in_features, 2)
        
    def forward(self, x):
        feat = self.backbone(x)          # [batch_size, in_features]
        age_out = self.age_head(feat)                   # [batch_size, 5]
        gender_out = self.gender_head(feat)          # [batch_size, 2]
        return age_out, gender_out