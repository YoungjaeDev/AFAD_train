import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm

from dataset import AFADDataset
from model import MultiTaskModel

def validate(model, val_loader, criterion_age, criterion_gender, device):
    model.eval()
    val_loss = 0.0
    val_loss_age = 0.0
    val_loss_gender = 0.0
    correct_gender = 0
    total_samples = 0
    
    # MAE 계산을 위한 변수
    age_abs_errors = []
    
    with torch.no_grad():
        for images, age, gender in tqdm(val_loader, desc="Validation", total=len(val_loader)):
            images = images.to(device)
            age = age.to(device)
            gender = gender.to(device)
            
            age_out, gender_out = model(images)
            
            loss_age = criterion_age(age_out, age)
            loss_gender = criterion_gender(gender_out, gender)
            loss = loss_age + loss_gender
            
            # MAE 계산을 위해 절대 오차 저장
            age_abs_error = torch.abs(age_out - age)
            age_abs_errors.extend(age_abs_error.cpu().numpy())
            
            val_loss += loss.item()
            val_loss_age += loss_age.item()
            val_loss_gender += loss_gender.item()
            
            # 성별 분류 정확도 계산
            _, predicted_gender = torch.max(gender_out, dim=1)
            correct_gender += (predicted_gender == gender).sum().item()
            total_samples += gender.size(0)
    
    val_loss /= len(val_loader)
    val_loss_age /= len(val_loader)
    val_loss_gender /= len(val_loader)
    val_acc_gender = correct_gender / total_samples
    
    # MAE 계산
    mae_age = np.mean(age_abs_errors)
    
    return {
        'val_loss': val_loss,
        'val_loss_age': val_loss_age,
        'val_loss_gender': val_loss_gender,
        'val_acc_gender': val_acc_gender,
        'mae_age': mae_age
    }

def main(config_path='config.yaml'):
    # Config 로드
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 설정값 추출
    model_name = cfg['model']['name']
    batch_size = cfg['training']['batch_size']
    root_dir = cfg['dataset']['root_dir']
    image_size = cfg['dataset']['image_size']
    val_split = cfg['dataset']['val_split']
    checkpoint_path = cfg['checkpoint']['path']
    
    # Transform 정의
    val_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]),
    ])
    
    # Dataset & Dataloader
    val_dataset = AFADDataset(
        root_dir=root_dir,
        transform=val_transform,
        split='val',
        val_split=val_split
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델 로드
    model = MultiTaskModel(backbone_name=model_name)
    model = nn.DataParallel(model)
    model.to(device)
    
    model.module.load_state_dict(torch.load(checkpoint_path))
    
    # Loss 함수 정의
    criterion_age = nn.MSELoss()
    criterion_gender = nn.CrossEntropyLoss()
    
    # 검증 실행
    results = validate(model, val_loader, criterion_age, criterion_gender, device)
    
    # 결과 출력
    print("=== Validation Results ===")
    print(f"Loss: {results['val_loss']:.4f}")
    print(f"Age Loss: {results['val_loss_age']:.4f}")
    print(f"Gender Loss: {results['val_loss_gender']:.4f}")
    print(f"Gender Accuracy: {results['val_acc_gender']:.4f}")
    print(f"Age MAE: {results['mae_age']:.4f}")

if __name__ == '__main__':
    main('config.yaml')