"""
딥러닝 기반 나이 및 성별 예측 모델 학습 프로그램

동작 순서:
1. 설정 파일(config.yaml)에서 학습 파라미터 로드
2. 데이터셋 및 데이터로더 생성
3. 모델 초기화 (MultiTask/Age-only/Gender-only)
4. 손실 함수 및 옵티마이저 설정
5. 학습 루프 실행
   - 학습 데이터로 모델 학습
   - 검증 데이터로 성능 평가
   - 최적 모델 저장
   - wandb로 학습 과정 모니터링
"""

import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
import wandb

from dataset import AFADDataset, HS_FADDataset, CombinedDataset, HS_FADGaussianDataset
from model import MultiTaskModel_KLD, AgeModel, GenderModel
from loss import WeightedMSELoss

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# DataLoader 생성 부분을 다음과 같이 수정
def get_balanced_sampler(dataset, power=0.3):
    """
    데이터셋의 클래스 불균형을 해소하기 위한 샘플러 생성
    power: 가중치 조정 강도 (0~1)
        - 1: 완전한 균형 (기존 방식)
        - 0: 원본 분포 유지
        - 0.5: 중간 정도의 균형
    """
    age_to_idx = {age: idx for idx, age in enumerate(dataset.age_classes)}
    age_labels = [age_to_idx[sample[1]] for sample in dataset.samples]
    
    # 각 클래스별 샘플 개수 계산
    class_counts = np.bincount(age_labels)
    
    # 최대 샘플 수를 기준으로 가중치 계산 (부드러운 버전)
    max_count = max(class_counts)
    
    # numpy의 power 함수 사용하여 가중치를 부드럽게 조정
    weights = (max_count / class_counts) ** power
    
    # # 가중치 정규화 (선택사항)
    # weights = weights / weights.mean()  # 평균이 1이 되도록 정규화
    
    # 각 샘플에 대한 weight 할당
    sample_weights = weights[age_labels]
    
    logger.info(f"Age classes: {dataset.age_classes}")
    logger.info(f"Class counts: {class_counts}")
    logger.info(f"Smoothed class weights (power={power}): {weights}")
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )
    
    return sampler

class DatasetFactory:
    """
    데이터셋 생성을 위한 팩토리 클래스
    AFAD, HS-FAD 등 다양한 데이터셋을 동일한 인터페이스로 생성
    """
    @staticmethod
    def create_dataset(dataset_name, root_dir, transform, split, val_split):
        dataset_map = {
            'AFADDataset': AFADDataset,
            'HS_FADDataset': HS_FADDataset,
            'HS_FADGaussianDataset': HS_FADGaussianDataset,
            'CombinedDataset': CombinedDataset
        }
        
        if dataset_name not in dataset_map:
            raise ValueError(f"Invalid dataset name: {dataset_name}")
            
        dataset_class = dataset_map[dataset_name]
        
        if dataset_name == 'CombinedDataset':
            return dataset_class(
                afad_root_dir=root_dir[0],
                hsfad_root_dir=root_dir[1],
                transform=transform,
                split=split,
                val_split=val_split
            )
        else:
            return dataset_class(
                root_dir=root_dir,
                transform=transform,
                split=split,
                val_split=val_split
            )

class ModelFactory:
    """
    모델 생성을 위한 팩토리 클래스
    단일 작업(나이/성별) 또는 다중 작업 모델을 생성
    """
    @staticmethod
    def create_model(model_type, model_name, dataset_name):
        if model_type == 'age_only':
            logger.info(f"Creating AgeModel with backbone: {model_name}")
            return AgeModel(backbone_name=model_name)
        elif model_type == 'gender_only':
            logger.info(f"Creating GenderModel with backbone: {model_name}")
            return GenderModel(backbone_name=model_name)
        else:
            if dataset_name == 'HS_FADGaussianDataset':
                logger.info(f"Creating MultiTaskModel_KLD with backbone: {model_name}")
                return MultiTaskModel_KLD(backbone_name=model_name)
            
            raise ValueError(f"Invalid dataset name: {dataset_name}")

class Validator:
    """
    모델 검증을 위한 클래스
    나이 예측, 성별 예측, 또는 둘 다를 검증
    """
    def __init__(self, model, criterion_age, criterion_gender, device, age_loss_weight=1.0):
        self.model = model
        self.criterion_age = criterion_age
        self.criterion_gender = criterion_gender
        self.device = device
        self.age_loss_weight = age_loss_weight

    def validate(self, val_loader, model_type='multitask'):
        self.model.eval()
        
        if model_type == 'age_only':
            return self._validate_age_only(val_loader)
        elif model_type == 'gender_only':
            return self._validate_gender_only(val_loader)
        return self._validate_multitask(val_loader)

    def _validate_age_only(self, val_loader):
        val_loss = 0.0
        age_losses_sum = {age: 0.0 for age in val_loader.dataset.age_classes}
        age_counts = {age: 0 for age in val_loader.dataset.age_classes}
        
        with torch.no_grad():
            for images, age, _, weight in val_loader:
                images = images.to(self.device)
                age = age.float().to(self.device)
                weight = weight.to(self.device)
                
                age_out = self.model(images)
                loss = self._compute_age_loss(age_out, age, weight)
                val_loss += loss.item()
                
                # 나이대별 loss 계산
                age_numpy = age.cpu().numpy()
                age_out_numpy = age_out.cpu().numpy()
                
                for idx in range(len(age_numpy)):
                    # 실제 나이 클래스 (가장 높은 확률을 가진 클래스)
                    true_age_idx = np.argmax(age_numpy[idx])
                    true_age = val_loader.dataset.age_classes[true_age_idx]
                    
                    individual_loss = self._compute_age_loss(
                        age_out[idx:idx+1], 
                        age[idx:idx+1], 
                        weight[idx:idx+1]
                    ).item()
                    
                    age_losses_sum[true_age] += individual_loss
                    age_counts[true_age] += 1
        
        # 각 나이대별 평균 loss 계산
        age_specific_losses = {
            age: losses_sum / counts if counts > 0 else 0.0
            for age, (losses_sum, counts) in zip(age_losses_sum.keys(), 
                                            zip(age_losses_sum.values(), 
                                                age_counts.values()))
        }
        # 로깅
        logger.info("Age-specific validation losses:")
        for age, loss in age_specific_losses.items():
            logger.info(f"Age {age}: {loss:.4f}")
        
        return val_loss / len(val_loader), age_specific_losses

    def _validate_gender_only(self, val_loader):
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, _, gender, _ in val_loader:
                images = images.to(self.device)
                gender = gender.to(self.device)
                
                gender_out = self.model(images)
                loss = self.criterion_gender(gender_out, gender)
                val_loss += loss.item()
                
                # 정확도 계산
                _, predicted = torch.max(gender_out.data, 1)
                total += gender.size(0)
                correct += (predicted == gender).sum().item()
        
        accuracy = 100 * correct / total
        logger.info(f'Gender Prediction Accuracy: {accuracy:.2f}%')
        return val_loss / len(val_loader), accuracy

    def _validate_multitask(self, val_loader):
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, age, gender, weight in val_loader:
                images = images.to(self.device)
                age = age.float().to(self.device)
                gender = gender.to(self.device)
                weight = weight.to(self.device)
                
                age_out, gender_out = self.model(images)
                age_loss = self._compute_age_loss(age_out, age, weight)
                gender_loss = self.criterion_gender(gender_out, gender)
                
                loss = self.age_loss_weight * age_loss + gender_loss
                val_loss += loss.item()
                
                # 성별 예측 정확도 계산
                _, predicted = torch.max(gender_out.data, 1)
                total += gender.size(0)
                correct += (predicted == gender).sum().item()
        
        accuracy = 100 * correct / total
        logger.info(f'Gender Prediction Accuracy: {accuracy:.2f}%')
        return val_loss / len(val_loader), accuracy

    def _compute_age_loss(self, age_out, age, weight):
        if isinstance(self.criterion_age, nn.KLDivLoss):
            return self.criterion_age(age_out, age)
        return self.criterion_age(age_out, age, weight)


def create_train_transform(image_size):
    """
    학습용 데이터 증강 및 전처리 파이프라인 생성
    - 크기 조정
    - 무작위 좌우 반전
    - 회전
    - 색상 변형
    - 정규화
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def create_val_transform(image_size):
    """
    검증용 데이터 전처리 파이프라인 생성
    - 크기 조정
    - 정규화
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def create_train_loader(dataset, cfg):
    """
    학습용 데이터로더 생성
    클래스 불균형 해소를 위한 샘플러 옵션 포함
    """
    if cfg['dataset'].get('balanced_sampler', False):
        sampler = get_balanced_sampler(dataset)
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    return DataLoader(
        dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg['training'].get('num_workers', 4),
        pin_memory=True
    )

def create_val_loader(dataset, cfg):
    """
    검증용 데이터로더 생성
    """
    return DataLoader(
        dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['training'].get('num_workers', 4),
        pin_memory=True
    )
    
def main(config_path='config.yaml'):
    """
    메인 학습 함수
    전체적인 학습 과정을 조율
    """
    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create transforms
    train_transform = create_train_transform(cfg['dataset']['image_size'])
    val_transform = create_val_transform(cfg['dataset']['image_size'])
    
    # Create datasets
    train_dataset = DatasetFactory.create_dataset(
        cfg['dataset']['name'],
        cfg['dataset']['root_dir'],
        train_transform,
        'train',
        cfg['dataset']['val_split']
    )
    val_dataset = DatasetFactory.create_dataset(
        cfg['dataset']['name'],
        cfg['dataset']['root_dir'],
        val_transform,
        'val',
        cfg['dataset']['val_split']
    )
    
    # Create data loaders
    train_loader = create_train_loader(train_dataset, cfg)
    val_loader = create_val_loader(val_dataset, cfg)
    
    # Create model and training components
    model = ModelFactory.create_model(
        cfg['model'].get('type', 'multitask'),
        cfg['model']['name'],
        cfg['dataset']['name']
    )
    model = torch.nn.DataParallel(model)
    model.to(device)
    
     # Create loss functions
    if cfg['dataset']['name'] == 'HS_FADGaussianDataset':
        logger.info("Using KLDivLoss for age prediction")
        criterion_age = nn.KLDivLoss(reduction='batchmean')
    else:
        logger.info("Using WeightedMSELoss for age prediction")
        criterion_age = WeightedMSELoss()
    logger.info("Using CrossEntropyLoss for gender prediction")
    criterion_gender = nn.CrossEntropyLoss()
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg['training']['lr'],
        weight_decay=cfg['training'].get('weight_decay', 0)
    )
    
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,      # 학습률을 절반으로 감소
        patience=2,      # 2 epoch 동안 개선이 없으면 lr 감소
        min_lr=1e-8,     # 최소 학습률
        verbose=True     # 학습률 변경 시 출력
    )
    
    # Create validator
    validator = Validator(
        model, 
        criterion_age, 
        criterion_gender, 
        device,
        age_loss_weight=cfg['training'].get('age_loss_weight', 1.0)
    )
    
    # Initialize wandb
    wandb_cfg = cfg.get('logging', {})
    if wandb_cfg.get('use_wandb', False):
        wandb.init(
            project=wandb_cfg['project'],
            entity=wandb_cfg['entity'],
            name=wandb_cfg['run_name'],
            config=cfg
        )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(cfg['training']['epochs']):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, age, gender, weight) in enumerate(train_loader):
            images = images.to(device)
            age = age.float().to(device)
            gender = gender.to(device)
            weight = weight.to(device)
            
            optimizer.zero_grad()
            
            if cfg['model'].get('type', 'multitask') == 'multitask':
                age_out, gender_out = model(images)
                age_loss = criterion_age(age_out, age, weight)
                gender_loss = criterion_gender(gender_out, gender)
                loss = cfg['training'].get('age_loss_weight', 1.0) * age_loss + gender_loss
            elif cfg['model'].get('type') == 'age_only':
                age_out = model(images)
                loss = criterion_age(age_out, age)
            else:  # gender_only
                gender_out = model(images)
                loss = criterion_gender(gender_out, gender)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % cfg['training'].get('log_interval', 10) == 0:
                logger.info(f'Train Epoch: {epoch} [{batch_idx * len(images)}/{len(train_loader.dataset)} '
                          f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        # Validation
        val_results = validator.validate(val_loader, cfg['model'].get('type', 'multitask'))
        
        if cfg['model'].get('type') == 'age_only':
            val_loss, age_losses = val_results
            if wandb_cfg.get('use_wandb', False):
                wandb.log({
                    'train_loss': train_loss / len(train_loader),
                    'val_loss': val_loss,
                    'epoch': epoch,
                    **{f"val_loss_age_{age}": loss for age, loss in age_losses.items()}
                })
                logger.info(f'Epoch {epoch} - Train Loss: {train_loss / len(train_loader):.6f} - Val Loss: {val_loss:.6f}')
        
        else:  # gender_only 또는 multitask
            val_loss, gender_accuracy = val_results
            if wandb_cfg.get('use_wandb', False):
                wandb.log({
                    'train_loss': train_loss / len(train_loader),
                    'val_loss': val_loss,
                    'gender_accuracy': gender_accuracy,
                    'epoch': epoch
                })
                logger.info(f'Epoch {epoch} - Train Loss: {train_loss / len(train_loader):.6f} - Val Loss: {val_loss:.6f} - Gender Accuracy: {gender_accuracy:.2f}%')
            
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(cfg['training']['save_dir'], wandb_cfg['run_name'], f"best_model_epoch_{epoch}.pth")
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            torch.save(model.module.state_dict(), save_path)
            logger.info(f'Saved best model with validation loss: {val_loss:.6f}')

        scheduler.step(val_loss)
    
    if wandb_cfg.get('use_wandb', False):
        wandb.finish()

if __name__ == '__main__':
    """
    프로그램 시작점
    설정 파일 경로를 인자로 받아 학습 시작
    """
    
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()
    
    main(args.config)