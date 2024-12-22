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
from model import MultiTaskModel, MultiTaskModel_KLD, AgeModel, GenderModel
from loss import WeightedMSELoss

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# DataLoader 생성 부분을 다음과 같이 수정
def get_balanced_sampler(dataset):
    """
    각 나이 클래스에 대한 WeightedRandomSampler를 생성
    """
    # 각 샘플의 나이 클래스 수집
    age_labels = [sample[1] for sample in dataset.samples]  # [1]은 age_class
    
    # 각 클래스별 샘플 개수 계산
    class_counts = np.bincount(age_labels)
    
    # 각 클래스의 weight 계산 (적은 샘플을 가진 클래스에 더 높은 가중치)
    weights = 1. / class_counts

    # 각 샘플에 대한 weight 할당
    sample_weights = weights[age_labels]
    
    logger.info(f"Class counts: {class_counts} | Class weights: {weights} | Sample weights: {sample_weights}")
    
    # WeightedRandomSampler 생성
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )
    
    return sampler

class DatasetFactory:
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
    @staticmethod
    def create_model(model_type, model_name, dataset_name):
        if model_type == 'age_only':
            return AgeModel(backbone_name=model_name)
        elif model_type == 'gender_only':
            return GenderModel(backbone_name=model_name)
        else:
            if dataset_name == 'HS_FADGaussianDataset':
                return MultiTaskModel_KLD(backbone_name=model_name)
            return MultiTaskModel(backbone_name=model_name)

class Validator:
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
        
        with torch.no_grad():
            for images, age, _, weight in val_loader:
                images = images.to(self.device)
                age = age.float().to(self.device)
                weight = weight.to(self.device)
                
                age_out = self.model(images)
                loss = self._compute_age_loss(age_out, age, weight)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)

    def _validate_gender_only(self, val_loader):
        val_loss = 0.0
        
        with torch.no_grad():
            for images, _, gender, _ in val_loader:
                images = images.to(self.device)
                gender = gender.to(self.device)
                
                gender_out = self.model(images)
                loss = self.criterion_gender(gender_out, gender)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)

    def _validate_multitask(self, val_loader):
        val_loss = 0.0
        
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
        
        return val_loss / len(val_loader)

    def _compute_age_loss(self, age_out, age, weight):
        if isinstance(self.criterion_age, nn.KLDivLoss):
            return self.criterion_age(age_out, age)
        return self.criterion_age(age_out, age, weight)


def create_train_transform(image_size):
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
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def create_train_loader(dataset, cfg):
    if cfg['training'].get('use_balanced_sampler', False):
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
    return DataLoader(
        dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['training'].get('num_workers', 4),
        pin_memory=True
    )
    
def main(config_path='config.yaml'):
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
    
     # Create loss functions
    if cfg['dataset']['name'] == 'HS_FADGaussianDataset':
        criterion_age = nn.KLDivLoss(reduction='batchmean')
    else:
        criterion_age = WeightedMSELoss()
    criterion_gender = nn.CrossEntropyLoss()
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg['training']['learning_rate'],
        weight_decay=cfg['training'].get('weight_decay', 0)
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
    if cfg.get('wandb', {}).get('use_wandb', False):
        wandb.init(
            project=cfg['wandb']['project_name'],
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
                loss = criterion_age(age_out, age, weight)
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
        val_loss = validator.validate(val_loader, cfg['model'].get('type', 'multitask'))
        
        # Logging
        if cfg.get('wandb', {}).get('use_wandb', False):
            wandb.log({
                'train_loss': train_loss / len(train_loader),
                'val_loss': val_loss,
                'epoch': epoch
            })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), cfg['training']['save_path'])
            logger.info(f'Saved best model with validation loss: {val_loss:.6f}')

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()
    
    main(args.config)