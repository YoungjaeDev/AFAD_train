import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
import wandb

from dataset import AFADDataset, HS_FADDataset, CombinedAgeDataset
from model import MultiTaskModel
from loss import WeightedMSELoss

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
    
    # WeightedRandomSampler 생성
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )
    
    return sampler

def main(config_path='config.yaml'):
    # -------------------
    # 1) Config 로드
    # -------------------
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    model_name = cfg['model']['name']
    epochs = cfg['training']['epochs']
    batch_size = cfg['training']['batch_size']
    lr = cfg['training']['lr']
    weight_decay = cfg['training']['weight_decay']
    log_interval = cfg['training']['log_interval']
    age_loss_weight = cfg['training']['age_loss_weight']
    
    root_dir = cfg['dataset']['root_dir']
    image_size = cfg['dataset']['image_size']
    val_split = cfg['dataset']['val_split']
    
    use_wandb = cfg['logging']['use_wandb']
    wandb_project = cfg['logging']['project']
    wandb_entity = cfg['logging']['entity']
    run_name = cfg['logging']['run_name']
    
    save_dir = os.path.join(cfg['training']['save_dir'], run_name)
    # -------------------
    # 2) wandb init
    # -------------------
    if use_wandb:
        wandb.init(project=wandb_project, entity=wandb_entity, name=run_name, config=cfg)
    
    # -------------------
    # 3) Transform 정의 (다양한 augmentation)
    # -------------------
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    # -------------------
    # 4) Dataset & Dataloader
    # -------------------
    dataset_name = cfg['dataset']['name']
    if dataset_name == 'AFADDataset':
        train_dataset = AFADDataset(
            root_dir=root_dir,
            transform=train_transform,
            split='train',
            val_split=val_split
        )
        val_dataset = AFADDataset(
            root_dir=root_dir,
            transform=val_transform,
            split='val',
            val_split=val_split
        )
    elif dataset_name == 'HS_FADDataset':
        train_dataset = HS_FADDataset(
            root_dir=root_dir,
            transform=train_transform,
            split='train',
            val_split=val_split
        )
        val_dataset = HS_FADDataset(
            root_dir=root_dir,
            transform=val_transform,
            split='val',
            val_split=val_split
        )
    elif dataset_name == 'CombinedAgeDataset':
        train_dataset = CombinedAgeDataset(
            afad_root_dir=root_dir[0],
            hsfad_root_dir=root_dir[1],
            transform=train_transform,
            split='train',
            val_split=val_split
        )
        val_dataset = CombinedAgeDataset(
            afad_root_dir=root_dir[0],
            hsfad_root_dir=root_dir[1],
            transform=val_transform,
            split='val',
            val_split=val_split
        )
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
    
    if cfg['dataset'].get('balanced_sampler', False):
        train_sampler = get_balanced_sampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # -------------------
    # 5) 모델, 손실함수, 옵티마이저 정의
    # -------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = MultiTaskModel(backbone_name=model_name)
    model = torch.nn.DataParallel(model)  # 간단한 GPU 병렬화
    model.to(device)
    
    # 나이(분류) -> MSE, 성별(분류) -> CrossEntropyLoss
    # criterion_age = nn.MSELoss()
    criterion_age = WeightedMSELoss()
    criterion_gender = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    
    # 스케쥴러...
    # 1. CosineAnnealingLR: 주기적으로 lr을 조정
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, 
    #     T_max=epochs,  # 한 주기의 길이
    #     eta_min=1e-6   # 최소 lr
    # )

    # # 2. ReduceLROnPlateau: validation loss가 개선되지 않을 때 lr 감소
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode='min',
    #     factor=0.1,    # lr 감소 비율
    #     patience=3,    # 몇 epoch 동안 개선이 없을 때 감소시킬지
    #     min_lr=1e-6
    # )

    # # 3. OneCycleLR: 처음에 lr을 증가시켰다가 점차 감소
    # scheduler = optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=lr,     # 최대 lr
    #     epochs=epochs,
    #     steps_per_epoch=len(train_loader)
    # )

    # ReduceLROnPlateau의 경우
    # scheduler.step(val_loss)  # validation 후

    # # 다른 스케줄러들
    # scheduler.step()  # epoch 끝날 때

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,      # 학습률을 절반으로 감소
        patience=2,      # 2 epoch 동안 개선이 없으면 lr 감소
        min_lr=1e-6,     # 최소 학습률
        verbose=True     # 학습률 변경 시 출력
    )

    # 저장 폴더 생성
    os.makedirs(save_dir, exist_ok=True)

    # -------------------
    # 6) 학습/검증 루프
    # -------------------
    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(epochs):
        print(f"=== Training Epoch {epoch+1}/{epochs} ===")
        model.train()
        running_loss = 0.0
        
        for step, (images, age, gender, weight) in enumerate(train_loader):
            images = images.to(device)
            age = age.float().to(device)
            gender = gender.long().to(device)
            weight = weight.to(device)
            
            optimizer.zero_grad()
            age_out, gender_out = model(images)
            
            loss_age = criterion_age(age_out, age, weight) * age_loss_weight
            loss_gender = criterion_gender(gender_out, gender)
            # loss = loss_age + loss_gender
            loss = loss_age + loss_gender
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            global_step += 1
            
            if (step + 1) % log_interval == 0:
                avg_loss = running_loss / log_interval
                print(f"[Epoch {epoch+1}/{epochs}] Step {step+1}/{len(train_loader)} | Loss: {avg_loss:.4f} | Age Loss: {loss_age.item():.4f} | Gender Loss: {loss_gender.item():.4f}")
                
                if use_wandb:
                    wandb.log({
                        'train_loss': avg_loss,
                        'train_loss_age': loss_age.item(),
                        'train_loss_gender': loss_gender.item(),
                        'epoch': epoch+1
                    }, step=global_step)
                
                running_loss = 0.0
        
        # -------------------
        # Validation
        # -------------------
        val_loss, val_loss_age, val_loss_gender, val_acc_gender = validate(model, val_loader, criterion_age, criterion_gender, device, age_loss_weight)
        
        print(f"=== Validation Epoch {epoch+1}/{epochs} ===")
        print(f"Val Loss: {val_loss:.4f} | Age Loss: {val_loss_age:.4f} | Gender Loss: {val_loss_gender:.4f} | Gender Acc: {val_acc_gender:.4f}")

        if use_wandb:
            wandb.log({
                'val_loss': val_loss,
                'val_loss_age': val_loss_age,
                'val_loss_gender': val_loss_gender,
                'val_acc_gender': val_acc_gender,
                'epoch': epoch+1
            }, step=global_step)
        
        # -------------------
        # 모델 저장 (best 모델)
        # -------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(save_dir, f"best_model_epoch_{epoch+1}.pth")
            torch.save(model.module.state_dict(), save_path)
            print(f"Best model saved at epoch {epoch+1} with val_loss {val_loss:.4f}")

        # epoch 끝날 때마다 scheduler step
        scheduler.step(val_loss)

    # 최종 모델도 저장
    save_final_path = os.path.join(save_dir, f"final_model_epoch_{epochs}.pth")
    torch.save(model.module.state_dict(), save_final_path)
    print(f"Final model saved at: {save_final_path}")

    if use_wandb:
        wandb.finish()

def validate(model, val_loader, criterion_age, criterion_gender, device, age_loss_weight):
    model.eval()
    val_loss = 0.0
    val_loss_age = 0.0
    val_loss_gender = 0.0
    correct_gender = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, age, gender, weight in val_loader:
            images = images.to(device)
            age = age.float().to(device)
            gender = gender.long().to(device)
            weight = weight.to(device)
            
            age_out, gender_out = model(images)
            
            loss_age = criterion_age(age_out, age, weight) * age_loss_weight
            loss_gender = criterion_gender(gender_out, gender)
            loss = loss_age + loss_gender
            
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
    
    return val_loss, val_loss_age, val_loss_gender, val_acc_gender
        

if __name__ == '__main__':
    main('config_hsfad.yaml')