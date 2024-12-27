"""
AFADDataset는 root_dir 내의 전체 이미지를 탐색한 뒤 무작위 셔플 후 train / val 비율로 나눕니다.
gender는 남성: 0, 여성: 1로 매핑했습니다.
나이는 float(age)로 반환해 회귀에 사용합니다.
"""

import os
import logging
import glob
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AFADDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train', val_split=0.2, seed=42):
        """
        root_dir: AFAD 데이터셋 최상위 폴더 (예: /path/to/AFAD)
        구조: root_dir/age/gender/*.jpg
        split: 'train' or 'val'
        val_split: 학습/검증 데이터 분할 비율
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # 모든 이미지 경로 수집
        # 예: /path/to/AFAD/20/111/*.jpg (20세, 남성)
        #     /path/to/AFAD/20/112/*.jpg (20세, 여성)
        all_image_paths = glob.glob(os.path.join(root_dir, '*', '*', '*.jpg'))
        
        # (img_path, age, gender) 형태의 리스트 생성
        self.samples = []
        for img_path in all_image_paths:
            # 예: /path/to/AFAD/20/111/img_XXX.jpg
            parts = img_path.split(os.sep)
            # parts[-3] = age, parts[-2] = gender
            age_str = parts[-3]   # 예: "20"
            gender_str = parts[-2]  # 예: "111" 또는 "112"
            
            age = int(age_str)
            gender = 0 if gender_str == '111' else 1  # 남성=0, 여성=1
            
            self.samples.append((img_path, age, gender))
        
        # 무작위 셔플 후 학습/검증 분할
        random.seed(seed)
        random.shuffle(self.samples)
        val_size = int(len(self.samples) * val_split)
        
        if split == 'train':
            self.samples = self.samples[val_size:]
        else:  # 'val'
            self.samples = self.samples[:val_size]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, age, gender = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        
        # 나이는 회귀 -> float
        age = float(age)
        # 성별은 분류 -> int
        gender = int(gender)
        
        return image, age, gender


class AFADClassificationDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train', val_split=0.2, seed=42):
        """
        root_dir: AFAD 데이터셋 최상위 폴더 (예: /path/to/AFAD)
        구조: root_dir/age/gender/*.jpg
        split: 'train' or 'val'
        val_split: 학습/검증 데이터 분할 비율
        
        나이 클래스:
        0: 10대 (15-19세)
        1: 20대 (20-29세)
        2: 30대 (30-39세)
        3: 40대 (40-49세)
        4: 50대 이상 (50세 이상)
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # 모든 이미지 경로 수집
        all_image_paths = glob.glob(os.path.join(root_dir, '*', '*', '*.jpg'))
        
        # (img_path, age_class, gender) 형태의 리스트 생성
        self.samples = []
        for img_path in all_image_paths:
            parts = img_path.split(os.sep)
            age = int(parts[-3])
            gender_str = parts[-2]
            
            # 나이를 클래스로 변환
            if age < 20:
                age_class = 0    # 10대
            elif age < 30:
                age_class = 1    # 20대
            elif age < 40:
                age_class = 2    # 30대
            elif age < 50:
                age_class = 3    # 40대
            else:
                age_class = 4    # 50대 이상
            
            gender = 0 if gender_str == '111' else 1  # 남성=0, 여성=1
            
            self.samples.append((img_path, age_class, gender))
        
        # 무작위 셔플 후 학습/검증 분할
        random.seed(seed)
        random.shuffle(self.samples)
        val_size = int(len(self.samples) * val_split)
        
        if split == 'train':
            self.samples = self.samples[val_size:]
        else:  # 'val'
            self.samples = self.samples[:val_size]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, age_class, gender = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        
        # 나이와 성별 모두 분류 문제로 처리
        return image, age_class, gender
    
    
class HS_FADDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train', val_split=0.2, seed=42):
        self.root_dir = root_dir
        self.transform = transform
        
        # 모든 이미지 경로 수집
        all_image_paths = glob.glob(os.path.join(root_dir, '*', '*', '*.jpg'))
        
        # 이미지를 그룹으로 묶기 (같은 인물의 이미지들)
        image_groups = {}
        for img_path in all_image_paths:
            filename = os.path.basename(img_path)
            group_id = '_'.join(filename.split('_')[:-1])
            
            if group_id not in image_groups:
                image_groups[group_id] = []
            image_groups[group_id].append(img_path)
        
        # 각 그룹의 이미지들을 처리
        self.samples = []
        age_counts = {}  # 나이별 카운트를 위한 딕셔너리
        
        for group_paths in image_groups.values():
            group_samples = []
            for img_path in group_paths:
                parts = img_path.split(os.sep)
                age = int(parts[-3])
                gender_str = parts[-2]
                
                # 나이 카운트 업데이트
                if age not in age_counts:
                    age_counts[age] = 0
                age_counts[age] += 1
                
                gender = 0 if gender_str == '111' else 1
                group_samples.append((img_path, age, gender))
            
            self.samples.append(group_samples)
        
        # 나이별 가중치 계산 (샘플 수의 역수를 기반으로)
        max_count = max(age_counts.values())
        self.age_weights = {
            age: max_count / count 
            for age, count in age_counts.items()
        }
        
        logger.info(f"Age weights: {self.age_weights}")
        
        # 그룹 단위로 셔플
        random.seed(seed)
        random.shuffle(self.samples)
        
        # train/val 분할
        val_size = int(len(self.samples) * val_split)
        if split == 'train':
            self.samples = self.samples[val_size:]
        else:  # 'val'
            self.samples = self.samples[:val_size]
        
        # 그룹을 다시 개별 샘플로 평탄화
        self.samples = [sample for group in self.samples for sample in group]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, age, gender = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        
        # 나이를 1-59 범위로 정규화 (기존 10-50 대신)
        normalized_age = (float(age) - 1) / 58.0  # 1-59세 범위를 0-1로 정규화
        
        # 가중치는 loss 계산을 위해 함께 반환
        weight = self.age_weights[age]
        
        return image, normalized_age, gender, weight

    @staticmethod
    def denormalize_age(normalized_age):
        """정규화된 나이값을 원래 스케일로 변환"""
        return normalized_age * 58.0 + 1


class HS_FADGaussianDataset(Dataset):
    def __init__(self, root_dir, age_classes=[10, 20, 30, 40, 50], sigma=5, transform=None, split='train', val_split=0.2, seed=42):
        self.root_dir = root_dir
        self.age_classes = age_classes  # 가능한 나이 클래스
        self.sigma = sigma              # Gaussian 분포의 표준편차
        self.transform = transform
        
        # 모든 이미지 경로 수집
        all_image_paths = glob.glob(os.path.join(root_dir, '*', '*', '*.jpg'))
        
        # 이미지를 그룹으로 묶기 (같은 인물의 이미지들)
        image_groups = {}
        for img_path in all_image_paths:
            filename = os.path.basename(img_path)
            group_id = '_'.join(filename.split('_')[:-1])
            
            if group_id not in image_groups:
                image_groups[group_id] = []
            image_groups[group_id].append(img_path)
        
        # 각 그룹의 이미지들을 처리
        self.samples = []
        age_counts = {}  # 나이별 카운트를 위한 딕셔너리
        
        for group_paths in image_groups.values():
            group_samples = []
            for img_path in group_paths:
                parts = img_path.split(os.sep)
                age = int(parts[-3])
                gender_str = parts[-2]
                
                # 나이 카운트 업데이트
                if age not in age_counts:
                    age_counts[age] = 0
                age_counts[age] += 1
                
                gender = 0 if gender_str == '111' else 1
                group_samples.append((img_path, age, gender))
            
            self.samples.append(group_samples)
        
        # 나이별 가중치 계산 (샘플 수의 역수를 기반으로)
        max_count = max(age_counts.values())
        self.age_weights = {
            age: max_count / count 
            for age, count in age_counts.items()
        }
        
        logger.info(f"Age weights: {self.age_weights}")
        
        # 그룹 단위로 셔플
        random.seed(seed)
        random.shuffle(self.samples)
        
        # train/val 분할
        val_size = int(len(self.samples) * val_split)
        if split == 'train':
            self.samples = self.samples[val_size:]
        else:  # 'val'
            self.samples = self.samples[:val_size]
        
        # 그룹을 다시 개별 샘플로 평탄화
        self.samples = [sample for group in self.samples for sample in group]
        

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        img_path, age, gender = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        
        # Gaussian 분포 라벨 생성
        age_distribution = self.generate_age_distribution(age)

        # 가중치는 loss 계산을 위해 함께 반환
        weight = self.age_weights[age]
        
        return image, age_distribution, gender, weight
    

    def generate_age_distribution(self, age):
        """
        주어진 나이를 기반으로 Gaussian 분포를 생성합니다.

        Args:
            age (int): 실제 나이 라벨

        Returns:
            torch.Tensor: Gaussian 분포 형태의 라벨 벡터
            
        if age == 30:
            tensor([1.1983e-08, 2.6393e-04, 1.0648e-01, 7.8678e-01, 1.0648e-01])
        """
        distribution = np.exp(-0.5 * ((np.array(self.age_classes) - age) / self.sigma) ** 2)
        distribution /= distribution.sum()  # 정규화
        return torch.tensor(distribution, dtype=torch.float32)


# AFAD랑 HS-FAD 데이터를 합친 데이터셋
class CombinedDataset(Dataset):
    def __init__(self, afad_root_dir, hsfad_root_dir, transform=None, split='train', val_split=0.2, seed=42, 
                 max_samples_per_age={
                     10: -1,
                     20: -1,
                     30: -1,
                     40: -1,
                     50: -1
                 }):
        self.transform = transform
        self.samples = []
        
        # AFAD 데이터 처리
        if afad_root_dir:
            # 나이대별로 이미지를 분류
            age_grouped_samples = {10: [], 20: [], 30: [], 40: [], 50: []}
            
            afad_images = glob.glob(os.path.join(afad_root_dir, '*', '*', '*.jpg'))
            for img_path in afad_images:
                parts = img_path.split(os.sep)
                age = int(parts[-3])
                gender_str = parts[-2]
                
                # AFAD 나이를 10단위로 매핑
                if age < 20:
                    normalized_age = 10
                elif age < 30:
                    normalized_age = 20
                elif age < 40:
                    normalized_age = 30
                elif age < 50:
                    normalized_age = 40
                else:
                    normalized_age = 50
                
                gender = 0 if gender_str == '111' else 1
                age_grouped_samples[normalized_age].append((img_path, normalized_age, gender, 'afad'))
            
            # 각 나이대별로 처리
            for age_group in age_grouped_samples:
                samples = age_grouped_samples[age_group]
                random.shuffle(samples)  # 랜덤하게 섞기
                
                # 최대 샘플 수 적용
                if max_samples_per_age[age_group] != -1:
                    samples = samples[:max_samples_per_age[age_group]]
                
                # 각 나이대별로 train/val 분할
                val_size = int(len(samples) * val_split)
                if split == 'train':
                    selected_samples = samples[val_size:]
                else:  # 'val'
                    selected_samples = samples[:val_size]
                
                self.samples.extend(selected_samples)

        # HS-FAD 데이터 처리 (비슷한 방식으로 수정)
        if hsfad_root_dir:
            hsfad_images = glob.glob(os.path.join(hsfad_root_dir, '*', '*', '*.jpg'))
            
            # 나이대별로 그룹화
            age_grouped_samples = {10: [], 20: [], 30: [], 40: [], 50: []}
            
            # 이미지를 그룹으로 묶기
            image_groups = {}
            for img_path in hsfad_images:
                filename = os.path.basename(img_path)
                group_id = '_'.join(filename.split('_')[:-1])
                
                if group_id not in image_groups:
                    image_groups[group_id] = []
                image_groups[group_id].append(img_path)
            
            # 각 그룹의 이미지들을 나이대별로 분류
            for group_paths in image_groups.values():
                for img_path in group_paths:
                    parts = img_path.split(os.sep)
                    age = int(parts[-3])
                    gender_str = parts[-2]
                    gender = 0 if gender_str == '111' else 1
                    age_grouped_samples[age].append((img_path, age, gender, 'hsfad'))
            
            # 각 나이대별로 처리
            for age_group in age_grouped_samples:
                samples = age_grouped_samples[age_group]
                random.shuffle(samples)
                
                # 최대 샘플 수 적용
                if max_samples_per_age[age_group] != -1:
                    samples = samples[:max_samples_per_age[age_group]]
                
                # 각 나이대별로 train/val 분할
                val_size = int(len(samples) * val_split)
                if split == 'train':
                    selected_samples = samples[val_size:]
                else:  # 'val'
                    selected_samples = samples[:val_size]
                
                self.samples.extend(selected_samples)

        # 최종 데이터 셔플
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, age, gender, dataset_type = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        
        # 나이를 0-1 사이로 정규화 (10-50세 범위 기준)
        normalized_age = (float(age) - 10) / 40.0
        
        return image, normalized_age, gender

    @staticmethod
    def denormalize_age(normalized_age):
        """정규화된 나이값을 원래 스케일로 변환"""
        return normalized_age * 40.0 + 10