"""
AFADDataset는 root_dir 내의 전체 이미지를 탐색한 뒤 무작위 셔플 후 train / val 비율로 나눕니다.
gender는 남성: 0, 여성: 1로 매핑했습니다.
나이는 float(age)로 반환해 회귀에 사용합니다.
"""

import os
import glob
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

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
        
        # 이미지를 그룹으로 묶기
        image_groups = {}
        for img_path in all_image_paths:
            # 파일명에서 그룹 ID 추출
            filename = os.path.basename(img_path)
            group_id = '_'.join(filename.split('_')[:-1])
            
            if group_id not in image_groups:
                image_groups[group_id] = []
            image_groups[group_id].append(img_path)
        
        # 각 그룹의 이미지들을 처리
        self.samples = []
        for group_paths in image_groups.values():
            group_samples = []
            for img_path in group_paths:
                parts = img_path.split(os.sep)
                age = int(parts[-3])
                gender_str = parts[-2]
                
                # 나이를 0-1 사이로 정규화 (10-50세 범위 기준)
                normalized_age = (float(age) - 10) / 40.0
                
                gender = 0 if gender_str == '111' else 1  # 남성=0, 여성=1
                group_samples.append((img_path, normalized_age, gender))
            
            self.samples.append(group_samples)
        
        # 그룹 단위로 셔플
        random.seed(seed)
        random.shuffle(self.samples)
        
        # 그룹 단위로 train/val 분할
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
        
        # age는 이미 정규화된 float 값, gender는 정수
        return image, age, gender

    @staticmethod
    def denormalize_age(normalized_age):
        """정규화된 나이값을 원래 스케일로 변환"""
        return normalized_age * 40.0 + 10
    
    
class CombinedAgeDataset(Dataset):
    def __init__(self, afad_root_dir, hsfad_root_dir, transform=None, split='train', val_split=0.2, seed=42):
        self.transform = transform
        self.samples = []
        
        # AFAD 데이터 처리
        if afad_root_dir:
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
                self.samples.append((img_path, normalized_age, gender, 'afad'))
        
        # HS-FAD 데이터 처리
        if hsfad_root_dir:
            hsfad_images = glob.glob(os.path.join(hsfad_root_dir, '*', '*', '*.jpg'))
            
            # 이미지를 그룹으로 묶기 (HS-FAD 특성)
            image_groups = {}
            for img_path in hsfad_images:
                filename = os.path.basename(img_path)
                group_id = '_'.join(filename.split('_')[:-1])
                
                if group_id not in image_groups:
                    image_groups[group_id] = []
                image_groups[group_id].append(img_path)
            
            # 각 그룹의 이미지들을 처리
            for group_paths in image_groups.values():
                group_samples = []
                for img_path in group_paths:
                    parts = img_path.split(os.sep)
                    age = int(parts[-3])  # HS-FAD는 이미 10단위
                    gender_str = parts[-2]
                    gender = 0 if gender_str == '111' else 1
                    group_samples.append((img_path, age, gender, 'hsfad'))
                self.samples.extend(group_samples)
        
        # 전체 데이터 셔플 (HS-FAD의 그룹 특성은 유지)
        random.seed(seed)
        if hsfad_root_dir:  # HS-FAD 데이터가 있는 경우
            # HS-FAD 데이터는 그룹으로 묶어서 셔플
            hsfad_samples = [s for s in self.samples if s[3] == 'hsfad']
            afad_samples = [s for s in self.samples if s[3] == 'afad']
            
            # HS-FAD 그룹 유지하면서 셔플
            hsfad_groups = {}
            for sample in hsfad_samples:
                group_id = '_'.join(os.path.basename(sample[0]).split('_')[:-1])
                if group_id not in hsfad_groups:
                    hsfad_groups[group_id] = []
                hsfad_groups[group_id].append(sample)
            
            hsfad_groups_list = list(hsfad_groups.values())
            random.shuffle(hsfad_groups_list)
            hsfad_samples = [sample for group in hsfad_groups_list for sample in group]
            
            # AFAD는 개별 셔플
            random.shuffle(afad_samples)
            
            # 데이터 합치기
            self.samples = afad_samples + hsfad_samples
        else:
            random.shuffle(self.samples)
        
        # Train/Val 분할
        val_size = int(len(self.samples) * val_split)
        if split == 'train':
            self.samples = self.samples[val_size:]
        else:  # 'val'
            self.samples = self.samples[:val_size]

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