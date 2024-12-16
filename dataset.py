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
