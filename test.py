"""
딥러닝 기반 나이 및 성별 예측 모델 테스트 프로그램

동작 순서:
1. 설정 파일(config.yaml)에서 테스트 파라미터 로드
2. 모델 초기화 및 가중치 로드
3. 이미지 전처리 변환 설정
4. 테스트 이미지에 대한 추론 실행
"""

import os
import torch
from glob import glob
from tqdm import tqdm
import yaml
from PIL import Image
from torchvision import transforms
import logging
from train import ModelFactory  # train.py에서 ModelFactory 임포트

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageTransform:
    """
    이미지 전처리를 위한 클래스
    """
    @staticmethod
    def get_transform(image_size=112):
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])


class SeparatePredictor:
    """
    나이와 성별 예측을 위한 개별 모델 사용 클래스
    """
    def __init__(self, age_model, gender_model, device, transform):
        self.age_model = age_model
        self.gender_model = gender_model
        self.device = device
        self.transform = transform
        self.age_classes = [10, 20, 30, 40, 50]  # HS-FAD 데이터셋의 나이 클래스

    def predict_single_image(self, image_path):
        """단일 이미지에 대한 나이와 성별 예측"""
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        self.age_model.eval()
        self.gender_model.eval()
        
        with torch.no_grad():
            # 나이 예측
            age_output = self.age_model(img_tensor)
            predicted_age = self.age_classes[torch.argmax(age_output, dim=1).item()]
    
            # 성별 예측
            gender_output = self.gender_model(img_tensor)
            gender_probs = torch.softmax(gender_output, dim=1)
            predicted_gender = torch.argmax(gender_probs, dim=1).item()
                    
        return predicted_age, predicted_gender

    @staticmethod
    def denormalize_age(normalized_age):
        """정규화된 나이값을 원래 스케일로 변환"""
        return normalized_age * 40.0 + 10


def load_config(config_path='config.yaml'):
    """설정 파일 로드"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
    
def main(config_path='config.yaml'):
    """
    메인 테스트 함수
    """
    # 설정 로드
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 나이 예측 모델 생성 및 로드
    age_model = ModelFactory.create_model(
        config['checkpoint']['age_model']['type'],
        config['model']['name'],
        config['dataset']['name']
    )
    age_model.load_state_dict(torch.load(config['checkpoint']['age_model']['path'], map_location=device))
    age_model.to(device)
    logger.info(f"Age model loaded from {config['checkpoint']['age_model']['path']}")

    # 성별 예측 모델 생성 및 로드
    gender_model = ModelFactory.create_model(
        config['checkpoint']['gender_model']['type'],
        config['model']['name'],
        config['dataset']['name']
    )
    gender_model.load_state_dict(torch.load(config['checkpoint']['gender_model']['path'], map_location=device))
    gender_model.to(device)
    logger.info(f"Gender model loaded from {config['checkpoint']['gender_model']['path']}")
    
    # 이미지 변환 및 예측기 초기화
    transform = ImageTransform.get_transform(config['dataset']['image_size'])
    predictor = SeparatePredictor(age_model, gender_model, device, transform)
    
    # 테스트 이미지 처리
    test_images = glob(os.path.join(config['test']['image_dir'], '*.*'))
    logger.info(f"Found {len(test_images)} images to process")
    
    for image_path in tqdm(test_images, desc="Processing images"):
        predicted_age, predicted_gender = predictor.predict_single_image(image_path)
        
        # 결과 출력
        result_str = f"Inference result - {os.path.basename(image_path)}"
        if predicted_age is not None:
            result_str += f" Age: {predicted_age:.1f}"
        if predicted_gender is not None:
            gender_str = 'Male' if predicted_gender == 0 else 'Female'
            result_str += f" Gender: {gender_str}"
        
        logger.info(result_str)
        
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='설정 파일 경로')
    args = parser.parse_args()
    
    main(args.config)