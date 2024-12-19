import os
import torch
from glob import glob
from tqdm import tqdm
import yaml
from PIL import Image
from torchvision import transforms
from model import MultiTaskModel

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def get_inference_transform(image_size=112):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def inference_single_image(model, image_path, device, transform):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)  # shape: [1, C, H, W]
    
    model.eval()
    with torch.no_grad():
        age_out, gender_out = model(img_tensor)  # age_out: [1], gender_out: [1, 2]
        predicted_age = age_out.item()
        gender_probs = torch.softmax(gender_out, dim=1)
        predicted_gender = torch.argmax(gender_probs, dim=1).item()  # 0=남, 1=여
    return predicted_age, predicted_gender

def denormalize_age(normalized_age):
    """정규화된 나이값을 원래 스케일로 변환"""
    return normalized_age * 40.0 + 10

def main(config_path='config.yaml'):
    config = load_config(config_path)
    model_name = config['model']['name']
    image_size = config['dataset']['image_size']
    test_image_dir = config['test']['image_dir']
    checkpoint_path = config['checkpoint']['path']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델 생성 및 로드
    model = MultiTaskModel(backbone_name=model_name)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    
    # Transform
    transform = get_inference_transform(image_size)
    
    for image_path in tqdm(glob(os.path.join(test_image_dir, '*.*'))):
        predicted_age, predicted_gender = inference_single_image(model, image_path, device, transform)
        predicted_age = denormalize_age(predicted_age)
        
        gender_str = 'Male' if predicted_gender == 0 else 'Female'
        print(f"Inference result - {image_path} Age: {predicted_age:.2f}, Gender: {gender_str}")

if __name__ == '__main__':
    main('config_hsfad.yaml')