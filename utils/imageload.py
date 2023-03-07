import torchvision.transforms as transforms
from PIL import Image

# 이미지를 읽어서 Grayscale로 변환
image_path = ""
image = Image.open(image_path).convert('L')

# 이미지를 PyTorch Tensor로 변환
transform = transforms.Compose([
    transforms.Resize((256, 256)),   # 이미지를 256x256 크기로 변환
    transforms.ToTensor()            # Tensor로 변환
])
tensor_image = transform(image)