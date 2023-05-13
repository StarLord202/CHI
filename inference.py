import argparse
import os
import glob
from PIL import Image
import torch
from torch import nn
import torchvision.transforms as transforms
from torchvision.models import resnet34


def get_model():
    model = resnet34()
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.relu = nn.PReLU(64)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(512, 33)
    )
    model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
    model.eval()
    model.cpu()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='Path to directory containing image samples')
    args = parser.parse_args()

    image_paths = glob.glob(os.path.join(args.image_dir, '*.png')) + glob.glob(
        os.path.join(args.image_dir, '*.jpg')) + glob.glob(os.path.join(args.image_dir, '*.jpeg'))

    model = get_model()
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize(mean = 0, std=255.0)
    ])

    for image_path in image_paths:

        class_to_ind = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5',
                        6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B',
                        12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H',
                        18: 'J', 19: 'K', 20: 'L', 21: 'M', 22: 'N', 23: 'P', 24: 'R',
                        25: 'S', 26: 'T', 27: 'U', 28: 'V', 29: 'W', 30: 'X', 31: 'Y', 32: 'Z'}

        image = Image.open(image_path)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        image = image.rotate(-90)
        image = transform(image)
        image = image.unsqueeze(0)
        image.cpu()

        with torch.no_grad():
            output = model(image)
        _, predicted = torch.max(output.data, 1)
        symbol = class_to_ind[predicted.item()]
        ascii_code = ord(symbol)

        print(f"{ascii_code}, {image_path}")

if __name__ == '__main__':
    main()