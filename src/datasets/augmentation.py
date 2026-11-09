import albumentations as A
from albumentations.pytorch import ToTensorV2

MEAN,STD=(0.485,0.456,0.406),(0.229,0.224,0.225)

def train_transform(crop=512):
    return A.Compose([
        A.RandomCrop(crop,crop),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1,contrast_limit=0.1,p=0.5),
        A.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0.02,p=0.3),
        A.Normalize(MEAN,STD),
        ToTensorV2(),
    ])

def val_transform():
    return A.Compose([A.Normalize(mean=MEAN, std=STD), ToTensorV2()])
