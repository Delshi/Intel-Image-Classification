import os

import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class ImageFolderDataset(Dataset):
    def __init__(self, data_dir, transform=None, is_train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []

        self.classes = sorted(
            [
                d
                for d in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, d))
            ]
        )
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Собираем все изображения с лейблами
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(img_size=150, is_train=True):
    if is_train:
        return T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(0.5),
                T.RandomVerticalFlip(0.25),
                T.RandomRotation(15),
                T.ColorJitter(
                    brightness=0.25, contrast=0.25, saturation=0.25, hue=0.15
                ),
                T.RandomAffine(
                    degrees=(0, 15), translate=(0.15, 0.15), scale=(0.85, 1.15)
                ),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        return T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
