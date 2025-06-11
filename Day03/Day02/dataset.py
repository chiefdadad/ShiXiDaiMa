import os
from PIL import Image
from torch.utils import data

class ImageTxtDataset(data.Dataset):
    def __init__(self, txt_path: str, folder_name: str, transform=None):
        self.transform = transform
        self.data_dir = folder_name  # 基础文件夹路径
        self.imgs_path = []
        self.labels = []
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()  # 分割每一行
            img_path = parts[0]  # 图像路径是第一个元素
            label = int(parts[-1])  # 标签是最后一个元素
            self.labels.append(label)
            # 确保图像路径是完整的绝对路径
            full_img_path = os.path.join(self.data_dir, img_path)
            self.imgs_path.append(full_img_path)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, i):
        path, label = self.imgs_path[i], self.labels[i]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label