import os
import json
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, image_dir, json_path, class_names=None, transform=None, image_size=(512, 512)):
        """
        :param image_dir: 图像文件夹路径
        :param json_path: 标注JSON文件路径
        :param class_names: 类别列表，例如 ["颅骨光环", "大脑镰", "大脑实质"]
        :param transform: 图像增强方法（可选）
        :param image_size: 输出图像和 mask 的尺寸（H, W）
        """
        self.image_dir = image_dir
        self.image_size = image_size
        self.transform = transform
        self.annotations = json.load(open(json_path, 'r', encoding='utf-8'))['annotations']
        # self.image_names = list(self.annotations.keys())
        self.image_names = [
            os.path.basename(p)
            for ext in ['*.jpg', '*.png']
            for p in glob.glob(os.path.join(image_dir, ext))
            if os.path.basename(p) in self.annotations
        ]
        # 类别名到channel索引的映射
        if class_names is None:
            self.class_names = sorted(list({ann['name'] for v in self.annotations.values() for ann in v['annotations']}))
        else:
            self.class_names = class_names

        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        print(self.class_to_idx)

    def __len__(self):
        return len(self.image_names)

    def _parse_polygon(self, vertex_str):
        # 将字符串解析为 [(x1,y1), (x2,y2), ...]
        # points = [tuple(map(int, p.strip().split(','))) for p in vertex_str.split(';') if p.strip()]
        points = [tuple(map(lambda x: int(float(x)), p.strip().split(','))) for p in vertex_str.split(';') if p.strip()]
        return points

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)

        # 读取图像
        image = Image.open(image_path).convert('RGB')
        original_w, original_h = image.size

        # --- Step 1: Pad to square ---
        pad_size = max(original_w, original_h)
        delta_w = pad_size - original_w
        delta_h = pad_size - original_h
        padding = (
        delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)  # (left, top, right, bottom)

        image_padded = ImageOps.expand(image, padding, fill=0)

        # --- Step 2: Resize to target size ---
        image_resized = image_padded.resize(self.image_size)
        image_tensor = transforms.ToTensor()(image_resized)  # [3, H, W]

        # --- Step 3: Init mask ---
        num_classes = len(self.class_names)
        masks = np.zeros((num_classes, *self.image_size), dtype=np.uint8)

        for ann in self.annotations[image_name]['annotations']:
            class_name = ann['name']
            if class_name not in self.class_to_idx:
                continue
            class_idx = self.class_to_idx[class_name]
            polygon = self._parse_polygon(ann['vertex'])  # 原图坐标

            # --- Step 4: Pad polygon coordinates ---
            polygon_padded = [
                (x + padding[0], y + padding[1]) for x, y in polygon
            ]

            # --- Step 5: Scale polygon to resized image ---
            scale_x = self.image_size[1] / pad_size
            scale_y = self.image_size[0] / pad_size
            polygon_scaled = [
                (int(x * scale_x), int(y * scale_y)) for x, y in polygon_padded
            ]

            # --- Step 6: Draw polygon to class-specific mask ---
            mask_img = Image.new('L', self.image_size, 0)
            ImageDraw.Draw(mask_img).polygon(polygon_scaled, outline=1, fill=1)
            masks[class_idx] = np.maximum(masks[class_idx], np.array(mask_img, dtype=np.uint8))

        mask_tensor = torch.from_numpy(masks).float()

        # --- Step 7: Optional transforms ---
        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, mask_tensor
