# infer.py

import os
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from model import self_net  # 确保 model.py 在同一目录下并定义了 self_net 模型
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
from tqdm import tqdm
import random

from OnlineAugment import OnlineAugment

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 设置随机种子以确保可重复性（可选）
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


# 数据增强收集函数（与训练脚本相同）
def collect_image_label_paths(image_dirs, label_dirs, img_extension='.jpg', lbl_extension='.png'):
    image_paths = []
    label_paths = []

    for img_dir, lbl_dir in zip(image_dirs, label_dirs):
        if not os.path.isdir(img_dir):
            logging.warning(f"图像目录不存在: {img_dir}")
            continue
        if not os.path.isdir(lbl_dir):
            logging.warning(f"标签目录不存在: {lbl_dir}")
            continue

        img_names = [f for f in os.listdir(img_dir) if f.lower().endswith(img_extension)]
        for img_name in img_names:
            img_path = os.path.join(img_dir, img_name)
            lbl_name = os.path.splitext(img_name)[0] + lbl_extension
            lbl_path = os.path.join(lbl_dir, lbl_name)
            if os.path.exists(lbl_path):
                image_paths.append(img_path)
                label_paths.append(lbl_path)
            else:
                logging.warning(f"未找到标签文件: {lbl_path} 对应于图像: {img_path}")

    logging.info(f"收集到 {len(image_paths)} 对图像和标签路径。")
    return image_paths, label_paths

# 混淆矩阵计算
def compute_confusion_matrix(preds, labels, num_classes=4):
    with torch.no_grad():
        preds = preds.view(-1)
        labels = labels.view(-1)
        mask = (labels >= 0) & (labels < num_classes)
        preds = preds[mask]
        labels = labels[mask]
        confusion = torch.bincount(
            num_classes * labels + preds, minlength=num_classes**2
        ).reshape(num_classes, num_classes)
    return confusion

# IoU 计算
def compute_iou_from_confusion(confusion_matrix, num_classes=4, ignore_background=True):
    intersection = torch.diag(confusion_matrix)
    ground_truth_set = confusion_matrix.sum(1)
    predicted_set = confusion_matrix.sum(0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / (union + 1e-10)

    if ignore_background:
        IoU_no_background = IoU[1:]
        mean_IoU = IoU_no_background.mean().item()
    else:
        mean_IoU = IoU.mean().item()

    return IoU, mean_IoU


class SteelDefectDataset(Dataset):
    def __init__(self, image_paths, label_paths, transform=None, online_augment=None):
        """
        初始化数据集

        Args:
            image_paths (list): 所有图像路径列表
            label_paths (list): 所有标签路径列表
            transform (albumentations.Compose): 数据增强变换
            online_augment (OnlineAugment): 在线数据增强
        """
        assert len(image_paths) == len(label_paths), "图像路径和标签路径的数量必须相同"
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform
        self.online_augment = online_augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        lbl_path = self.label_paths[idx]
        image = np.array(Image.open(img_path).convert("L"))
        label = np.array(Image.open(lbl_path).convert("L"))

        # 应用在线增强
        if self.online_augment:
            image, label = self.online_augment(image, label)

        # 应用 albumentations 增强
        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask']

        label = label.long()
        return image, label

def main():
    # 已训练模型的路径
    model_path = 'checkpoint/model.pth'  # 根据需要调整

    # 这里假设您有与训练时相同的外部数据集路径
    external_image_dirs = [
        'NEU_Seg-main/images/training',
        'NEU_Seg-main/images/test',
    ]
    external_label_dirs = [
        'NEU_Seg-main/annotations/training',
        'NEU_Seg-main/annotations/test',
    ]

    # 收集外部图像和标签路径
    external_image_paths, external_label_paths = collect_image_label_paths(external_image_dirs, external_label_dirs)

    # 确保有外部数据
    if len(external_image_paths) == 0 or len(external_label_paths) == 0:
        logging.warning("没有收集到任何外部训练图像路径。在线增强可能无法正常工作。")

    # 定义增强概率
    augmentation_probs = {
        1: 0.12125,  # 四张图片拼接
        2: 0.10125,  # 上下拼接
        3: 0.09375,  # 左右拼接
        4: 0.05875,  # 左上角拼接
        5: 0.1875,   # 加椒盐噪声
        6: 0.4375,   # 保持原图
        7: 0.0       # 选择裁剪（训练时设置为0）
    }
    total = sum(augmentation_probs.values())
    augmentation_probs = {k: v / total for k, v in augmentation_probs.items()}

    # 定义在线增强实例
    online_augment = OnlineAugment(
        image_paths=external_image_paths,
        label_paths=external_label_paths,
        augment_prob=1,  # 与训练时相同
        augmentation_probs=augmentation_probs
    )

    # 定义数据增强
    test_transform = A.Compose([
        A.Resize(200, 200),
        # A.HorizontalFlip(p=0.5),
        # A.ImageCompression(quality_lower=70, quality_upper=80, p=0.3),  # 添加JPEG压缩
        A.Normalize(mean=0.445450, std=0.12117627335),
        ToTensorV2(),
    ])

    # 创建数据集和数据加载器
    test_dataset = SteelDefectDataset(
        image_paths=external_image_paths,
        label_paths=external_label_paths,
        transform=test_transform,
        online_augment=online_augment  # 应用在线增强
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # 单样本处理
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 加载模型
    model = self_net(1, 4)  # 根据您的模型定义调整输入输出通道
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location='cpu')
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict, strict=False)
        print(f"加载已训练的模型：{model_path}")
        logging.info(f"加载已训练的模型：{model_path}")
    else:
        print(f"未找到模型文件 {model_path}")
        logging.error(f"未找到模型文件 {model_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    num_classes = 4
    confusion_matrix = torch.zeros(num_classes, num_classes).to(device)


    with torch.no_grad():
        for idx, (image, label) in enumerate(tqdm(test_loader, desc="推理")):
            image = image.to(device)
            label = label.to(device)

            # 进行前向传播
            outputs = model(image)[0]  # 假设模型输出是一个元组，第一个元素是主要输出
            preds = torch.argmax(outputs, dim=1)

            # 更新混淆矩阵
            confusion_matrix += compute_confusion_matrix(preds, label, num_classes=num_classes)

            # 计算单样本 IoU
            sample_confusion = compute_confusion_matrix(preds, label, num_classes=num_classes)
            _, sample_mean_IoU = compute_iou_from_confusion(sample_confusion, num_classes=num_classes, ignore_background=True)


    # 计算总体 IoU
    IoU, mean_IoU = compute_iou_from_confusion(confusion_matrix, num_classes=num_classes, ignore_background=True)
    print(f"总体平均 IoU（排除背景）：{mean_IoU:.4f}")
    logging.info(f"总体平均 IoU（排除背景）：{mean_IoU:.4f}")
    for cls in range(num_classes):
        print(f"类别 {cls} 的 IoU：{IoU[cls]:.4f}")
        logging.info(f"类别 {cls} 的 IoU：{IoU[cls]:.4f}")

if __name__ == "__main__":
    main()
