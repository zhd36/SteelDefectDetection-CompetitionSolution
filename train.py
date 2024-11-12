# train_combined_modified.py

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from datetime import datetime
from model import self_net  # 确保 model.py 在同一目录下并定义了 self_net 模型
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image,ImageEnhance
import numpy as np
import logging
from tqdm import tqdm
import heapq
from sklearn.model_selection import KFold

from lovasz_softmax import lovasz_softmax  # 确保已安装 lovasz_softmax 库
from OnlineAugment import OnlineAugment

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def collect_image_label_paths(image_dirs, label_dirs, img_extension='.jpg', lbl_extension='.png'):
    """
    从多个图像和标签目录中收集图像和标签路径。

    Args:
        image_dirs (list): 图像目录列表。
        label_dirs (list): 标签目录列表。
        img_extension (str): 图像文件扩展名。
        lbl_extension (str): 标签文件扩展名。

    Returns:
        list, list: 图像路径列表和标签路径列表。
    """
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

        # 应用在线增强（仅在训练集）
        if self.online_augment:
            image, label = self.online_augment(image, label)

        # 应用 albumentations 增强
        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask']

        label = label.long()
        return image, label

# 数据增强
train_transform = A.Compose([
    A.Resize(200, 200),
    A.HorizontalFlip(p=0.5),
    #A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.35, p=0.01),  # 亮度和对比度调整
    A.ImageCompression(quality_lower=70, quality_upper=80, p=0.3),  # 添加JPEG压缩
    A.Normalize(mean=0.445450, std=0.12117627335),
    ToTensorV2(),
])


val_transform = A.Compose([
    A.Resize(200, 200),
    A.Normalize(mean=0.445450, std=0.12117627335),
    ToTensorV2(),
])


# 混淆矩阵计算
def compute_confusion_matrix(preds, labels, num_classes=4):
    with torch.no_grad():
        preds = preds.view(-1)
        labels = labels.view(-1)
        mask = (labels >= 0) & (labels < num_classes)
        preds = preds[mask]
        labels = labels[mask]
        confusion = torch.bincount(num_classes * labels + preds, minlength=num_classes**2).reshape(num_classes, num_classes)
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

def validate_model(model, val_loader, criterion, writer=None, epoch=None, fold=1, num_classes=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    running_loss = 0.0
    confusion_matrix = torch.zeros(num_classes, num_classes).to(device)

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Fold {fold} 验证"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            # 主输出是第一个输出
            main_output = outputs[0] if isinstance(outputs, (list, tuple)) else outputs

            # 计算主输出损失
            ce_loss = criterion(main_output, labels)
            probas = torch.softmax(main_output, dim=1)
            lovasz_loss = lovasz_softmax(probas, labels, per_image=True)
            loss = ce_loss + lovasz_loss

            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(main_output, dim=1)  # 只使用主输出进行预测
            confusion_matrix += compute_confusion_matrix(preds, labels, num_classes)

    avg_loss = running_loss / len(val_loader.dataset)
    IoU, mean_IoU = compute_iou_from_confusion(confusion_matrix, num_classes, ignore_background=True)

    print(f"Fold {fold} 验证损失: {avg_loss:.4f}")
    logging.info(f"Fold {fold} 验证损失: {avg_loss:.4f}")
    print("每类 IoU:")
    logging.info("每类 IoU:")
    for cls in range(num_classes):
        print(f"类别 {cls} IoU: {IoU[cls]:.4f}")
        logging.info(f"类别 {cls} IoU: {IoU[cls]:.4f}")
    print(f"Fold {fold} 平均 IoU（排除背景）: {mean_IoU:.4f}")
    logging.info(f"Fold {fold} 平均 IoU（排除背景）: {mean_IoU:.4f}")

    if writer and epoch is not None:
        writer.add_scalar('Loss/val', avg_loss, epoch)
        writer.add_scalar('IoU/val_mean_no_background', mean_IoU, epoch)
        for cls in range(num_classes):
            writer.add_scalar(f'IoU/val_class_{cls}', IoU[cls].item(), epoch)

    return mean_IoU

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10, save_path="model_fold{fold}_best", writer=None, fold=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    if torch.cuda.device_count() > 1:
        logging.info(f"使用 {torch.cuda.device_count()} 张 GPU 进行训练。")
        model = nn.DataParallel(model)

    # 确保 checkpoint 目录存在
    os.makedirs('checkpoint', exist_ok=True)
    
    # 获取当前时间用于日志记录
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if writer is None:
        writer = SummaryWriter(f'runs_cv/experiment_fold_{fold}_{current_time}')

    best_miou = 0.0  # 初始化最佳 mIoU
    best_loss = float('inf')  # 初始化最佳（最低）训练损失
    best_epoch_miou = 0   # 记录最佳 mIoU 的 epoch
    best_epoch_loss = 0   # 记录最佳训练损失的 epoch
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"第 {fold} 折训练 Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]

            # 计算每个输出的损失
            ce_losses = []
            lovasz_losses = []

            for output in outputs:
                ce_loss = criterion(output, labels)
                prob = torch.softmax(output, dim=1)
                lovasz_loss = lovasz_softmax(prob, labels, per_image=True)
                
                ce_losses.append(ce_loss)
                lovasz_losses.append(lovasz_loss)

            #可以调整权重
            ce_weight = [1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05]
            lovasz_weight = [1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05]
            
            loss = sum(w_ce * ce + w_lov * lovasz
                       for w_ce, ce, w_lov, lovasz in zip(ce_weight, ce_losses, lovasz_weight, lovasz_losses))

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"第 {fold} 折 Epoch [{epoch+1}/{num_epochs}]，损失: {epoch_loss:.4f}")
        logging.info(f"第 {fold} 折 Epoch [{epoch+1}/{num_epochs}]，损失: {epoch_loss:.4f}")
        writer.add_scalar('Loss/train', epoch_loss, epoch)

        # 更新学习率调度器
        scheduler.step()

        # 在每个 epoch 结束后验证模型，并检查是否是新的最佳 mIoU
        val_miou = validate_model(model, val_loader, criterion, writer, epoch, fold, num_classes=4)

        # 检查是否是新的最佳 mIoU
        if val_miou > best_miou:
            best_miou = val_miou
            best_epoch_miou = epoch + 1
            # 保存最佳 mIoU 模型，保留四位小数
            save_path_miou = f"checkpoint/{save_path}_miou.pth"
            torch.save(model.state_dict(), save_path_miou)
            print(f"新的最佳 mIoU: {best_miou:.4f} 在 Epoch {best_epoch_miou}。模型已保存为 {save_path_miou}")
            logging.info(f"新的最佳 mIoU: {best_miou:.4f} 在 Epoch {best_epoch_miou}。模型已保存为 {save_path_miou}")

        # 检查是否是新的最佳训练损失
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch_loss = epoch + 1
            # 保存最佳训练损失模型，保留四位小数
            save_path_loss = f"checkpoint/{save_path}_loss.pth"
            torch.save(model.state_dict(), save_path_loss)
            print(f"新的最佳（最低）训练损失: {best_loss:.4f} 在 Epoch {best_epoch_loss}。模型已保存为 {save_path_loss}")
            logging.info(f"新的最佳（最低）训练损失: {best_loss:.4f} 在 Epoch {best_epoch_loss}。模型已保存为 {save_path_loss}")


    writer.close()
    print(f"训练完成。最佳 mIoU: {best_miou:.4f} 出现在 Epoch {best_epoch_miou}")
    logging.info(f"训练完成。最佳 mIoU: {best_miou:.4f} 出现在 Epoch {best_epoch_miou}")
    print(f"最低训练损失: {best_loss:.4f} 出现在 Epoch {best_epoch_loss}")
    logging.info(f"最低训练损失: {best_loss:.4f} 出现在 Epoch {best_epoch_loss}")



def main():
    # 数据路径
    external_image_dirs = [
        'NEU_Seg-main/images/training',
        'NEU_Seg-main/images/test',

    ]
    external_label_dirs = [
        'NEU_Seg-main/annotations/training',
        'NEU_Seg-main/annotations/test',

    ]
    main_image_dir =[
        'DataB/Img',
    ]
    main_label_dir = [
        'DataB/Lab',
    ]

    # 收集所有外部图像和标签路径
    external_image_paths, external_label_paths = collect_image_label_paths(external_image_dirs, external_label_dirs)

    # 收集所有主要图像和标签路径
    main_image_paths, main_label_paths = collect_image_label_paths(main_image_dir, main_label_dir)

    # 确保有数据
    assert len(external_image_paths) > 0, "没有收集到任何外部训练图像路径。"
    assert len(main_image_paths) > 0, "没有收集到任何主要数据集图像路径。"

    # 定义超参数
    num_epochs = 2000
    batch_size = 48
    learning_rate = 4e-4
    save_path = "model_best"  # 单一模型保存路径

    # 每种增强的概率参数（确保各自的概率根据需求设置）
    augmentation_probs = {
        1: 0.12125,  # 四张图片拼接
        2: 0.10125,  # 上下拼接
        3: 0.09375,  # 左右拼接
        4: 0.05875,  # 左上角拼接
        5: 0.1875,   # 加椒盐噪声
        6: 0.4375,   # 保持原图
        7: 0.0       # 选择裁剪
    }
    # 计算总和
    total = sum(augmentation_probs.values())

    # 归一化，使所有值的和为1
    augmentation_probs = {k: v / total for k, v in augmentation_probs.items()}

    # 定义在线增强实例仅应用于外部数据
    external_online_augment = OnlineAugment(
        image_paths=external_image_paths,
        label_paths=external_label_paths,
        augment_prob=1.0,  # 总体增强概率
        augmentation_probs=augmentation_probs
    )

    # 创建A训练数据集（带在线增强）
    external_train_dataset = SteelDefectDataset(
        image_paths=external_image_paths,
        label_paths=external_label_paths,
        transform=train_transform,
        online_augment=external_online_augment
    )

    # 创建B训练数据集（不应用在线增强）
    main_train_dataset = SteelDefectDataset(
        image_paths=main_image_paths,
        label_paths=main_label_paths,
        transform=train_transform,
        online_augment=None  # 不应用在线增强
    )
    
    # Create main validation dataset (without online augmentation)
    main_val_dataset = SteelDefectDataset(
        image_paths=external_image_paths,
        label_paths=external_label_paths,
        transform=val_transform,  # Use validation-specific transforms
        online_augment=external_online_augment
    )

    # 如果需要使 external_train_dataset 比 main_train_dataset 多训练，可以调整重复次数
    external_train_dataset_repeated = ConcatDataset([external_train_dataset] * 1)  # 这里重复1次，即不重复

    # 合并所有训练数据集
    combined_train_dataset = ConcatDataset([external_train_dataset_repeated, main_train_dataset])

    # 创建DataLoader
    train_loader = DataLoader(
        combined_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    
    # Create validation DataLoader
    val_loader = DataLoader(
        main_val_dataset,
        batch_size=batch_size,  # You can adjust the batch size for validation
        shuffle=False,         # No need to shuffle for validation
        num_workers=8,         # Fewer workers for validation
        pin_memory=True
    )

    # 初始化模型、损失函数、优化器和学习率调度器
    model = self_net(in_ch=1, out_ch=4)  # 根据您的模型定义调整参数

    # 加载预训练模型权重（如果有的话）
    pretrained_model_path = 'checkpoint/model.pth'  # 假设是已经训练好的权重路径

    if os.path.exists(pretrained_model_path):
        state_dict = torch.load(pretrained_model_path)
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        # 加载模型权重
        model.load_state_dict(state_dict, strict=False)
        print(f"加载预训练的模型权重：{pretrained_model_path}")
        logging.info(f"加载预训练的模型权重：{pretrained_model_path}")
    else:
        print("没有找到预训练模型，初始化为随机权重")
        logging.info("没有找到预训练模型，初始化为随机权重")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)  # 这里可以调整 momentum 的值
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)

    # 训练模型
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        save_path=save_path,
        writer=None,  # 在 train_model 内部创建 SummaryWriter
        fold=1
    )
    


if __name__ == "__main__":
    main()
