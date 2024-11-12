import random
import albumentations as A
import numpy as np
from PIL import Image,ImageEnhance


class OnlineAugment:
    def __init__(self, image_paths, label_paths, augment_prob=0.7, augmentation_probs=None):
        """
        初始化在线增强类

        Args:
            image_paths (list): 所有图像路径列表
            label_paths (list): 所有标签路径列表
            augment_prob (float): 应用增强的总体概率
            augmentation_probs (dict): 每种增强方式的独立概率
        """
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.augment_prob = augment_prob
        # 如果没有提供 augmentation_probs，则均等概率
        self.augmentation_probs = augmentation_probs or  {
            1: 0.12125,  # 四张图片拼接
            2: 0.10125,  # 上下拼接
            3: 0.09375,  # 左右拼接
            4: 0.05875,  # 左上角拼接
            5: 0.1875,   # 加椒盐噪声
            6: 0.4375,   # 保持原图
            7: 0.0       # 选择裁剪（训练时设置为0）
        }

    def __call__(self, image, label):
        if random.random() > self.augment_prob:
            return image, label  # 不进行增强

        # 按照设定的概率选择增强方式
        augmentation_type = random.choices(
            population=list(self.augmentation_probs.keys()),
            weights=list(self.augmentation_probs.values()),
            k=1
        )[0]

        # 应用对应增强方法
        if augmentation_type == 1:
            image, label = self.augment_type_1(image, label)
        elif augmentation_type == 2:
            image, label = self.augment_type_2(image, label)
        elif augmentation_type == 3:
            image, label = self.augment_type_3(image, label)
        elif augmentation_type == 4:
            image, label = self.augment_type_4(image, label)
        elif augmentation_type == 5:
            image, label = self.augment_type_5(image, label)
        elif augmentation_type == 6:
            image, label = self.augment_type_6(image, label)
        elif augmentation_type == 7:
            image, label = self.augment_type_7(image, label)
        # 如果选择增强类型 6，则保持原图不变

        return image, label

    def get_random_image(self):
        random_idx = random.randint(0, len(self.image_paths) - 1)
        img_path = self.image_paths[random_idx]
        label_path = self.label_paths[random_idx]
        image = Image.open(img_path).convert("L")
        label = Image.open(label_path).convert("L")
        return np.array(image), np.array(label)

    def augment_type_1(self, image, label):
        """
        增强类型1：两张图片拼到一起，每张图片裁剪左上角、右上角、左下角或右下角的部分，然后拼接为原图大小。
        """
        # 随机抽取两张图片
        images = [image]
        labels = [label]
        for _ in range(1):  # 仅增加一张随机图片
            img, lbl = self.get_random_image()
            images.append(img)
            labels.append(lbl)

        # 定义裁剪尺寸和拼接位置
        crop_height = image.shape[0] // 2
        crop_width = image.shape[1] // 2
        positions = [(0, 0), (crop_width, crop_height), (0, crop_height), (crop_width, 0)]

        # 随机翻转、亮度变换和裁剪每张图像和标签
        augmented_images = []
        augmented_labels = []
        for img, lbl in zip(images, labels):
            pil_img = Image.fromarray(img)
            pil_lbl = Image.fromarray(lbl)

            # 随机亮度变换
            if random.random() > 0.95:
                enhancer = ImageEnhance.Brightness(pil_img)
                brightness_factor = random.uniform(0.9, 1.1)
                pil_img = enhancer.enhance(brightness_factor)

            # 随机水平翻转
            if random.random() > 0.5:
                pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
                pil_lbl = pil_lbl.transpose(Image.FLIP_LEFT_RIGHT)

            # 转回 numpy 数组
            img = np.array(pil_img)
            lbl = np.array(pil_lbl)

            # 裁剪不同区域的图像
            if len(augmented_images) == 0:
                # 第一张图像：左上角和右下角
                augmented_images.append([
                    img[:crop_height, :crop_width],       # 左上角
                    img[crop_height:, crop_width:]        # 右下角
                ])
                augmented_labels.append([
                    lbl[:crop_height, :crop_width],       # 左上角
                    lbl[crop_height:, crop_width:]        # 右下角
                ])
            else:
                # 第二张图像：右上角和左下角
                augmented_images.append([
                    img[:crop_height, crop_width:],       # 右上角
                    img[crop_height:, :crop_width]        # 左下角
                ])
                augmented_labels.append([
                    lbl[:crop_height, crop_width:],       # 右上角
                    lbl[crop_height:, :crop_width]        # 左下角
                ])

        # 创建空的拼接图像
        combined_image = Image.new('L', (image.shape[1], image.shape[0]))
        combined_label = Image.new('L', (label.shape[1], label.shape[0]))

        # 将裁剪的部分按位置粘贴到新图像上
        for img_part, lbl_part, pos in zip(augmented_images[0] + augmented_images[1], augmented_labels[0] + augmented_labels[1], positions):
            combined_image.paste(Image.fromarray(img_part), pos)
            combined_label.paste(Image.fromarray(lbl_part), pos)

        return np.array(combined_image), np.array(combined_label)

    
    def augment_type_2(self, image, label):
        """
        增强类型2：两张图片上下拼接，各取原始图片的一半，并对对应的半部分进行随机水平翻转。
        在拼接前进行亮度变换。
        """
        img2, lbl2 = self.get_random_image()
        height = image.shape[0] // 2

        # 分割图像和标签
        top_image = image[:height, :]
        bottom_image = img2[height:, :]
        top_label = label[:height, :]
        bottom_label = lbl2[height:, :]

        # 随机水平翻转上半部分
        if random.random() > 0.5:
            pil_top_img = Image.fromarray(top_image).transpose(Image.FLIP_LEFT_RIGHT)
            pil_top_lbl = Image.fromarray(top_label).transpose(Image.FLIP_LEFT_RIGHT)
            top_image = np.array(pil_top_img)
            top_label = np.array(pil_top_lbl)

        # 随机水平翻转下半部分
        if random.random() > 0.5:
            pil_bottom_img = Image.fromarray(bottom_image).transpose(Image.FLIP_LEFT_RIGHT)
            pil_bottom_lbl = Image.fromarray(bottom_label).transpose(Image.FLIP_LEFT_RIGHT)
            bottom_image = np.array(pil_bottom_img)
            bottom_label = np.array(pil_bottom_lbl)

        # 拼接图像和标签
        combined_image = np.vstack((top_image, bottom_image))
        combined_label = np.vstack((top_label, bottom_label))

        return combined_image, combined_label


    def augment_type_3(self, image, label):
        """
        增强类型3：两张图片左右拼接，各取原始图片的一半，并对对应的半部分进行随机水平翻转。
        在拼接前进行亮度变换。
        """
        img2, lbl2 = self.get_random_image()
        width = image.shape[1] // 2

        # 分割图像和标签
        left_image = image[:, :width]
        right_image = img2[:, width:]
        left_label = label[:, :width]
        right_label = lbl2[:, width:]

        # 随机亮度变换（对左半部分图像和标签）
        if random.random() > 0.95:
            enhancer = ImageEnhance.Brightness(Image.fromarray(left_image))
            brightness_factor = random.uniform(0.8, 1.2)
            left_image = np.array(enhancer.enhance(brightness_factor))

            enhancer = ImageEnhance.Brightness(Image.fromarray(left_label))
            left_label = np.array(enhancer.enhance(brightness_factor))


        # 随机水平翻转左半部分
        if random.random() > 0.5:
            pil_left_img = Image.fromarray(left_image).transpose(Image.FLIP_LEFT_RIGHT)
            pil_left_lbl = Image.fromarray(left_label).transpose(Image.FLIP_LEFT_RIGHT)
            left_image = np.array(pil_left_img)
            left_label = np.array(pil_left_lbl)

        # 随机水平翻转右半部分
        if random.random() > 0.5:
            pil_right_img = Image.fromarray(right_image).transpose(Image.FLIP_LEFT_RIGHT)
            pil_right_lbl = Image.fromarray(right_label).transpose(Image.FLIP_LEFT_RIGHT)
            right_image = np.array(pil_right_img)
            right_label = np.array(pil_right_lbl)

        # 拼接图像和标签
        combined_image = np.hstack((left_image, right_image))
        combined_label = np.hstack((left_label, right_label))

        return combined_image, combined_label



    def augment_type_4(self, image, label):
        # Step 1: 随机翻转和亮度变换
        pil_image = Image.fromarray(image)
        pil_label = Image.fromarray(label)

        # 随机水平翻转
        if random.random() > 0.5:
            pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
            pil_label = pil_label.transpose(Image.FLIP_LEFT_RIGHT)

        # 转回 numpy 数组
        image = np.array(pil_image)
        label = np.array(pil_label)

        # Step 2: 替换左上角1/4
        img2, lbl2 = self.get_random_image()
        pil_img2 = Image.fromarray(img2)
        pil_lbl2 = Image.fromarray(lbl2)

        # 随机水平翻转第二张图像
        if random.random() > 0.5:
            pil_img2 = pil_img2.transpose(Image.FLIP_LEFT_RIGHT)
            pil_lbl2 = pil_lbl2.transpose(Image.FLIP_LEFT_RIGHT)


        # 转回 numpy 数组
        img2 = np.array(pil_img2)
        lbl2 = np.array(pil_lbl2)

        # 定义左上角1/4区域
        quarter_height = image.shape[0] // 2
        quarter_width = image.shape[1] // 2

        # 替换左上角1/4区域
        combined_image = img2.copy()
        combined_label = lbl2.copy()

        combined_image[:quarter_height, :quarter_width] = image[:quarter_height, :quarter_width]
        combined_label[:quarter_height, :quarter_width] = label[:quarter_height, :quarter_width]

        # 定义小矩形的最大尺寸，允许扩展到原图的 60%
        max_rect_height = int(image.shape[0] * 0.6)  # 最大高度为原图的 60%
        max_rect_width = int(image.shape[1] * 0.6)  # 最大宽度为原图的 60%

        # 随机选择形状类型：0-1为正方形，2为横向长方形，3为纵向长方形
        shape_type = random.choice([0, 1, 2, 3])

        if shape_type in [0, 1]:  # 正方形
            rect_height = random.randint(max_rect_height // 2, int(max_rect_height / 1.3))
            rect_width = random.randint(max_rect_height // 2, int(max_rect_height / 1.3))
            rect_top = random.randint(0, min(quarter_height, image.shape[0] - rect_height))
            rect_left = random.randint(0, min(quarter_width, image.shape[1] - rect_width))

        elif shape_type == 2:  # 横向长方形
            rect_height = random.randint(max_rect_height // 16, max_rect_height // 4)
            rect_width = random.randint(int(max_rect_width / 1.5), max_rect_width)
            rect_top = random.randint(0, min(quarter_height // 2, image.shape[0] - rect_height))
            rect_left = random.randint(quarter_width // 2, min(quarter_width, image.shape[1] - rect_width))

        else:  # 纵向长方形
            rect_height = random.randint(int(max_rect_height / 1.5), max_rect_height)
            rect_width = random.randint(max_rect_width // 16, max_rect_width // 4)
            rect_top = random.randint(quarter_height // 2, min(quarter_height, image.shape[0] - rect_height))
            rect_left = random.randint(0, min(quarter_width // 2, image.shape[1] - rect_width))

        # 替换目标图像和标签的对应区域
        combined_image[rect_top:rect_top + rect_height, rect_left:rect_left + rect_width] = image[rect_top:rect_top + rect_height, rect_left:rect_left + rect_width]
        combined_label[rect_top:rect_top + rect_height, rect_left:rect_left + rect_width] = label[rect_top:rect_top + rect_height, rect_left:rect_left + rect_width]

        return combined_image, combined_label



    def augment_type_5(self, image, label):
        """
        增强类型5：变化。
        """
        s_vs_p = 0.5 + np.random.uniform(-0.01, 0.01)  # 在 0.5 的基础上随机浮动 ±0.05
        amount = 0.05 + np.random.uniform(-0.005, 0.005)  # 在 0.05 的基础上随机浮动 ±0.005

        out = image.copy()
        # 添加椒盐噪声
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords[0], coords[1]] = 255

        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords[0], coords[1]] = 0
        return out, label
    
    def augment_type_6(self, image, label):
        """
        增强类型6：使用Albumentations库进行多种增强操作。

        具体操作包括：
            - 调整大小到200x200
            - 随机水平翻转
            - 随机90度旋转
            - 随机调整亮度和对比度
            - 添加高斯噪声
            - 添加高斯模糊
            - 降采样模拟低分辨率
            - 随机伽马变换
            - 添加JPEG压缩
            - 归一化
        """
        # 定义增强流水线
        train_transform = A.Compose([
            A.Resize(200, 200),
            A.HorizontalFlip(p=0.5),
            # A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.35, p=0.2),  # 亮度和对比度调整
            A.GaussNoise(var_limit=(0.0, 0.1), p=0.15),  # 高斯噪声，噪声方差范围为 [0, 0.1]
            #A.GaussianBlur(blur_limit=(5, 11), sigma_limit=(0.5, 1.5), p=0.2),  # 高斯模糊
            #A.Downscale(scale_min=0.5, scale_max=0.99, p=0.25),  # 低分辨率模拟
            A.RandomGamma(gamma_limit=(70, 150), p=0.15),  # 伽马变换
        ], additional_targets={'mask': 'mask'})

        # 应用增强
        augmented = train_transform(image=image, mask=label)

        # 获取增强后的图像和标签
        image_aug = augmented['image']
        label_aug = augmented['mask']

        return image_aug, label_aug

    def augment_type_7(self, image, label):
        """
        增强类型1：四张图片拼到一起，每张图片裁剪左上角、右上角、左下角、右下角四部分，然后拼接为200x200。
        """
        images = [image]
        labels = [label]
        for _ in range(3):
            img, lbl = self.get_random_image()
            images.append(img)
            labels.append(lbl)

        # 定义裁剪尺寸和拼接位置
        crop_height = image.shape[0] // 2
        crop_width = image.shape[1] // 2
        positions = [(0, 0), (crop_width, 0), (0, crop_height), (crop_width, crop_height)]

        # 随机翻转和裁剪每张图像和标签
        augmented_images = []
        augmented_labels = []
        for img, lbl in zip(images, labels):
            pil_img = Image.fromarray(img)
            pil_lbl = Image.fromarray(lbl)
            
            # 随机亮度变换
            if random.random() > 0.95:
                enhancer = ImageEnhance.Brightness(pil_img)
                brightness_factor = random.uniform(0.9, 1.1)  # 设置亮度变化范围为0.8到1.2
                pil_img = enhancer.enhance(brightness_factor)
            
            # 随机水平翻转
            if random.random() > 0.5:
                pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
                pil_lbl = pil_lbl.transpose(Image.FLIP_LEFT_RIGHT)

            # 转回 numpy 数组
            img = np.array(pil_img)
            lbl = np.array(pil_lbl)

            # 进行裁剪，获取左上角、右上角、左下角、右下角的四个部分
            augmented_images.append([
                img[:crop_height, :crop_width],       # 左上角
                img[:crop_height, crop_width:],       # 右上角
                img[crop_height:, :crop_width],       # 左下角
                img[crop_height:, crop_width:]        # 右下角
            ])

            augmented_labels.append([
                lbl[:crop_height, :crop_width],       # 左上角
                lbl[:crop_height, crop_width:],       # 右上角
                lbl[crop_height:, :crop_width],       # 左下角
                lbl[crop_height:, crop_width:]        # 右下角
            ])

        # 创建空的拼接图像
        combined_image = Image.new('L', (image.shape[1], image.shape[0]))
        combined_label = Image.new('L', (label.shape[1], label.shape[0]))

        # 将四个部分按位置粘贴到 200x200 的新图像上
        for img_part, lbl_part, pos in zip(augmented_images[0] + augmented_images[1] + augmented_images[2] + augmented_images[3],
                                           augmented_labels[0] + augmented_labels[1] + augmented_labels[2] + augmented_labels[3], 
                                           positions):
            combined_image.paste(Image.fromarray(img_part), pos)
            combined_label.paste(Image.fromarray(lbl_part), pos)

        return np.array(combined_image), np.array(combined_label) 
