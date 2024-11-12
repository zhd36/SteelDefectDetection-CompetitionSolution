import random
import albumentations as A
import numpy as np
from PIL import Image

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

            # # 随机水平翻转
            # if random.random() > 0.5:
            #     pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
            #     pil_lbl = pil_lbl.transpose(Image.FLIP_LEFT_RIGHT)

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

    
    def augment_type_2(self, image, label):
        """
        增强类型2：两张图片上下拼接，各取原始图片的一半，并对对应的半部分进行随机翻转。
        """
        img2, lbl2 = self.get_random_image()
        height = image.shape[0] // 2

        # 分割图像和标签
        top_image = image[:height, :]
        bottom_image = img2[height:, :]
        top_label = label[:height, :]
        bottom_label = lbl2[height:, :]

        # 随机翻转上半部分
        if random.random() > 0.5:
            pil_top_img = Image.fromarray(top_image).transpose(Image.FLIP_LEFT_RIGHT)
            pil_top_lbl = Image.fromarray(top_label).transpose(Image.FLIP_LEFT_RIGHT)
            top_image = np.array(pil_top_img)
            top_label = np.array(pil_top_lbl)

        # 随机翻转下半部分
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
        增强类型3：两张图片左右拼接，各取原始图片的一半，并对对应的半部分进行随机翻转。
        """
        img2, lbl2 = self.get_random_image()
        width = image.shape[1] // 2

        # 分割图像和标签
        left_image = image[:, :width]
        right_image = img2[:, width:]
        left_label = label[:, :width]
        right_label = lbl2[:, width:]

        # 随机翻转左半部分
        if random.random() > 0.5:
            pil_left_img = Image.fromarray(left_image).transpose(Image.FLIP_TOP_BOTTOM)
            pil_left_lbl = Image.fromarray(left_label).transpose(Image.FLIP_TOP_BOTTOM)
            left_image = np.array(pil_left_img)
            left_label = np.array(pil_left_lbl)

        # 随机翻转右半部分
        if random.random() > 0.5:
            pil_right_img = Image.fromarray(right_image).transpose(Image.FLIP_TOP_BOTTOM)
            pil_right_lbl = Image.fromarray(right_label).transpose(Image.FLIP_TOP_BOTTOM)
            right_image = np.array(pil_right_img)
            right_label = np.array(pil_right_lbl)

        # 拼接图像和标签
        combined_image = np.hstack((left_image, right_image))
        combined_label = np.hstack((left_label, right_label))
        return combined_image, combined_label
 
    def augment_type_4(self, image, label):
        # Step 1: 随机翻转和旋转
        pil_image = Image.fromarray(image)
        pil_label = Image.fromarray(label)

        # 随机水平翻转
        if random.random() > 0.5:
            pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
            pil_label = pil_label.transpose(Image.FLIP_LEFT_RIGHT)

        # 随机旋转（0, 90, 180, 270度）
        rotation_angle = random.choice([0, 90, 180, 270])
        pil_image = pil_image.rotate(rotation_angle)
        pil_label = pil_label.rotate(rotation_angle)

        # 转回numpy数组
        image = np.array(pil_image)
        label = np.array(pil_label)

        # Step 2: 替换左上角1/4
        img2, lbl2 = self.get_random_image()
        pil_img2 = Image.fromarray(img2)
        pil_lbl2 = Image.fromarray(lbl2)

        if random.random() > 0.5:
            pil_img2 = pil_img2.transpose(Image.FLIP_LEFT_RIGHT)
            pil_lbl2 = pil_lbl2.transpose(Image.FLIP_LEFT_RIGHT)

        rotation_angle = random.choice([0, 90, 180, 270])
        pil_img2 = pil_img2.rotate(rotation_angle)
        pil_lbl2 = pil_lbl2.rotate(rotation_angle)

        img2 = np.array(pil_img2)
        lbl2 = np.array(pil_lbl2)

        quarter_height = image.shape[0] // 2
        quarter_width = image.shape[1] // 2

        quarter_image = image[:quarter_height, :quarter_width]
        quarter_label = label[:quarter_height, :quarter_width]

        combined_image = img2.copy()
        combined_label = lbl2.copy()

        combined_image[:quarter_height, :quarter_width] = quarter_image
        combined_label[:quarter_height, :quarter_width] = quarter_label

        # 定义小矩形的最大尺寸，允许扩展到原图的 60%
        max_rect_height = int(image.shape[0] * 0.6)  # 最大高度为原图的 60%
        max_rect_width = int(image.shape[1] * 0.6)  # 最大宽度为原图的 60%

        # 随机选择形状类型：0-1为正方形，2为横向长方形，3为纵向长方形
        shape_type = random.choice([0, 1, 2, 3])

        if shape_type in [0, 1]:  # 0 和 1 生成类似正方形的矩形，宽高接近
            rect_height = random.randint(max_rect_height // 2, int(max_rect_height/1.3 ))
            rect_width = random.randint(max_rect_height // 2, int(max_rect_height /1.3))  # 与高度相似

            # 正方形在左上角1/4区域内均匀分布
            rect_top = random.randint(0, min(quarter_height, image.shape[0] - rect_height))
            rect_left = random.randint(0, min(quarter_width, image.shape[1] - rect_width))

        elif shape_type == 2:  # 2 生成偏向横向长方形的矩形，宽度更大
            rect_height = random.randint(max_rect_height // 16, max_rect_height // 4)  # 较小的高度
            rect_width = random.randint(int(max_rect_width / 1.5), max_rect_width)  # 较大的宽度

            # 横向长方形倾向于在左上角1/4区域的右上角
            rect_top = random.randint(0, min(quarter_height // 2, image.shape[0] - rect_height))  # 上半部分
            rect_left = random.randint(quarter_width // 2, min(quarter_width, image.shape[1] - rect_width))  # 右半部分

        else:  # 3 生成偏向纵向长方形的矩形，高度更大
            rect_height = random.randint(int(max_rect_height / 1.5), max_rect_height)  # 较大的高度
            rect_width = random.randint(max_rect_width // 16, max_rect_width // 4)  # 较小的宽度

            # 纵向长方形倾向于在左上角1/4区域的左下角
            rect_top = random.randint(quarter_height // 2, min(quarter_height, image.shape[0] - rect_height))  # 下半部分
            rect_left = random.randint(0, min(quarter_width // 2, image.shape[1] - rect_width))  # 左半部分

        # 裁剪源图像和标签的对应区域
        source_cropped_image = image[rect_top:rect_top + rect_height, rect_left:rect_left + rect_width]
        source_cropped_label = label[rect_top:rect_top + rect_height, rect_left:rect_left + rect_width]

        # 替换目标图像和标签的对应区域
        combined_image[rect_top:rect_top + rect_height, rect_left:rect_left + rect_width] = source_cropped_image
        combined_label[rect_top:rect_top + rect_height, rect_left:rect_left + rect_width] = source_cropped_label

        return combined_image, combined_label
    
    
    def augment_type_5(self, image, label):
        """
        增强类型5：变化。
        """
        s_vs_p = 0.5 + np.random.uniform(-0.05, 0.05)  # 在 0.5 的基础上随机浮动 ±0.05
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
        """
        # 定义增强流水线
        train_transform = A.Compose([
            A.Resize(200, 200),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.35, p=0.2),  # 亮度和对比度调整
            #A.GaussNoise(var_limit=(0.0, 0.1), p=0.15),  # 高斯噪声，噪声方差范围为 [0, 0.1]
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
        增强类型7：
        1. 从原始图像中裁剪一个区域。
        2. 从另一张随机图像中裁剪相同大小的区域，先随机缩放，再随机旋转。
        3. 将处理后的区域替换到原始图像的裁剪区域上。
        """
        # 1. 从原始图像中裁剪一个区域
        h, w = image.shape
        crop_h = h // 4  # 1/4 高度
        crop_w = w // 4  # 1/4 宽度

        # 随机裁剪位置
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)

        # 2. 从另一张随机图像中裁剪相同大小的区域
        img2, lbl2 = self.get_random_image()

        # 将第二张图像转换为 PIL 格式以进行操作
        pil_img2 = Image.fromarray(img2)
        pil_lbl2 = Image.fromarray(lbl2)

        # 2.1. 随机缩放
        scale_factor = random.uniform(1.1, 1.3)  # 缩放因子在1.1到1.3之间
        new_size = (int(pil_img2.width * scale_factor), int(pil_img2.height * scale_factor))
        pil_img2 = pil_img2.resize(new_size, Image.BILINEAR)
        pil_lbl2 = pil_lbl2.resize(new_size, Image.NEAREST)

        # 2.2. 随机旋转任意角度
        rotation_angle = random.uniform(0, 360)  # 随机旋转角度
        pil_img2 = pil_img2.rotate(rotation_angle, resample=Image.BILINEAR, expand=True)
        pil_lbl2 = pil_lbl2.rotate(rotation_angle, resample=Image.NEAREST, expand=True)

        # 转回 numpy 数组
        img2_rotated = np.array(pil_img2)
        lbl2_rotated = np.array(pil_lbl2)

        # 确保旋转后的图像足够大以进行裁剪
        rotated_h, rotated_w = img2_rotated.shape
        if rotated_h < crop_h or rotated_w < crop_w:
            # 如果旋转后的图像太小，进行填充以确保可以裁剪
            pad_h = max(crop_h - rotated_h, 0)
            pad_w = max(crop_w - rotated_w, 0)
            img2_rotated = np.pad(
                img2_rotated,
                ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)),
                mode='reflect'
            )
            lbl2_rotated = np.pad(
                lbl2_rotated,
                ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)),
                mode='constant',
                constant_values=0
            )

        # 2.3. 随机裁剪相同大小的区域
        rotated_h, rotated_w = img2_rotated.shape
        top2 = random.randint(0, rotated_h - crop_h)
        left2 = random.randint(0, rotated_w - crop_w)
        cropped_image2 = img2_rotated[top2:top2 + crop_h, left2:left2 + crop_w]
        cropped_label2 = lbl2_rotated[top2:top2 + crop_h, left2:left2 + crop_w]

        # 3. 将处理后的区域替换到原始图像的裁剪区域上
        image[top:top + crop_h, left:left + crop_w] = cropped_image2
        label[top:top + crop_h, left:left + crop_w] = cropped_label2

        return image, label