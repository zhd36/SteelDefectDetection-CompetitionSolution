# 钢材表面缺陷检测与分割解决方案
"DeepInspect-深度检测"队

竞赛官网 http://bdc.saikr.com/vse/50185

## 项目结构

- `checkpoint/`: 用于保存模型检查点的文件夹。
- `OnlineAugment.py`: 在线数据增强脚本。
- `README.md`: 项目文档（即本文件）。
- `eval_with_aug.py`: 使用数据增强对模型进行评估的脚本。
- `lovasz_softmax.py`: Lovasz-Softmax 损失函数的实现。
- `model.py`: 深度学习模型的定义文件。
- `train.py`: 用于训练模型的脚本。
