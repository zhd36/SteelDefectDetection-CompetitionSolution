# 钢材表面缺陷检测与分割解决方案

"DeepInspect-深度检测"队

本项目是参加钢材表面缺陷检测与分割竞赛的解决方案，旨在通过深度学习技术实现高效的钢材表面缺陷检测和分割。

竞赛官网：[钢材表面缺陷检测与分割竞赛](http://bdc.saikr.com/vse/50185)

## 项目结构

- `OnlineAugment.py`: 实现在线数据增强的脚本。通过数据增强提高模型的泛化能力，特别适用于具有多种形态的缺陷检测任务。
- `eval_with_aug.py`: 使用数据增强对模型进行评估的脚本，用于评估模型在增强数据集上的表现。
- `lovasz_softmax.py`: Lovasz-Softmax 损失函数的实现。该损失函数适用于分割任务，有助于优化 mIoU（mean Intersection over Union）指标。
- `model.py`: 模型定义文件。基于 [U-2-Net](https://github.com/xuebinqin/U-2-Net) 结构，并加入深度可分离卷积，以减少计算量和参数量，提高模型效率。
- `train.py`: 用于模型训练的脚本。在国赛中使用了 A 和 B 数据集，A 集数据使用在线数据增强，B 集数据采用简单增强策略，以提升模型在不同数据集上的表现。
- `README.md`: 项目文档（即本文件），描述项目的结构、使用方法和相关细节。
- `checkpoint/model.pth`: 以 A 数据集训练得到的最终模型，mIoU 达到 0.89。此模型在国赛中用作预训练模型，为后续的微调和模型改进提供了基础。
- `checkpoint/model_B.pth`: 使用 `model.pth` 作为预训练模型，在 B 数据集上继续训练得到的模型，用于提升模型在 B 集上的分割精度。

## 参考资料

- [U-2-Net GitHub 仓库](https://github.com/xuebinqin/U-2-Net): 项目中使用的基础网络结构。
- [Lovasz-Softmax Loss](https://arxiv.org/abs/1705.08790): 专为优化 mIoU 而设计的分割任务损失函数。
- [MobileNet: Depthwise Separable Convolutions](https://arxiv.org/abs/1704.04861): MobileNet 提供的深度可分离卷积技术思路，用于降低模型参数量和计算复杂度。

## 致谢

感谢竞赛主办方提供的支持，以及所有开源项目和研究对本项目的帮助。特别感谢 U-2-Net 的开源实现和 Lovasz-Softmax 损失函数的研究者。此外，感谢 **o1-mini** 在代码编写过程中提供的帮助与支持。
