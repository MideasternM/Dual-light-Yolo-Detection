# Multispectral YOLOv11 with Transformer

[English](#english) | [中文](#中文)

## English

### Overview
This project implements a multispectral object detection system based on YOLOv11 architecture with transformer, capable of processing paired RGB-IR (Infrared) images for enhanced detection performance.

### Features
- Dual-stream processing for RGB and IR images
- Transformer-based feature fusion
- Compatible with YOLOv11 architecture
- Supports Weights & Biases logging
- Flexible training and evaluation options

### Installation
```bash
git clone [repository-url]
cd yolo_soft
pip install -r requirements.txt
```

### Usage

#### Training
```bash
python train.py --weights yolo11n.pt \
                --data data.yaml \
                --cfg models/yolov11n-one_stream.yaml \
                --batch-size 4 \
                --img-size 640
```

#### Validation
```bash
python test.py --weights runs/train/exp/weights/best.pt \
               --data data.yaml \
               --img-size 640 \
               --batch-size 16
```

#### Inference
```bash
python detect.py --weights weights/best.pt \
                 --source1 test/rgb \
                 --source2 test/ir \
                 --img-size 640 \
                 --conf-thres 0.4
```

### Dataset Format
- RGB and IR images should be paired and aligned
- Data configuration should be specified in YAML format
- Support YOLO format annotations

### Performance
- Supports both accuracy and speed optimization
- Includes WandB logging for performance tracking
- Provides comprehensive evaluation metrics

## 中文

### 概述
本项目实现了一个基于YOLOv11架构的多光谱目标检测系统，集成了Transformer结构，能够处理配对的RGB-IR（红外）图像以提升检测性能。

### 特点
- 双流处理RGB和红外图像
- 基于Transformer的特征融合
- 兼容YOLOv11架构
- 支持Weights & Biases日志记录
- 灵活的训练和评估选项

### 安装
```bash
git clone [仓库地址]
cd yolo_soft
pip install -r requirements.txt
```

### 使用方法

#### 训练
```bash
python train.py --weights yolo11n.pt \
                --data data.yaml \
                --cfg models/yolov11n-one_stream.yaml \
                --batch-size 4 \
                --img-size 640
```

#### 验证
```bash
python test.py --weights runs/train/exp/weights/best.pt \
               --data data.yaml \
               --img-size 640 \
               --batch-size 16
```

#### 推理
```bash
python detect.py --weights weights/best.pt \
                 --source1 test/rgb \
                 --source2 test/ir \
                 --img-size 640 \
                 --conf-thres 0.4
```

### 数据集格式
- RGB和IR图像需要配对且对齐
- 数据配置需要以YAML格式指定
- 支持YOLO格式标注

### 性能
- 支持精度和速度优化
- 包含WandB日志记录功能
- 提供完整的评估指标

