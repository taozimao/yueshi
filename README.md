# 参赛作品名：

悦食
健康吃、愉悦聚

## 作品简介

使用用 paddle 上现有的食品数据集，做一个二分类的问题，糖尿病人“可以吃的食物”和"忌口食物"两大类，然后再做一个可以微信扫描识别的接口；用微信扫描某一食品，弹出绿色笑脸表示可以放心食用，弹出红色哭脸表示注意忌口～～

未来想往控糖饮食健康管理平台迭代，扫一扫即时获得食物的血糖指数，更集合了健康饮食搭配、血糖血压记录、食友小聚、饮食课堂、咨询和商城等多项服务。

## 实现方式

基于 paddleX 和 MobileNetV2 模型来实现

实现方式如下所示：

## 安装 paddleX

```
pip install paddlex -i https://mirror.baidu.com/pypi/simple
```

## 数据集标注和划分

基于 food-101 网络开源的数据集，重新分类为两类，一类是糖尿病患者可以吃的食物为一类；另一类是不宜吃的食物。

数据划分权重：

```
!paddlex --split_dataset --format ImageNet --dataset_dir '/home/aistudio/dia-food' --val_value 0.2 --test_value 0.1
```

## 配置 GPU

```
import matplotlib
matplotlib.use('Agg')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import paddlex as pdx
```

## 定义图像处理流程

```
from paddlex.cls import transforms
train_transforms = transforms.Compose([
    transforms.RandomCrop(crop_size=224),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize()
])
eval_transforms = transforms.Compose([
    transforms.ResizeByShort(short_size=256),
    transforms.CenterCrop(crop_size=224),
    transforms.Normalize()
])
```

## 定义数据集

```
train_dataset = pdx.datasets.ImageNet(
    data_dir='/home/aistudio/work/dia-food',
    file_list='/home/aistudio/work/dia-food/train_list.txt',
    label_list='/home/aistudio/work/dia-food/labels.txt',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.ImageNet(
    data_dir='/home/aistudio/work/dia-food',
    file_list='/home/aistudio/work/dia-food/val_list.txt',
    label_list='/home/aistudio/work/dia-food/labels.txt',
    transforms=eval_transforms)
```

## 模型开始训练

```

num_classes = len(train_dataset.labels)
model = pdx.cls.MobileNetV2(num_classes=num_classes)
model.train(num_epochs=10,
            train_dataset=train_dataset,
            train_batch_size=32,
            eval_dataset=eval_dataset,
            lr_decay_epochs=[4, 6, 8],
            save_interval_epochs=1,
            learning_rate=0.025,
            save_dir='output/mobilenetv2',
            use_vdl=True)
```

## 模型预测

```
import paddlex as pdx
model = pdx.load_model('/home/aistudio/work/mobilenetv2/best_model')
image_name = '/home/aistudio/work/nn.jpg'
result = model.predict(image_name)
print("Predict Result:", result)
```
