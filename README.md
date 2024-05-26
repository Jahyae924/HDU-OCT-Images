# 项目概述

### 1. 参考代码

代码源自github项目https://github.com/Zhenhui-Jin/Tianchi-APTOS/

### 2. 准备工作

- 数据集的下载与存放（详见[Google Drive](https://drive.google.com/file/d/1zFad4ILa-Cb60YP5aOnwsOQT7YcwbvQv/view?usp=sharing)）,数据解压后是个名为`data`的文件夹，存放在项目的根目录下
- 删除load_model部分

### 3. 如何启动

```shell
## 执行命令
$ python main.py
## 本项目只进行了从图像中预测CST以及四种积液的存在
$ 1
```

### 4. 损失函数较高问题如何解决

由于源代码在分类问题的损失函数的选择上已经包含了sigmoid函数，因此在两个子任务中的分类任务的最后不应再使用sigmoid函数，删去即可（代码中已删除）

### 5. 后续问题解决？

除却从图像中预测，也可以从csv文件中预测治疗后视力值VA与是否要继续注射抗VEGF药物。模型也应该从头开始训练而不是加载已有模型。

