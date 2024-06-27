# 

# MLOps-Production

[<a href="https://colab.research.google.com/drive/1XUKG661hk4xSdLeAIU6ExNpuL3lTj-vG?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> 

## Performance
[UIT-CVID21 Dataset](https://mmlab.uit.edu.vn/dataset/2023/09/25/datasets4)

| Model | #Params | FLOPs | Accuracy | Time Inference | Model Size |
|:---------------|:----:|:---:|:--:|:--:|:--:|
| MobileNetv3 (Small) |     1.9M     |   7.5G    |     49.81%     | 41.91s  |    6.2MB     |
| ResNet(50)          |     25.5M    |   523.2G  |     53.97%     | 532.94s |    94.4MB    |

## Knowledge Distillation

| Model | #Params | FLOPs | Accuracy | Time Inference | Model Size |
|:---------------|:----:|:---:|:--:|:--:|:--:|
| MobileNetv3 (Large) - Teacher |     5.8M     |   7.7G    |     57.14%     | 20.62s  |    17MB     |
| MobileNetv3 (Custom) - Student without Teacher       |     0.18M    |   1.4G  |     54.37%     | 6.38s |    0.73MB    |
| MobileNetv3 (Custom) - Knowledge Distillation      |     0.18M    |   1.4G  |     44.18%     | 6.42s |    0.73MB    |

## Installation
```
pip install -r requirements.txt
```
