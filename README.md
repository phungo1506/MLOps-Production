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
Training and testing on GPU
| Model | #Params | FLOPs | Accuracy | Time Inference | Model Size |
|:---------------|:----:|:---:|:--:|:--:|:--:|
| MobileNetv3 (Large) - Teacher |     5.8M     |   7.7G    |     57.14%     | 20.62s  |    17MB     |
| MobileNetv3 (Custom) - Student without Teacher       |     0.18M    |   1.4G  |     54.37%     | 6.38s |    0.73MB    |
| MobileNetv3 (Custom) - Knowledge Distillation      |     0.18M    |   1.4G  |     44.18%     | 6.42s |    0.73MB    |


Training and testing on GPU A100
| Model | #Params | FLOPs | Accuracy | Time Inference | Model Size |
|:---------------|:----:|:---:|:--:|:--:|:--:|
| Efficientnet - Teacher |     168.62M     |   601.7G    |     44.18%     | 428.68s  |    675.8MB     |
| MobileNetv3 (Custom) - Student without Teacher       |     0.18M    |   1.4G  |     53.15%     | 5.74s |    0.73MB    |
| MobileNetv3 (Custom) - Knowledge Distillation      |     0.18M    |   1.4G  |     44.18%     | 6.00s |    0.73MB    |

## Quantization
| Model | Quantized | Accuracy | Time Inference | Model Size |
|:---------------|:---:|:--:|:--:|:--:|
| MobileNetv3 (Small)|    -   |     49.81%     | 52.17s  |    6.2MB     |
| MobileNetv3 (Small)| Dynamic Quantization |    49.81%      | 38.15s |    4.72MB    |
| MobileNetv3 (Small) | Static Quantization |    44.18%     | 23.88s |    1.81MB    |
| MobileNetv3 (Small) | Selective Static Quantization |  44.18%     | 22.36s |    1.9MB    |
| MobileNetv3 (Small) | Quantization Aware Training |     24.00%     | 21.28s |    1.8MB    |
| ResNet(50)   |    -    |     53.97%     | 609.62s  |    94.38MB     |
| ResNet(50)    |   Dynamic Quantization |   54.05%     | 562.98  |    93.97MB    |
| ResNet(50)    |   Static Quantization   |     15.02%     | 274.45s |    24.11MB    |
| ResNet(50)    |   Selective Static Quantization  |     15.32%     | 279.02s |    27.96MB    |
| ResNet(50)    |   Selective Static Quantization  |     44.18%     | 280.36s |    24.11MB    |


## ONNX model
| Model | #Params | FLOPs | Accuracy | Time Inference | Model Size |
|:---------------|:----:|:---:|:--:|:--:|:--:|
| MobileNetv3 (Samll) |     1.9M     |   7.5G    |     49.96%     | 3.27s  |    5.78MB     |

## Installation
```
pip install -r requirements.txt
```
