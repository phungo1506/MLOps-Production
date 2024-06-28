# 

# MLOps-Production
This project aims to train the model, reduce the model size using knowledge distillation or quantization methods to reduce hardware costs, and then use NVIDIA Triton Inference Server to deploy the classification model.

<a href="https://colab.research.google.com/drive/1XUKG661hk4xSdLeAIU6ExNpuL3lTj-vG?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> 

## Performance
[UIT-CVID21 Dataset](https://mmlab.uit.edu.vn/dataset/2023/09/25/datasets4)

| Model | #Params | FLOPs | Accuracy | Time Inference | Model Size |
|:---------------|:----:|:---:|:--:|:--:|:--:|
| MobileNetv3 (Small) |     1.9M     |   7.5G    |     49.81%     | 41.91s  |    6.2MB     |
| ResNet(50)          |     25.5M    |   523.2G  |     53.97%     | 532.94s |    94.4MB    |

### Knowledge Distillation
Training and testing on GPU L4
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

### Quantization
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


### ONNX model
| Model | #Params | FLOPs | Accuracy | Time Inference | Model Size |
|:---------------|:----:|:---:|:--:|:--:|:--:|
| MobileNetv3 (Samll) |     1.9M     |   7.5G    |     49.96%     | 3.27s  |    5.78MB     |

<!-- ## Environments

OS Ubuntu 20.04 (WSL2)
Python 3.8.10
Triton Inference Server 2.34.0
Pillow 9.3.0
ONNX 1.15.0
ONNX Runtime 1.16.0
Docker 24.0.6 -->

## Installation
```
pip install -r requirements.txt
```

## Deploying Model with Triton Inference Server on GCP
Create a compute on GCP here I only use CPU. You can use GPU, just change the command a little.

Step 1: 
``` 
git clone -b r23.05 https://github.com/triton-inference-server/server.git 
```
Step 2:
``` 
cd server/docs/examples 
```
Then create a folder containing the model file, config file and class name file. The directory structure is as follows:
<img src="https://github.com/phungo1506/MLOps-Production/blob/main/images/Structure%20folder.png"/>

Step 3:
``` 
docker run --rm --net=host -v ${PWD}/model_repository:/models  
nvcr.io/nvidia/tritonserver:23.05-py3 tritonserver --model-repository=/models 
```
``

## Demo
Use colab notebook as client to demo <a href="https://colab.research.google.com/drive/1XUKG661hk4xSdLeAIU6ExNpuL3lTj-vG?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> 

## Further Reading
*[Quantization — PyTorch ](https://pytorch.org/docs/stable/quantization.html)
*[Knowledge Distillation - Pytorch](https://pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html)
*[ONXX - Pytorch](https://pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html)
*[ONNX - Tutorials](https://github.com/onnx/tutorials)
*[Architecture Example](https://github.com/maciejbalawejder/Deep-Learning-Collection/tree/main)