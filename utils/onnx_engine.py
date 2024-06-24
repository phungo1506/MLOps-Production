import torch
import time
import onnx
import onnxruntime as ort
import numpy as np
from tqdm import tqdm
import os

def export_to_onnx(model, example_inputs, onnx_file_path):
    model.eval()
    torch.onnx.export(
        model,                               # model being run
        example_inputs,                      # model input (or a tuple for multiple inputs)
        onnx_file_path,                      # where to save the model (can be a file or file-like object)
        export_params=True,                  # store the trained parameter weights inside the model file
        opset_version=12,                    # the ONNX version to export the model to
        do_constant_folding=True,            # whether to execute constant folding for optimization
        input_names=['input'],               # the model's input names
        output_names=['output'],             # the model's output names
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # variable length axes
    )
    print(f"Model exported to {onnx_file_path}")

def load_onnx_model(onnx_model_path):
    return ort.InferenceSession(onnx_model_path)

def preprocess_input(input_tensor):
    return input_tensor.cpu().numpy()

def get_file_size(file_path):
    size = os.path.getsize(file_path)
    size_mb = size / (1024 * 1024)  # Convert to megabytes
    return size_mb

def evaluate_onnx_model(onnx_model, dataloader, device):
    # Set the appropriate provider
    onnx_model.set_providers(['CUDAExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider'])
    
    correct = 0
    total = 0
    total_inference_time = 0.0  # To accumulate total inference time

    for inputs, labels in tqdm(dataloader):
        inputs = preprocess_input(inputs)

        # Record the start time of inference
        start_time = time.time()

        # Run the model on the input data
        ort_inputs = {onnx_model.get_inputs()[0].name: inputs}
        ort_outs = onnx_model.run(None, ort_inputs)

        # Record the end time of inference
        end_time = time.time()

        # Calculate inference time for this batch
        inference_time = end_time - start_time
        total_inference_time += inference_time

        # Get the output predictions
        outputs = torch.tensor(ort_outs[0])
        _, predicted = torch.max(outputs, 1)

        # Update the accuracy metrics
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_inference_time = total_inference_time / len(dataloader)

    return accuracy, total_inference_time
