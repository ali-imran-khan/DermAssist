import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf
import os
from timm import create_model
import onnx
import onnx_tf

def load_pytorch_model(model_path):
    """Load the trained PyTorch model"""
    print("Loading PyTorch model...")
    # Create the same model architecture as in training
    model = create_model('efficientnet_b3', pretrained=False, num_classes=2)
    
    # Load the saved weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set to evaluation mode
    return model

def convert_to_onnx(pytorch_model, onnx_path):
    """Convert PyTorch model to ONNX format"""
    print("Converting to ONNX format...")
    # Create a dummy input tensor with the same dimensions used during training
    dummy_input = torch.randn(1, 3, 300, 300)
    
    # Export the model to ONNX format
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    # Verify the ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model saved to {onnx_path}")
    return onnx_model

def convert_onnx_to_tf(onnx_model, tf_model_dir):
    """Convert ONNX model to TensorFlow SavedModel format"""
    print("Converting ONNX to TensorFlow format...")
    # Convert ONNX model to TensorFlow
    tf_rep = onnx_tf.backend.prepare(onnx_model)
    tf_rep.export_graph(tf_model_dir)
    print(f"TensorFlow model saved to {tf_model_dir}")
    return tf_model_dir

def convert_tf_to_tflite(tf_model_dir, tflite_path, quantize=False):
    """Convert TensorFlow model to TFLite format"""
    print("Converting to TFLite format...")
    # Load the SavedModel
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)
    
    # Set optimization options
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the model
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to {tflite_path}")
    return tflite_path

def main():
    # Define paths
    pytorch_model_path = "eczema_classifier_efficientnet_b3.pth"  # Your PyTorch model path
    onnx_path = "eczema_classifier.onnx"
    tf_model_dir = "tf_model"
    tflite_path = "eczema_classifier.tflite"
    
    # Create output directories if they don't exist
    if not os.path.exists(tf_model_dir):
        os.makedirs(tf_model_dir)
    
    # Step 1: Load PyTorch model
    pytorch_model = load_pytorch_model(pytorch_model_path)
    
    # Step 2: Convert to ONNX
    onnx_model = convert_to_onnx(pytorch_model, onnx_path)
    
    # Step 3: Convert ONNX to TensorFlow
    tf_model_path = convert_onnx_to_tf(onnx_model, tf_model_dir)
    
    # Step 4: Convert TensorFlow to TFLite
    # Set quantize=True if you want a smaller, quantized model
    tflite_model_path = convert_tf_to_tflite(tf_model_path, tflite_path, quantize=False)
    
    print("\nConversion complete!")
    print(f"TFLite model size: {os.path.getsize(tflite_path) / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    main()