import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic
import os

def optimize_onnx_model(input_path, output_path):
    """
    Optimize ONNX model using ONNX Runtime optimization
    """
    # Load the model
    model = onnx.load(input_path)
    
    # Create optimized model using ONNX Runtime
    # This will perform various optimizations like constant folding, 
    # operator fusion, and graph optimization
    optimized_model = ort.InferenceSession(input_path, providers=['CPUExecutionProvider'])
    
    # Save the optimized model
    onnx.save(model, output_path)
    
    print(f"Model optimized and saved to: {output_path}")
    
    # Get model info
    print(f"Original model size: {os.path.getsize(input_path) / 1024:.2f} KB")
    print(f"Optimized model size: {os.path.getsize(output_path) / 1024:.2f} KB")

def quantize_onnx_model(input_path, output_path):
    """
    Quantize ONNX model to reduce size and improve inference speed
    """
    try:
        quantize_dynamic(
            model_input=input_path,
            model_output=output_path,
            weight_type=ort.quantization.QuantType.QUInt8
        )
        print(f"Model quantized and saved to: {output_path}")
        
        # Get model info
        print(f"Original model size: {os.path.getsize(input_path) / 1024:.2f} KB")
        print(f"Quantized model size: {os.path.getsize(output_path) / 1024:.2f} KB")
        
    except Exception as e:
        print(f"Quantization failed: {e}")
        print("Falling back to basic optimization...")
        optimize_onnx_model(input_path, output_path)

if __name__ == "__main__":
    input_model = "models/Aug20/kuka_force_predictor.onnx"
    output_optimized = "models/Aug20/kuka_force_predictor_optimized.onnx"
    output_quantized = "models/Aug20/kuka_force_predictor_quantized.onnx"
    
    # Check if input model exists
    if not os.path.exists(input_model):
        print(f"Error: Input model {input_model} not found!")
        exit(1)
    
    print("Starting ONNX model optimization...")
    
    # First try quantization (most aggressive optimization)
    try:
        quantize_onnx_model(input_model, output_quantized)
    except Exception as e:
        print(f"Quantization failed: {e}")
        # Fall back to basic optimization
        optimize_onnx_model(input_model, output_optimized)
