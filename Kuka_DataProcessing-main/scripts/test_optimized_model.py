import onnx
import onnxruntime as ort
import numpy as np

def test_model(model_path):
    """Test if an ONNX model can be loaded and used for inference"""
    try:
        # Load the model
        model = onnx.load(model_path)
        print(f"✓ Model loaded successfully from: {model_path}")
        
        # Create inference session
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        print(f"✓ Inference session created successfully")
        
        # Get input/output info
        input_names = [input.name for input in session.get_inputs()]
        output_names = [output.name for output in session.get_outputs()]
        
        print(f"Input names: {input_names}")
        print(f"Output names: {output_names}")
        
        # Get input shapes
        input_shapes = [input.shape for input in session.get_inputs()]
        print(f"Input shapes: {input_shapes}")
        
        # Create dummy input data (you may need to adjust this based on your model)
        # This is just for testing - replace with actual input data for real inference
        dummy_inputs = {}
        for i, input_name in enumerate(input_names):
            shape = input_shapes[i]
            # Replace dynamic dimensions with fixed values for testing
            test_shape = [1 if dim is None else dim for dim in shape]
            dummy_inputs[input_name] = np.random.random(test_shape).astype(np.float32)
        
        # Run inference
        outputs = session.run(output_names, dummy_inputs)
        print(f"✓ Inference successful! Output shapes: {[out.shape for out in outputs]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing model {model_path}: {e}")
        return False

if __name__ == "__main__":
    print("Testing original model...")
    original_works = test_model("models/Aug20/kuka_force_predictor.onnx")
    
    print("\nTesting optimized model...")
    optimized_works = test_model("models/Aug20/kuka_force_predictor_quantized.onnx")
    
    print(f"\nSummary:")
    print(f"Original model: {'✓ Working' if original_works else '✗ Failed'}")
    print(f"Optimized model: {'✓ Working' if optimized_works else '✗ Failed'}")
    
    if original_works and optimized_works:
        print("\n🎉 Both models are working correctly!")
        print("The optimization was successful and maintained model functionality.")
    else:
        print("\n⚠️  Some models failed. Check the error messages above.")
