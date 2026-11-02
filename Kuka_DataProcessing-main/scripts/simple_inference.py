import onnxruntime as ort
import numpy as np

def run_inference():
    """Simple ONNX inference with specific input"""
    
    # Model path
    model_path = "models/Aug20/kuka_force_predictor.onnx"
    
    # Your specific input
    input_data = np.array([160, 170, 190, 0.1, 0.3, 0.5, 0, 0, 0], dtype=np.float32)
    
    try:
        # Create inference session
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # Get input name
        input_name = session.get_inputs()[0].name
        
        # Reshape input to [1, 9] (batch dimension)
        input_reshaped = input_data.reshape(1, -1)
        
        # Run inference
        outputs = session.run(None, {input_name: input_reshaped})
        
        # Get prediction
        prediction = outputs[0]
        
        print(f"Input: {input_data}")
        print(f"Prediction: {prediction.flatten()}")
        
        return prediction
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    print("Running ONNX inference...")
    result = run_inference()









