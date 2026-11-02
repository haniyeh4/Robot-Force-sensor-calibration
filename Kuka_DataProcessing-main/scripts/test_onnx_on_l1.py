import onnxruntime as ort
import numpy as np
import json
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

def load_l1_data(file_path, max_samples=None):
    """Load L1 data from JSONL file efficiently"""
    data = []
    sample_count = 0
    
    print(f"Loading data from {file_path}...")
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f):
            if max_samples and sample_count >= max_samples:
                break
                
            try:
                record = json.loads(line.strip())
                
                # Extract robot data
                robot_data = record.get('robot_data', {})
                
                # Input features: X_Pos (position) + X_Dot_Cmds (velocity commands)
                x_pos = robot_data.get('X_Pos', [0, 0, 0, 0, 0, 0])
                x_dot_cmds = robot_data.get('X_Dot_Cmds', [0, 0, 0, 0, 0, 0])
                
                # Combine into input vector (9 features)
                input_features = x_pos + x_dot_cmds[:3]  # Only first 3 velocity components
                
                # Target: FF_EE (end-effector forces)
                target_forces = robot_data.get('FF_EE', [0, 0, 0, 0, 0, 0])
                
                data.append({
                    'input': input_features,
                    'target': target_forces,
                    'timestamp': record.get('timestamp', 0)
                })
                
                sample_count += 1
                
                if sample_count % 1000 == 0:
                    print(f"Loaded {sample_count} samples...")
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num + 1}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num + 1}: {e}")
                continue
    
    print(f"Successfully loaded {len(data)} samples")
    return data

def test_onnx_model(model_path, test_data):
    """Test ONNX model on test data and calculate metrics"""
    
    print(f"Loading ONNX model from {model_path}...")
    
    try:
        # Create inference session
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # Get input name
        input_name = session.get_inputs()[0].name
        print(f"Model input name: {input_name}")
        print(f"Model input shape: {session.get_inputs()[0].shape}")
        print(f"Model output shape: {session.get_outputs()[0].shape}")
        
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return None
    
    # Prepare data
    inputs = []
    targets = []
    
    print("Preparing test data...")
    for sample in test_data:
        # Convert to numpy arrays and ensure correct data types
        input_array = np.array(sample['input'], dtype=np.float32).reshape(1, -1)
        target_array = np.array(sample['target'], dtype=np.float32)
        
        inputs.append(input_array)
        targets.append(target_array)
    
    # Run inference
    print("Running inference...")
    predictions = []
    inference_times = []
    
    for i, (input_array, target_array) in enumerate(zip(inputs, targets)):
        start_time = time.time()
        
        try:
            # Run inference
            outputs = session.run(None, {input_name: input_array})
            prediction = outputs[0].flatten()
            
            predictions.append(prediction)
            inference_times.append(time.time() - start_time)
            
            if i % 1000 == 0:
                print(f"Processed {i} samples...")
                
        except Exception as e:
            print(f"Error during inference for sample {i}: {e}")
            # Use zeros as fallback
            predictions.append(np.zeros_like(target_array))
            inference_times.append(0)
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    
    # Overall MSE and MAE
    mse_overall = mean_squared_error(targets, predictions)
    mae_overall = mean_absolute_error(targets, predictions)
    
    # Per-output MSE and MAE
    mse_per_output = []
    mae_per_output = []
    
    for i in range(targets.shape[1]):
        mse_per_output.append(mean_squared_error(targets[:, i], predictions[:, i]))
        mae_per_output.append(mean_absolute_error(targets[:, i], predictions[:, i]))
    
    # Calculate R-squared for each output
    r2_per_output = []
    for i in range(targets.shape[1]):
        ss_res = np.sum((targets[:, i] - predictions[:, i]) ** 2)
        ss_tot = np.sum((targets[:, i] - np.mean(targets[:, i])) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        r2_per_output.append(r2)
    
    # Performance metrics
    avg_inference_time = np.mean(inference_times)
    total_inference_time = np.sum(inference_times)
    
    # Print results
    print("\n" + "="*60)
    print("ONNX MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"Dataset: L1 from Aug20")
    print(f"Total samples: {len(test_data)}")
    print(f"Model: {os.path.basename(model_path)}")
    print()
    
    print("OVERALL METRICS:")
    print(f"  Mean Squared Error (MSE): {mse_overall:.6f}")
    print(f"  Mean Absolute Error (MAE): {mae_overall:.6f}")
    print(f"  Root Mean Squared Error (RMSE): {np.sqrt(mse_overall):.6f}")
    print()
    
    print("PER-OUTPUT METRICS:")
    output_names = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
    for i, name in enumerate(output_names):
        print(f"  {name}:")
        print(f"    MSE: {mse_per_output[i]:.6f}")
        print(f"    MAE: {mae_per_output[i]:.6f}")
        print(f"    R²:  {r2_per_output[i]:.6f}")
    print()
    
    print("PERFORMANCE METRICS:")
    print(f"  Average inference time: {avg_inference_time*1000:.3f} ms")
    print(f"  Total inference time: {total_inference_time:.3f} s")
    print(f"  Throughput: {len(test_data)/total_inference_time:.1f} samples/s")
    print()
    
    # Sample predictions
    print("SAMPLE PREDICTIONS (first 5 samples):")
    print("Sample | Target | Prediction | Error")
    print("-" * 50)
    for i in range(min(5, len(test_data))):
        target = targets[i]
        pred = predictions[i]
        error = np.abs(target - pred)
        print(f"{i+1:6d} | {target[0]:6.2f} | {pred[0]:10.2f} | {error[0]:6.2f}")
    
    return {
        'mse_overall': mse_overall,
        'mae_overall': mae_overall,
        'mse_per_output': mse_per_output,
        'mae_per_output': mae_per_output,
        'r2_per_output': r2_per_output,
        'avg_inference_time': avg_inference_time,
        'total_inference_time': total_inference_time,
        'predictions': predictions,
        'targets': targets
    }

def main():
    """Main function to run the evaluation"""
    
    # File paths
    model_path = "models/Aug20/kuka_force_predictor.onnx"
    data_path = "data/Aug20/2_DyF_L1.jsonl"
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: ONNX model not found at {model_path}")
        return
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
    
    print("ONNX Model Evaluation on L1 Dataset")
    print("="*50)
    
    # Load test data (limit to first 10000 samples for faster testing)
    # Remove max_samples parameter to test on full dataset
    test_data = load_l1_data(data_path, max_samples=10000)
    
    if not test_data:
        print("No test data loaded. Exiting.")
        return
    
    # Test ONNX model
    results = test_onnx_model(model_path, test_data)
    
    if results:
        print("\nEvaluation completed successfully!")
        
        # Save results to file
        output_file = "l1_evaluation_results.json"
        results_to_save = {
            'mse_overall': float(results['mse_overall']),
            'mae_overall': float(results['mae_overall']),
            'mse_per_output': [float(x) for x in results['mse_per_output']],
            'mae_per_output': [float(x) for x in results['mae_per_output']],
            'r2_per_output': [float(x) for x in results['r2_per_output']],
            'avg_inference_time': float(results['avg_inference_time']),
            'total_inference_time': float(results['total_inference_time']),
            'dataset_size': len(test_data)
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"Results saved to {output_file}")
    else:
        print("Evaluation failed!")

if __name__ == "__main__":
    main()









