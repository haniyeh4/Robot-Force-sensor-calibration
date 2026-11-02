import onnx
import numpy as np
import json
import os

def extract_mlp_weights_from_onnx(model_path):
    """
    Extract weights and biases from a scikit-learn MLP ONNX model
    The model uses MatMul + Add operations instead of Gemm
    """
    print(f"Extracting MLP weights from: {model_path}")
    print("=" * 50)
    
    try:
        # Load the model
        model = onnx.load(model_path)
        graph = model.graph
        
        # Find all initializers (weights and biases)
        initializers = {init.name: init for init in graph.initializer}
        
        print(f"Found {len(initializers)} initializers:")
        for name, init in initializers.items():
            shape = list(init.dims)
            print(f"  {name}: shape {shape}, dtype {init.data_type}")
        
        # Extract weights and biases for each layer
        layers = []
        
        # Layer 1: MatMul 1 -> Add 2
        if 'coefficient' in initializers and 'intercepts' in initializers:
            try:
                w1 = onnx.numpy_helper.to_array(initializers['coefficient'])
                b1 = onnx.numpy_helper.to_array(initializers['intercepts'])
                
                layers.append({
                    'layer': 1,
                    'type': 'hidden',
                    'weights': w1.tolist(),
                    'biases': b1.tolist(),
                    'input_size': w1.shape[0],
                    'output_size': w1.shape[1],
                    'activation': 'relu'
                })
                print(f"Layer 1: {w1.shape[0]} -> {w1.shape[1]} (ReLU)")
            except Exception as e:
                print(f"Error processing Layer 1: {e}")
        
        # Layer 2: MatMul 4 -> Add 5
        if 'coefficient1' in initializers and 'intercepts1' in initializers:
            try:
                w2 = onnx.numpy_helper.to_array(initializers['coefficient1'])
                b2 = onnx.numpy_helper.to_array(initializers['intercepts1'])
                
                layers.append({
                    'layer': 2,
                    'type': 'hidden',
                    'weights': w2.tolist(),
                    'biases': b2.tolist(),
                    'input_size': w2.shape[0],
                    'output_size': w2.shape[1],
                    'activation': 'relu'
                })
                print(f"Layer 2: {w2.shape[0]} -> {w2.shape[1]} (ReLU)")
            except Exception as e:
                print(f"Error processing Layer 2: {e}")
        
        # Layer 3: MatMul 7 -> Add 8
        if 'coefficient2' in initializers and 'intercepts2' in initializers:
            try:
                w3 = onnx.numpy_helper.to_array(initializers['coefficient2'])
                b3 = onnx.numpy_helper.to_array(initializers['intercepts2'])
                
                layers.append({
                    'layer': 3,
                    'type': 'output',
                    'weights': w3.tolist(),
                    'biases': b3.tolist(),
                    'input_size': w3.shape[0],
                    'output_size': w3.shape[1],
                    'activation': 'linear'
                })
                print(f"Layer 3: {w3.shape[0]} -> {w3.shape[1]} (Linear)")
            except Exception as e:
                print(f"Error processing Layer 3: {e}")
        
        print(f"Successfully processed {len(layers)} layers")
        
        if len(layers) == 0:
            print("No layers were processed successfully!")
            return None, None
        
        # Create model summary
        model_info = {
            'model_type': 'MLP (scikit-learn)',
            'input_size': 9,
            'output_size': 6,
            'num_layers': len(layers),
            'layers': layers,
            'total_parameters': sum(
                layer['input_size'] * layer['output_size'] + layer['output_size'] 
                for layer in layers
            )
        }
        
        print(f"\nModel Summary:")
        print(f"  Input size: {model_info['input_size']}")
        print(f"  Output size: {model_info['output_size']}")
        print(f"  Number of layers: {model_info['num_layers']}")
        print(f"  Total parameters: {model_info['total_parameters']:,}")
        
        return model_info, initializers
        
    except Exception as e:
        print(f"✗ Error extracting weights: {e}")
        return None, None

def save_weights_to_json(model_info, output_path):
    """Save the extracted weights to a JSON file"""
    try:
        with open(output_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        print(f"✓ Weights saved to: {output_path}")
        return True
    except Exception as e:
        print(f"✗ Error saving weights: {e}")
        return False

def create_pytorch_like_structure(model_info):
    """Create a PyTorch-like structure for the weights"""
    pytorch_weights = {}
    
    for i, layer in enumerate(model_info['layers']):
        layer_num = i + 1
        
        # Convert to PyTorch format (transpose weights for linear layers)
        weights = np.array(layer['weights'])
        biases = np.array(layer['biases'])
        
        # For scikit-learn MLP, weights are already in the correct orientation
        # (input_features, output_features)
        
        pytorch_weights[f'layer{layer_num}.weight'] = weights.T.tolist()  # Transpose for PyTorch
        pytorch_weights[f'layer{layer_num}.bias'] = biases.flatten().tolist()
    
    return pytorch_weights

def save_pytorch_format(pytorch_weights, output_path):
    """Save weights in PyTorch-like format"""
    try:
        with open(output_path, 'w') as f:
            json.dump(pytorch_weights, f, indent=2)
        print(f"✓ PyTorch format weights saved to: {output_path}")
        return True
    except Exception as e:
        print(f"✗ Error saving PyTorch format: {e}")
        return False

if __name__ == "__main__":
    model_path = "models/Aug20/kuka_force_predictor.onnx"
    
    # Extract weights
    model_info, initializers = extract_mlp_weights_from_onnx(model_path)
    
    if model_info is not None:
        # Save in readable format
        json_path = "models/Aug20/mlp_weights_extracted.json"
        save_weights_to_json(model_info, json_path)
        
        # Save in PyTorch-like format
        pytorch_weights = create_pytorch_like_structure(model_info)
        pytorch_path = "models/Aug20/mlp_weights_pytorch_format.json"
        save_pytorch_format(pytorch_weights, pytorch_path)
        
        print(f"\n" + "=" * 50)
        print(f"Weight extraction complete!")
        print(f"Files created:")
        print(f"  - {json_path} (readable format)")
        print(f"  - {pytorch_path} (PyTorch-like format)")
    else:
        print("Failed to extract weights from the model.")
