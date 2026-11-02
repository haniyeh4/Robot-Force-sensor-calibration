import json
import numpy as np

def load_and_analyze_weights():
    """Load and analyze the extracted weights"""
    print("MLP Weight Extraction Summary")
    print("=" * 50)
    
    # Load the extracted weights
    try:
        with open("models/Aug20/mlp_weights_extracted.json", 'r') as f:
            weights = json.load(f)
        
        print(f"Model Type: {weights['model_type']}")
        print(f"Input Size: {weights['input_size']}")
        print(f"Output Size: {weights['output_size']}")
        print(f"Number of Layers: {weights['num_layers']}")
        print(f"Total Parameters: {weights['total_parameters']:,}")
        
        print(f"\nLayer Details:")
        for layer in weights['layers']:
            print(f"  Layer {layer['layer']}: {layer['input_size']} -> {layer['output_size']} ({layer['activation']})")
            print(f"    Weights: {layer['input_size']} × {layer['output_size']} = {layer['input_size'] * layer['output_size']:,} parameters")
            print(f"    Biases: {layer['output_size']:,} parameters")
            print(f"    Total: {layer['input_size'] * layer['output_size'] + layer['output_size']:,} parameters")
        
        return weights
        
    except Exception as e:
        print(f"Error loading weights: {e}")
        return None

def demonstrate_usage():
    """Demonstrate how to use the extracted weights"""
    print(f"\n" + "=" * 50)
    print("Usage Examples:")
    print("=" * 50)
    
    print(f"\n1. Load weights in Python:")
    print(f"   with open('models/Aug20/mlp_weights_extracted.json', 'r') as f:")
    print(f"       weights = json.load(f)")
    
    print(f"\n2. Access specific layer weights:")
    print(f"   layer1_weights = np.array(weights['layers'][0]['weights'])")
    print(f"   layer1_biases = np.array(weights['layers'][0]['biases'])")
    
    print(f"\n3. Use in PyTorch:")
    print(f"   from torch import nn")
    print(f"   model = nn.Sequential(")
    print(f"       nn.Linear(9, 100),")
    print(f"       nn.ReLU(),")
    print(f"       nn.Linear(100, 100),")
    print(f"       nn.ReLU(),")
    print(f"       nn.Linear(100, 6)")
    print(f"   )")
    print(f"   # Load weights from the pytorch_format.json file")
    
    print(f"\n4. Use in NumPy (manual forward pass):")
    print(f"   def forward(x, weights):")
    print(f"       for i, layer in enumerate(weights['layers']):")
    print(f"           w = np.array(layer['weights'])")
    print(f"           b = np.array(layer['biases'])")
    print(f"           x = x @ w + b")
    print(f"           if layer['activation'] == 'relu':")
    print(f"               x = np.maximum(0, x)")
    print(f"       return x")
    
    print(f"\n5. Convert to other formats:")
    print(f"   - PyTorch: Use mlp_weights_pytorch_format.json")
    print(f"   - TensorFlow: Convert weights to appropriate format")
    print(f"   - ONNX: Already have the original model")

def show_file_info():
    """Show information about the created files"""
    print(f"\n" + "=" * 50)
    print("Generated Files:")
    print("=" * 50)
    
    files = [
        ("models/Aug20/mlp_weights_extracted.json", "Human-readable format with layer information"),
        ("models/Aug20/mlp_weights_pytorch_format.json", "PyTorch-compatible format"),
        ("models/Aug20/kuka_force_predictor_quantized.onnx", "Optimized ONNX model (67% smaller)")
    ]
    
    for file_path, description in files:
        try:
            import os
            size = os.path.getsize(file_path) / 1024  # KB
            print(f"  {file_path}")
            print(f"    Size: {size:.1f} KB")
            print(f"    Description: {description}")
        except:
            print(f"  {file_path} - {description}")

if __name__ == "__main__":
    # Load and analyze weights
    weights = load_and_analyze_weights()
    
    if weights is not None:
        # Show usage examples
        demonstrate_usage()
        
        # Show file information
        show_file_info()
        
        print(f"\n" + "=" * 50)
        print("🎉 Weight extraction completed successfully!")
        print("You can now use these weights in your own ML frameworks.")
        print("The original error about 'Gemm' nodes was because this is a")
        print("scikit-learn MLP model, not a PyTorch model.")


