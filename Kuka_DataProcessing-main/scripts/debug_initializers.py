import onnx
import numpy as np

def debug_initializers(model_path):
    """Debug the initializers to understand their data format"""
    print(f"Debugging initializers in: {model_path}")
    print("=" * 50)
    
    try:
        # Load the model
        model = onnx.load(model_path)
        graph = model.graph
        
        # Examine each initializer in detail
        for i, init in enumerate(graph.initializer):
            print(f"\nInitializer {i}: {init.name}")
            print(f"  Data type: {init.data_type}")
            print(f"  Dimensions: {list(init.dims)}")
            print(f"  Raw data size: {len(init.raw_data)} bytes")
            
            # Try to get the data using numpy helper (this is the correct way)
            try:
                data = onnx.numpy_helper.to_array(init)
                print(f"  ✓ Numpy helper data: shape {data.shape}, dtype {data.dtype}")
                print(f"  First 5 values: {data.flatten()[:5]}")
                print(f"  Data range: [{data.min():.6f}, {data.max():.6f}]")
                print(f"  Data mean: {data.mean():.6f}")
                print(f"  Data std: {data.std():.6f}")
            except Exception as e:
                print(f"  ✗ Error with numpy helper: {e}")
            
            print("-" * 30)
        
    except Exception as e:
        print(f"✗ Error debugging initializers: {e}")

if __name__ == "__main__":
    model_path = "models/Aug20/kuka_force_predictor.onnx"
    debug_initializers(model_path)
