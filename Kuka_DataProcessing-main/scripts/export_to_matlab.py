import onnx
import numpy as np
import scipy.io as sio
import os

def export_onnx_to_matlab(model_path, output_path):
    """
    Export ONNX model weights to MATLAB .mat file
    Creates W1,b1,W2,b2,W3,b3 variables (all float32)
    """
    print(f"Exporting ONNX weights to MATLAB: {output_path}")
    print("=" * 50)
    
    try:
        # Load the ONNX model
        model = onnx.load(model_path)
        graph = model.graph
        
        # Find all initializers (weights and biases)
        initializers = {init.name: init for init in graph.initializer}
        
        print(f"Found {len(initializers)} initializers:")
        for name, init in initializers.items():
            shape = list(init.dims)
            print(f"  {name}: shape {shape}, dtype {init.data_type}")
        
        # Extract weights and biases for each layer
        weights_dict = {}
        
        # Layer 1: MatMul 1 -> Add 2
        if 'coefficient' in initializers and 'intercepts' in initializers:
            w1 = onnx.numpy_helper.to_array(initializers['coefficient'])
            b1 = onnx.numpy_helper.to_array(initializers['intercepts'])
            
            # Convert to float32 and store
            weights_dict['W1'] = w1.astype(np.float32)
            weights_dict['b1'] = b1.astype(np.float32)
            
            print(f"Layer 1: W1 shape {w1.shape}, b1 shape {b1.shape}")
        
        # Layer 2: MatMul 4 -> Add 5
        if 'coefficient1' in initializers and 'intercepts1' in initializers:
            w2 = onnx.numpy_helper.to_array(initializers['coefficient1'])
            b2 = onnx.numpy_helper.to_array(initializers['intercepts1'])
            
            # Convert to float32 and store
            weights_dict['W2'] = w2.astype(np.float32)
            weights_dict['b2'] = b2.astype(np.float32)
            
            print(f"Layer 2: W2 shape {w2.shape}, b2 shape {b2.shape}")
        
        # Layer 3: MatMul 7 -> Add 8
        if 'coefficient2' in initializers and 'intercepts2' in initializers:
            w3 = onnx.numpy_helper.to_array(initializers['coefficient2'])
            b3 = onnx.numpy_helper.to_array(initializers['intercepts2'])
            
            # Convert to float32 and store
            weights_dict['W3'] = w3.astype(np.float32)
            weights_dict['b3'] = b3.astype(np.float32)
            
            print(f"Layer 3: W3 shape {w3.shape}, b3 shape {b3.shape}")
        
        # Verify all weights are extracted
        expected_vars = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']
        missing_vars = [var for var in expected_vars if var not in weights_dict]
        
        if missing_vars:
            print(f"Warning: Missing variables: {missing_vars}")
            return False
        
        print(f"\nSuccessfully extracted all weights:")
        for var_name, var_data in weights_dict.items():
            print(f"  {var_name}: shape {var_data.shape}, dtype {var_data.dtype}")
        
        # Save to MATLAB .mat file
        sio.savemat(output_path, weights_dict)
        print(f"\n✓ Weights saved to MATLAB file: {output_path}")
        
        # Verify file was created and show size
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / 1024  # KB
            print(f"File size: {file_size:.1f} KB")
            
            # Show summary of what was saved
            print(f"\nVariables saved to {output_path}:")
            for var_name, var_data in weights_dict.items():
                print(f"  {var_name}: {var_data.shape} ({var_data.dtype})")
        
        return True
        
    except Exception as e:
        print(f"✗ Error exporting to MATLAB: {e}")
        return False

def verify_matlab_file(file_path):
    """Verify the MATLAB file can be loaded and contains expected variables"""
    print(f"\n" + "=" * 50)
    print(f"Verifying MATLAB file: {file_path}")
    print("=" * 50)
    
    try:
        # Load the .mat file
        mat_contents = sio.loadmat(file_path)
        
        print(f"File loaded successfully!")
        print(f"Variables found: {list(mat_contents.keys())}")
        
        # Check for expected variables (excluding MATLAB metadata)
        expected_vars = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']
        matlab_vars = [key for key in mat_contents.keys() if not key.startswith('__')]
        
        print(f"\nExpected variables: {expected_vars}")
        print(f"Found variables: {matlab_vars}")
        
        # Verify each variable
        for var_name in expected_vars:
            if var_name in mat_contents:
                var_data = mat_contents[var_name]
                print(f"  ✓ {var_name}: shape {var_data.shape}, dtype {var_data.dtype}")
            else:
                print(f"  ✗ {var_name}: NOT FOUND")
        
        # Check data types
        print(f"\nData type verification:")
        for var_name in expected_vars:
            if var_name in mat_contents:
                var_data = mat_contents[var_name]
                if var_data.dtype == np.float32:
                    print(f"  ✓ {var_name}: float32")
                else:
                    print(f"  ⚠ {var_name}: {var_data.dtype} (not float32)")
        
        return True
        
    except Exception as e:
        print(f"✗ Error verifying MATLAB file: {e}")
        return False

if __name__ == "__main__":
    # Input and output paths
    onnx_model_path = "models/Aug20/kuka_force_predictor.onnx"
    matlab_output_path = "models/Aug20/force_mlp_params.mat"
    
    print("ONNX to MATLAB Weight Export")
    print("=" * 50)
    print(f"Input ONNX: {onnx_model_path}")
    print(f"Output MATLAB: {matlab_output_path}")
    
    # Export weights to MATLAB
    success = export_onnx_to_matlab(onnx_model_path, matlab_output_path)
    
    if success:
        # Verify the exported file
        verify_matlab_file(matlab_output_path)
        
        print(f"\n" + "=" * 50)
        print("🎉 Export completed successfully!")
        print(f"MATLAB file created: {matlab_output_path}")
        print("You can now load this file in MATLAB using:")
        print(f"  load('{matlab_output_path}')")
        print("This will create variables: W1, b1, W2, b2, W3, b3")
    else:
        print("Export failed. Check the error messages above.")
