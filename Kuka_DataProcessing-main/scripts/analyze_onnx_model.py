import onnx
import onnxruntime as ort
import numpy as np

def analyze_onnx_model(model_path):
    """Analyze the structure of an ONNX model"""
    print(f"Analyzing ONNX model: {model_path}")
    print("=" * 50)
    
    try:
        # Load the model
        model = onnx.load(model_path)
        print(f"✓ Model loaded successfully")
        
        # Basic model info
        print(f"Model IR version: {model.ir_version}")
        print(f"Opset version: {model.opset_import[0].version}")
        print(f"Producer: {model.producer_name}")
        
        # Graph info
        graph = model.graph
        print(f"\nGraph name: {graph.name}")
        print(f"Number of inputs: {len(graph.input)}")
        print(f"Number of outputs: {len(graph.output)}")
        print(f"Number of nodes: {len(graph.node)}")
        
        # Input details
        print(f"\nInputs:")
        for i, input_info in enumerate(graph.input):
            print(f"  {i}: {input_info.name} - Shape: {[dim.dim_value if hasattr(dim, 'dim_value') else '?' for dim in input_info.type.tensor_type.shape.dim]}")
        
        # Output details
        print(f"\nOutputs:")
        for i, output_info in enumerate(graph.output):
            print(f"  {i}: {output_info.name} - Shape: {[dim.dim_value if hasattr(dim, 'dim_value') else '?' for dim in output_info.type.tensor_type.shape.dim]}")
        
        # Node analysis
        print(f"\nNodes (operations):")
        op_types = {}
        for i, node in enumerate(graph.node):
            op_type = node.op_type
            if op_type not in op_types:
                op_types[op_type] = []
            op_types[op_type].append(i)
            
            print(f"  {i}: {op_type} - Inputs: {node.input} - Outputs: {node.output}")
        
        # Summary of operation types
        print(f"\nOperation type summary:")
        for op_type, indices in op_types.items():
            print(f"  {op_type}: {len(indices)} nodes {indices}")
        
        # Check for specific operation types you might be looking for
        print(f"\nLooking for specific operations:")
        gemm_nodes = [n for n in graph.node if n.op_type == "Gemm"]
        matmul_nodes = [n for n in graph.node if n.op_type == "MatMul"]
        add_nodes = [n for n in graph.node if n.op_type == "Add"]
        mul_nodes = [n for n in graph.node if n.op_type == "Mul"]
        
        print(f"  Gemm nodes: {len(gemm_nodes)}")
        print(f"  MatMul nodes: {len(matmul_nodes)}")
        print(f"  Add nodes: {len(add_nodes)}")
        print(f"  Mul nodes: {len(mul_nodes)}")
        
        # If no Gemm nodes, show what the actual linear operations might be
        if len(gemm_nodes) == 0:
            print(f"\nNo Gemm nodes found. Linear operations might be implemented as:")
            print(f"  - MatMul + Add (separate operations)")
            print(f"  - Conv operations (for 1D convolutions)")
            print(f"  - Other custom operations")
            
            # Show MatMul + Add combinations that might form linear layers
            if len(matmul_nodes) > 0 and len(add_nodes) > 0:
                print(f"\nPotential linear layer combinations (MatMul + Add):")
                for i, matmul_node in enumerate(graph.node):
                    if matmul_node.op_type == "MatMul":
                        for j, add_node in enumerate(graph.node):
                            if add_node.op_type == "Add":
                                # Check if Add follows MatMul
                                if add_node.input[0] == matmul_node.output[0]:
                                    print(f"  MatMul {i} -> Add {j}")
        
        return model, graph
        
    except Exception as e:
        print(f"✗ Error analyzing model: {e}")
        return None, None

def test_inference(model_path):
    """Test basic inference to understand input/output requirements"""
    print(f"\n" + "=" * 50)
    print(f"Testing inference...")
    
    try:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # Get input/output info
        input_names = [input.name for input in session.get_inputs()]
        output_names = [output.name for output in session.get_outputs()]
        
        print(f"Input names: {input_names}")
        print(f"Output names: {output_names}")
        
        # Get input shapes
        input_shapes = [input.shape for input in session.get_inputs()]
        print(f"Input shapes: {input_shapes}")
        
        # Create dummy input data
        dummy_inputs = {}
        for i, input_name in enumerate(input_names):
            shape = input_shapes[i]
            # Replace dynamic dimensions with fixed values for testing
            test_shape = [1 if dim is None else dim for dim in shape]
            dummy_inputs[input_name] = np.random.random(test_shape).astype(np.float32)
        
        # Run inference
        outputs = session.run(output_names, dummy_inputs)
        print(f"✓ Inference successful!")
        print(f"Output shapes: {[out.shape for out in outputs]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        return False

if __name__ == "__main__":
    model_path = "models/Aug20/kuka_force_predictor.onnx"
    
    # Analyze the model structure
    model, graph = analyze_onnx_model(model_path)
    
    if model is not None:
        # Test inference
        test_inference(model_path)
        
        print(f"\n" + "=" * 50)
        print(f"Analysis complete!")
        print(f"Use this information to understand how to properly extract weights and biases from your model.")
