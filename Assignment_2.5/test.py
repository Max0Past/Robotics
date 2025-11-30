import torch
import sys
from ultralytics import checks

def verify_environment():
    """
    Performs a comprehensive check of the GPU environment for YOLO training.
    """
    print("--- Environment Verification Started ---\n")

    # 1. Python and System info
    print(f"Python version: {sys.version.split()[0]}")
    
    # 2. PyTorch CUDA check
    print("\n[PyTorch Check]")
    print(f"PyTorch Version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        
        print(f"GPU Device Count: {device_count}")
        print(f"Current Device ID: {current_device}")
        print(f"Device Name: {device_name}")
        
        # Check CUDA version compiled with PyTorch
        print(f"CUDA Version (PyTorch build): {torch.version.cuda}")
        
        # Check CuDNN version (essential for convolution optimization)
        print(f"CuDNN Version: {torch.backends.cudnn.version()}")
    else:
        print("WARNING: PyTorch is running on CPU. Training will be extremely slow.")
        print("Possible causes: Incorrect PyTorch installation or missing drivers.")

    # 3. Ultralytics internal check
    print("\n[Ultralytics Internal Check]")
    try:
        # ultralytics.checks() prints hardware info automatically to stdout
        checks() 
    except Exception as e:
        print(f"Error running ultralytics checks: {e}")

    # 4. Simple Tensor Test
    if cuda_available:
        print("\n[Tensor Computation Test]")
        try:
            x = torch.rand(5, 3).to('cuda')
            print("Successfully moved tensor to GPU.")
            y = x * x
            print("Successfully performed operation on GPU.")
        except Exception as e:
            print(f"Error during tensor operation: {e}")

    print("\n--- Verification Finished ---")

if __name__ == "__main__":
    verify_environment()