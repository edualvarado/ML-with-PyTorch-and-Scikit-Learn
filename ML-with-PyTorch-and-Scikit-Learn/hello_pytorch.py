import torch

def main():
    print("PyTorch version:", torch.__version__)

    # Check if CUDA (GPU) is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    # Create a tensor and move it to the selected device
    x = torch.tensor([1.0, 2.0, 3.0])
    x = x.to(device)
    print("Tensor on device:", x)

    # Simple operation
    y = x * 2
    print("Result of x * 2:", y)

if __name__ == "__main__":
    main()