import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    count = torch.cuda.device_count()
    print(f"Device Count: {count}")
    for i in range(count):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No CUDA devices found.")
