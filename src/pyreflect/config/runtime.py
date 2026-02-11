import torch

DEVICE = torch.device("cpu")  # Force CPU
#DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device for model training: {DEVICE}')
