# import torch
# if torch.backends.mps.is_available():
#     mps_device = torch.device("mps")
#     x = torch.ones(1, device=mps_device)
#     print (x)
# else:
#     print ("MPS device not found.")
    
    
import torch
import torchvision

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)


import torch

print("PyTorch version:", torch.__version__)
print("Is MPS available?", torch.backends.mps.is_available())
print("Is MPS built?", torch.backends.mps.is_built())

source automated-detection-of-nematode-Eggs-/venv/bin/activate
pip install -r requirements.txt