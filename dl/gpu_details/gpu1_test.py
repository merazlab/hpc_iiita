# pytorch gpu specification details

import torch

device_name = torch.cuda.get_device_name(0)
print(device_name)
no_of_cuda = torch.cuda.device_count()
print(no_of_cuda)
current_cuda =  torch.cuda.current_device()
print(current_cuda)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')


