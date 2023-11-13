import torch


def print_gpu_info(device):
    print("Available devices ", torch.cuda.device_count())
    print("Current cuda device ", torch.cuda.current_device())
    print(torch.cuda.get_device_name(device))
