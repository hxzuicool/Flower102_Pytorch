import torch


if __name__ == '__main__':
    available = torch.cuda.is_available()
    print(available)

    print(torch.cuda.device_count())

    print(torch.cuda.get_device_capability())

    print(torch.cuda.get_device_name())

    print(torch.cuda.get_device_properties(torch.device(0)))

    if torch.cuda.is_available():
        device = torch.cuda.device('cuda:0')
    else:
        device = torch.cuda.device('cpu')

    print(device)

    device = torch.device('cuda', 0)
    print(device)