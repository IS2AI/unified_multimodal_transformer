import numpy as np
import matplotlib.pyplot as plt

# dataloader
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torchaudio
from torchvision import transforms

def plot_sample(data_wav, data_rgb, data_thr, label):

    '''
    Parameters:
        data_wav: torch.Tensor, torch.Size([1, 64, 94])
        data_rgb: torch.Tensor, torch.Size([3, 128, 128])
        data_thr: torch.Tensor, torch.Size([3, 128, 128])
        label: torch.Tensor
    Returns:
        Plot all three data. 
    '''

    rows = 1 # 
    columns = 3 # data_wav, data_rgb, data_thr

    fig = plt.figure(figsize=(3*columns, 2*rows))
    grid = plt.GridSpec(rows, columns, wspace = .25, hspace = .25,figure=fig)

    row = 0
    cols = 0 
    exec (f"plt.subplot(grid{[row * columns + cols]})")
    plt.imshow(data_rgb.permute(1,2,0))
    plt.title(f"RGB: Label = {label}")
    cols = 1 
    exec (f"plt.subplot(grid{[row * columns + cols]})")
    plt.imshow(data_thr.permute(1,2,0))
    plt.title(f"THERMAL: Label = {label}")
    cols = 2 
    exec (f"plt.subplot(grid{[row * columns + cols]})")
    plt.imshow(data_wav.permute(1,2,0))
    plt.title(f"WAV: Label = {label}")

def plot_image(image):
    image = image.numpy()
    image = np.rollaxis(image,0,3)
    plt.imshow(image)
    plt.show()

def plot_wav(wav):
    plt.imshow(wav.squeeze().cpu())
    plt.show()


def _get_dataloaders(dataset, train_size=0.6, val_size=0.2, test_size=0.2):
    N_dataset = len(dataset)
    indices = list(range(N_dataset))

    split_val = int(np.floor(N_dataset * train_size))
    split_test = int(np.floor(N_dataset * (train_size + val_size)))

    np.random.seed(42)
    np.random.shuffle(indices)
    train_idx, valid_idx, test_idx = indices[:split_val], indices[split_val:split_test], indices[split_test:]


    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, sampler=train_sampler,shuffle=False,
        num_workers=2, pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, sampler=valid_sampler,shuffle=False,
        num_workers=2, pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, sampler=test_sampler,shuffle=False,
        num_workers=2, pin_memory=True,
    )

    return train_loader, valid_loader, test_loader

