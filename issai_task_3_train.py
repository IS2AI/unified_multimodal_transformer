from speaker_verification.dataset import TrainDataset
from speaker_verification.dataset import ValidDataset
from speaker_verification.sampler import ProtoSampler

from speaker_verification.loss import PrototypicalLoss
from speaker_verification.train import train_model
from speaker_verification.parser import createParser
from speaker_verification.transforms import Audio_Transforms
from speaker_verification.transforms import Image_Transforms
from speaker_verification.models import Model

import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import wandb
import os


if __name__== "__main__":

    parser = createParser()
    namespace = parser.parse_args()

    # Input parameters

    # device
    n_gpu = namespace.n_gpu
    seed_number = 42

    # dataset
    annotations_file = namespace.annotation_file
    path_to_train_dataset = namespace.path_to_train_dataset
    path_to_valid_dataset = namespace.path_to_valid_dataset
    path_to_valid_list = namespace.path_to_valid_list
    dataset_type = namespace.dataset_type

    # model
    library = namespace.library
    model_name = namespace.model_name
    fine_tune = namespace.fine_tune
    pool=namespace.pool
    exp_name = namespace.exp_name
    pretrained_weights = namespace.pretrained_weights
    embedding_size = namespace.embedding_size

    # audio transforms
    sample_rate = namespace.sample_rate
    sample_duration=namespace.sample_duration # seconds
    n_fft=namespace.n_fft # from Korean code
    win_length=namespace.win_length
    hop_length=namespace.hop_length
    n_mels=namespace.n_mels

    # image transform 
    image_resize = namespace.image_resize

    # train_dataloader
    n_batch = namespace.n_batch
    n_ways = namespace.n_ways
    n_support = namespace.n_support
    n_query = namespace.n_query

    #valid_dataloader
    batch_size = namespace.batch_size

    # loss
    dist_type = namespace.dist_type

    # optimizer
    weight_decay = namespace.weight_decay

    # train 
    num_epochs = namespace.num_epochs
    save_dir = namespace.save_dir
    data_type = namespace.data_type # ['wav', 'rgb']
    wandb_use = namespace.wandb


    input_parameters = {}
    input_parameters["n_gpu"] = n_gpu
    input_parameters["seed_number"] = seed_number

    input_parameters["annotation_file"] = annotations_file
    input_parameters["path_to_train_dataset"] = path_to_train_dataset
    input_parameters["path_to_valid_dataset"] = path_to_valid_dataset
    input_parameters["path_to_valid_list"] = path_to_valid_list
    input_parameters["dataset_type"] = dataset_type

    input_parameters["n_batch"] = n_batch
    input_parameters["n_ways"] = n_ways
    input_parameters["n_support"] = n_support
    input_parameters["n_query"] = n_query
    input_parameters["valid_batch_size"] = batch_size
    input_parameters["dist_type"] = dist_type
    input_parameters["library"] = library

    input_parameters["model_name"] = model_name
    input_parameters["fine_tune"] = fine_tune
    input_parameters["pool"] = pool
    input_parameters["exp_name"] = exp_name
    input_parameters["pretrained_weights"] = pretrained_weights
    input_parameters["embedding_size"] = embedding_size

    # audio transform
    input_parameters["sample_rate"] = sample_rate
    input_parameters["sample_duration"] = sample_duration
    input_parameters["n_fft"] = n_fft
    input_parameters["win_length"] = win_length
    input_parameters["hop_length"] = hop_length
    input_parameters["n_mels"] = n_mels

    # image transform
    input_parameters["image_resize"] = image_resize

    input_parameters["num_epochs"] = num_epochs
    input_parameters["save_dir"] = save_dir
    input_parameters["data_type"] = data_type
    input_parameters["weight_decay"] = weight_decay


    # Wandb
    if wandb_use:
        wandb.login(key="77340cc6897c81e12e0e5beccd2bc6f29a1eacec")
        run = wandb.init(# Set the project where this run will be logged
                        project="Speaker Verification",
                        # Track hyperparameters and run metadata
                        name=f"{exp_name}", 
                        config={
                            "n_batch": n_batch,
                            "n_ways": n_ways,
                            "n_support": n_support,
                            "n_query": n_query,
                            "valid_batch_size": batch_size,
                            "dist_type": dist_type,
                            "library": library,
                            "model_name": model_name,
                            "fine_tune": fine_tune,
                            "pool": pool,
                            "exp_name": exp_name,
                            "pretrained_weights": pretrained_weights,
                            "embedding_size": embedding_size,
                            "sample_rate": sample_rate,
                            "sample_duration": sample_duration,
                            "n_fft": n_fft,
                            "win_length": win_length,
                            "hop_length": hop_length,
                            "n_mels": n_mels,
                            "image_resize": image_resize,
                            "num_epochs": num_epochs,
                            "data_type": data_type,
                            "weight_decay": weight_decay,
                        })
    else:
        wandb = None


    torch.save(input_parameters,f'{save_dir}/{data_type[0]}_{exp_name}_input_parameters')
    #------------------------------------------------------------------

    torch.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    np.random.seed(seed_number)
    random.seed(seed_number)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)

    device = torch.device(f"cuda:{str(n_gpu)}" if torch.cuda.is_available() else "cpu")

    model = Model(library=library, 
            pretrained_weights=pretrained_weights, 
            fine_tune=fine_tune, 
            embedding_size=embedding_size,
            model_name = model_name,
            pool=pool,
            data_type=data_type)

    model = model.to(device)

    audio_T = None
    image_T = None

    if 'wav' in data_type:
        audio_T = Audio_Transforms(sample_rate=sample_rate,
                                    sample_duration=sample_duration,
                                    n_fft=n_fft, 
                                    win_length=win_length,
                                    hop_length=hop_length,
                                    window_fn=torch.hamming_window,
                                    n_mels=n_mels,
                                    model_name=model_name,
                                    library=library)
        audio_T = audio_T.transform

    if 'rgb' in data_type or 'thr' in data_type:
        image_T = Image_Transforms(model_name=model_name,
                                library=library)
        image_T = image_T.transform  

    # Dataset
    train_dataset = TrainDataset(annotations_file=annotations_file, 
                                path_to_train_dataset=path_to_train_dataset, 
                                data_type=data_type, 
                                dataset_type=dataset_type,
                                train_type = 'train',
                                image_transform=image_T, 
                                audio_transform=audio_T)


    valid_dataset= ValidDataset(path_to_valid_dataset=path_to_valid_dataset, 
                                path_to_valid_list=path_to_valid_list, 
                                data_type=data_type,
                                dataset_type=dataset_type,
                                image_transform=image_T, 
                                audio_transform=audio_T)

    # sampler
    train_sampler = ProtoSampler(train_dataset.labels,
                                n_batch,
                                n_ways, # n_way
                                n_support, # n_shots
                                n_query)

    # dataloader
    train_dataloader = DataLoader(dataset=train_dataset, 
                            batch_sampler=train_sampler)

    valid_dataloader = DataLoader(dataset=valid_dataset,
                            batch_size=batch_size,
                            shuffle=True)

    # loss
    criterion = PrototypicalLoss(dist_type=dist_type)
    criterion = criterion.to(device)

    # optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3, weight_decay = weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.95)

    # train
    model = train_model(model=model,
                    train_dataloader=train_dataloader, 
                    valid_dataloader=valid_dataloader,
                    train_sampler=train_sampler,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device,
                    num_epochs=num_epochs,
                    save_dir=save_dir,
                    exp_name=exp_name,
                    data_type=data_type,
                    wandb=wandb)

    if wandb_use:
        # # Mark the run as finished
        wandb.finish()




