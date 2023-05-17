from speaker_verification import transforms as T
from speaker_verification.dataset import SpeakingFacesDataset
from speaker_verification.dataset import ValidDataset
from speaker_verification.sampler import ProtoSampler
from speaker_verification.sampler import ValidSampler
from speaker_verification.models import ResNet
from speaker_verification.models import SelfAttentivePool2d
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
import timm
import wandb

from timeit import default_timer as timer


if __name__== "__main__":

    parser = createParser()
    namespace = parser.parse_args()

    # Input parameters

    # device
    n_gpu = namespace.n_gpu
    seed_number = 42

    # dataset
    ANNOTATIONS_FILE = namespace.annotation_file
    PATH2DATASET = namespace.path2dataset
    DATASET_DIR = f"{PATH2DATASET}/data_v2"

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

    # train 
    num_epochs = namespace.num_epochs
    save_dir = namespace.save_dir
    modality = namespace.modality


    input_parameters = {}
    input_parameters["n_gpu"] = n_gpu
    input_parameters["seed_number"] = seed_number
    input_parameters["annotation_file"] = ANNOTATIONS_FILE
    input_parameters["dataset_dir"] = DATASET_DIR
    input_parameters["path2dataset"] = PATH2DATASET
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
    input_parameters["modality"] = modality


    # Wandb
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
                        "modality": modality,
                    })


    torch.save(input_parameters,f'{save_dir}/{modality}_{exp_name}_input_parameters')
    #------------------------------------------------------------------

    torch.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    np.random.seed(seed_number)
    random.seed(seed_number)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True

    device = torch.device(f"cuda:{str(n_gpu)}" if torch.cuda.is_available() else "cpu")

    model = Model(library=library, 
                    pretrained_weights=pretrained_weights, 
                    fine_tune=fine_tune, 
                    embedding_size=embedding_size, 
                    modality = modality,
                    model_name = model_name,
                    pool=pool)

    model = model.to(device)

    audio_T = Audio_Transforms(sample_rate=sample_rate,
                                sample_duration=sample_duration, # seconds
                                n_fft=n_fft, # from Korean code
                                win_length=win_length,
                                hop_length=hop_length,
                                window_fn=torch.hamming_window,
                                n_mels=n_mels)

    image_T = Image_Transforms(model,
                                library=library,
                                model_name = model_name,
                                resize=image_resize)

    # Dataset
    train_dataset = SpeakingFacesDataset(ANNOTATIONS_FILE,DATASET_DIR,'train',
                                    image_transform=image_T.transform, 
                                    audio_transform=audio_T.transform)

    valid_dataset = ValidDataset(PATH2DATASET,'valid',
                                image_transform=image_T.transform, 
                                audio_transform=audio_T.transform)

    # sampler
    train_sampler = ProtoSampler(train_dataset.labels,
                                n_batch,
                                n_ways, # n_way
                                n_support, # n_shots
                                n_query)

    # dataloader
    train_dataloader = DataLoader(dataset=train_dataset, 
                            batch_sampler=train_sampler,
                            num_workers=4, pin_memory=True
                            )

    valid_dataloader = DataLoader(dataset=valid_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=4, 
                            pin_memory=True)

    # loss
    criterion = PrototypicalLoss(dist_type=dist_type)
    criterion = criterion.to(device)

    # optimizer + scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.95)

    # train
    model = train_model(model,
                        train_dataloader, 
                        valid_dataloader,
                        train_sampler,
                        criterion,
                        optimizer,
                        scheduler,
                        device,
                        num_epochs,
                        save_dir,
                        exp_name,
                        modality,
                        wandb)

    # # Mark the run as finished
    wandb.finish()




