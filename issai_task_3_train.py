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

import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import timm

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
    DATASET_DIR = namespace.dataset_dir
    PATH2DATASET = "/workdir/sf_pv"

    # train_dataloader
    n_batch = namespace.n_batch
    n_ways = namespace.n_ways
    n_support = namespace.n_support
    n_query = namespace.n_query

    #valid_dataloader
    valid_batch_size = namespace.valid_batch_size

    # loss
    dist_type = namespace.dist_type

    # model
    model_choice = namespace.model_choice
    fine_tune = namespace.fine_tune
    exp_name = namespace.exp_name

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
    input_parameters["valid_batch_size"] = valid_batch_size
    input_parameters["dist_type"] = dist_type
    input_parameters["model_choice"] = model_choice
    input_parameters["fine_tune"] = fine_tune
    input_parameters["exp_name"] = exp_name
    input_parameters["num_epochs"] = num_epochs
    input_parameters["save_dir"] = save_dir
    input_parameters["modality"] = modality

    torch.save(input_parameters,f'{save_dir}/{modality}_{exp_name}_input_parameters')
    #------------------------------------------------------------------

    torch.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    np.random.seed(seed_number)
    random.seed(seed_number)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True

    device = torch.device(f"cuda:{str(n_gpu)}" if torch.cuda.is_available() else "cpu")

    # model
    if model_choice == "resnet1":
        model = ResNet(pretrained_weights=True, 
                        fine_tune=fine_tune, 
                        embedding_size=128, 
                        modality = modality, 
                        filter_size="default", 
                        from_torch=True
                    )
        image_transform = T.image_transform
        
    elif model_choice == "resnet2":
        if modality == "wav":
            in_channels = 1
        else:
            in_channels = 3
        model = timm.create_model('resnet34', pretrained=True, num_classes=128, in_chans=in_channels)
        image_transform = T.image_transform

    elif model_choice == "resnet3":
        if modality == "wav":
            in_channels = 1
            model = timm.create_model('resnet34', pretrained=True, num_classes=128, in_chans=in_channels)
            model.global_pool = SelfAttentivePool2d()
        else:
            in_channels = 3
            model = timm.create_model('resnet34', pretrained=True, num_classes=128, in_chans=in_channels)
        image_transform = T.image_transform

    elif model_choice == "vit1":
        model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=128)
        img_transform  = T.Image_Transforms(model)
        image_transform = img_transform.timm

    if fine_tune:
        for param in model.parameters():
            param.requires_grad = True

    model = model.to(device)

    # Dataset
    train_dataset = SpeakingFacesDataset(ANNOTATIONS_FILE,DATASET_DIR,'train',
                                        image_transform=image_transform, 
                                        audio_transform=T.audio_transform)
    valid_dataset = ValidDataset(PATH2DATASET,'valid',
                            image_transform=image_transform, 
                            audio_transform=T.audio_transform)

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
                            batch_size=valid_batch_size,
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
                        modality)




