from speaker_verification import transforms as T
from speaker_verification.dataset import SpeakingFacesDataset
from speaker_verification.sampler import ProtoSampler
from speaker_verification.sampler import ValidSampler
from speaker_verification.models_handmade.resnet import ResNet34
from speaker_verification.loss import PrototypicalLoss
from speaker_verification.train import train_model
from speaker_verification.parser import createParser

import torch
from torch.utils.data import DataLoader
import numpy as np
import random
  
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

    # train_dataloader
    n_batch = namespace.n_batch
    n_ways = namespace.n_ways
    n_support = namespace.n_support
    n_query = namespace.n_query

    # train 
    num_epochs = namespace.num_epochs
    save_dir = namespace.save_dir

    input_parameters = {}
    input_parameters["n_gpu"] = n_gpu
    input_parameters["seed_number"] = seed_number
    input_parameters["annotation_file"] = ANNOTATIONS_FILE
    input_parameters["dataset_dir"] = DATASET_DIR
    input_parameters["n_batch"] = n_batch
    input_parameters["n_ways"] = n_ways
    input_parameters["n_support"] = n_support
    input_parameters["n_query"] = n_query
    input_parameters["num_epochs"] = num_epochs
    input_parameters["save_dir"] = save_dir

    torch.save(input_parameters,f'{save_dir}/input_parameters')
    #------------------------------------------------------------------

    torch.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    np.random.seed(seed_number)
    random.seed(seed_number)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True

    device = torch.device(f"cuda:{str(n_gpu)}" if torch.cuda.is_available() else "cpu")

    # Dataset
    train_dataset = SpeakingFacesDataset(ANNOTATIONS_FILE,DATASET_DIR,'train',
                                        image_transform=T.image_transform, 
                                        audio_transform=T.audio_transform)
    valid_dataset = SpeakingFacesDataset(ANNOTATIONS_FILE,DATASET_DIR,'valid',
                                        image_transform=T.image_transform, 
                                        audio_transform=T.audio_transform)

    # sampler
    train_sampler = ProtoSampler(labels = train_dataset.labels,
                                        n_batch = 10,
                                        n_ways = 3, # n_way
                                        n_support = 1, # n_shots
                                        n_query = 1)
                                        
    val_sampler = ValidSampler(labels = valid_dataset.labels)

    # dataloader
    train_dataloader = DataLoader(dataset=train_dataset, 
                            batch_sampler=train_sampler,
                            num_workers=2, pin_memory=True
                            )

    valid_dataloader = DataLoader(dataset = valid_dataset,
                                batch_sampler=val_sampler,
                                num_workers=4, pin_memory=True
                                )
    # model
    model = ResNet34()
    model = model.to(device)

    # optimizer + scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5, last_epoch=-1)

    # loss
    criterion = PrototypicalLoss(dist_type='cosine_similarity')
    criterion = criterion.to(device)

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
                        save_dir)




