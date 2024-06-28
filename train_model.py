from utils.dataset import TrainDataset
from utils.dataset import ValidDataset
from utils.sampler import SFProtoSampler
from utils.sampler import VoxCelebProtoSampler

from utils.loss import PrototypicalLoss
from utils.train import train_model
from utils.parser import createParser
from utils.transforms import Audio_Transforms
from utils.transforms import Image_Transforms
from utils.models import Model

import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import wandb
import os
import yaml

if __name__== "__main__":

    parser = createParser()
    namespace = parser.parse_args()

    # Input parameters
    n_gpu = namespace.n_gpu
    seed_number = namespace.seed
    print("SEED {}".format(seed_number))
    
    # dataset
    annotations_file = namespace.annotation_file
    path_to_train_dataset = namespace.path_to_train_dataset
    path_to_valid_dataset = namespace.path_to_valid_dataset
    path_to_valid_list = namespace.path_to_valid_list
    dataset_type = namespace.dataset_type
    exp_name = namespace.exp_name
    save_dir = namespace.save_dir
    config_file = namespace.config_file

    
    with open(config_file) as cf_file:
        config = yaml.safe_load( cf_file.read())
    
    # model
    library = config['library']
    model_name = config['model_name']
    fine_tune = config['fine_tune']
    transfer = config['transfer']
    pool=config['pool']
    
    # transfer_exp_path = config['transfer_exp_path'] # TODO do we need it?
    data_type=config['data_type']
    pretrained_weights = config['pretrained_weights']
    embedding_size = config['embedding_size']

    # audio transforms
    sample_rate = config['sample_rate']
    sample_duration=config['sample_duration']
    n_fft=config['n_fft']
    win_length=config['win_length']
    hop_length=config['hop_length']
    n_mels=config['n_mels']

    # image transform 
    image_resize = config['image_resize']

    # train_dataloader
    n_batch = config['n_batch']
    n_ways = config['n_ways']
    n_support = config['n_support']
    n_query = config['n_query']
    name_sampler = config['sampler']

    #valid_dataloader
    batch_size = config['batch_size']

    # loss
    dist_type = config['dist_type']
    loss_type = config['loss_type']

    # optimizer
    weight_decay = config['weight_decay']
    learning_rate = float(config['lr'])
    scheduler_type = config['scheduler']
    gamma = config['gamma']
    t_max = config['t_max']
    step_size = config['step_size']
    
    
    # train 
    num_epochs = config['num_epochs']
    # save_dir = config['save_dir']
    wandb_use = config['wandb']
    # mode = 'train' # TODO pass this argument in the evaluate and stuff

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
    input_parameters["loss_type"] = loss_type
    input_parameters["library"] = library
    input_parameters["sampler"] = name_sampler

    input_parameters["model_name"] = model_name
    input_parameters["fine_tune"] = fine_tune
    input_parameters["transfer"] = transfer
    input_parameters["pool"] = pool # TODO do we need it?
    input_parameters["exp_name"] = exp_name
    input_parameters["pretrained_weights"] = pretrained_weights
    input_parameters["embedding_size"] = embedding_size
    input_parameters["batch_size"] = batch_size
    input_parameters["wandb"] = wandb_use # TODO to delete?
    
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
    input_parameters["lr"] = learning_rate
    
    # print("SAVE DIR", save_dir)
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

    audio_Train = None
    audio_Test = None
    rgb_T = None
    thr_T = None
    
    # Transforms for each modality
    if 'wav' in data_type:
        audio_Train = Audio_Transforms(sample_rate=sample_rate,
                                    sample_duration=sample_duration,
                                    n_fft=n_fft, 
                                    win_length=win_length,
                                    hop_length=hop_length,
                                    window_fn=torch.hamming_window,
                                    n_mels=n_mels,
                                    model_name=model_name,
                                    library=library,
                                    mode="train",
                                    dataset_type=dataset_type)
        audio_Train = audio_Train.transform
        audio_Test = Audio_Transforms(sample_rate=sample_rate,
                                    sample_duration=sample_duration,
                                    n_fft=n_fft, 
                                    win_length=win_length,
                                    hop_length=hop_length,
                                    window_fn=torch.hamming_window,
                                    n_mels=n_mels,
                                    model_name=model_name,
                                    library=library,
                                    mode="test",
                                    dataset_type=dataset_type)
        audio_Test = audio_Test.transform

    if 'rgb' in data_type:
        rgb_T = Image_Transforms(model_name=model_name,
                                library=library, modality="rgb", dataset_type=dataset_type)
        rgb_T = rgb_T.transform  
        
    if 'thr' in data_type:
        thr_T = Image_Transforms(model_name=model_name,
                                library=library, modality="thr", dataset_type=dataset_type)
        thr_T = thr_T.transform   

    # Dataset
    train_dataset = TrainDataset(annotations_file=annotations_file, 
                                path_to_train_dataset=path_to_train_dataset, 
                                data_type=data_type, 
                                dataset_type=dataset_type,
                                train_type = 'train',
                                rgb_transform=rgb_T, 
                                thr_transform=thr_T,
                                audio_transform=audio_Train)

    valid_dataset= ValidDataset(path_to_valid_dataset=path_to_valid_dataset, 
                                path_to_valid_list=path_to_valid_list, 
                                data_type=data_type,
                                dataset_type=dataset_type,
                                rgb_transform=rgb_T, 
                                thr_transform=thr_T,
                                audio_transform=audio_Test)
    
    if loss_type == 'classification':
        train_dataloader = DataLoader(dataset=train_dataset,
                        shuffle=True,
                        batch_size=batch_size,
                        num_workers=4)
        train_sampler = None

    elif loss_type == 'metric_learning':
        if name_sampler =="SFProtoSampler":
            train_sampler = SFProtoSampler(train_dataset.labels,
                                        n_batch,
                                        n_ways, # n_way
                                        n_support, # n_shots
                                        n_query)
      
        elif name_sampler =="VoxCelebProtoSampler":
            train_sampler = VoxCelebProtoSampler(train_dataset.labels,
                                        n_ways, # n_way
                                        n_support, # n_shots
                                        n_query)

        train_dataloader = DataLoader(dataset=train_dataset, 
                                batch_sampler=train_sampler)
    valid_dataloader = DataLoader(dataset=valid_dataset,
                            batch_size=batch_size)
    pretrained_model = Model(library=library, 
            pretrained_weights=pretrained_weights, 
            fine_tune=fine_tune, 
            embedding_size=embedding_size,
            model_name = model_name,
            pool=pool,
            data_type=data_type)
    
    
    if loss_type == 'classification':
        n_classes = len(np.unique(train_dataset.labels))
        classification_layer = torch.nn.Linear(embedding_size, n_classes)
        model = torch.nn.Sequential()
        model.add_module('pretrained_model', pretrained_model)
        model.add_module('classification_layer', classification_layer)
    elif loss_type == 'metric_learning':
        model = pretrained_model
    
    if transfer:
        PATH=f'{transfer_exp_path}_best_eer.pth'
        print("Loading model saved as {}".format(PATH))
        model.load_state_dict(torch.load(PATH))
        
    model = model.to(device)

    # loss
    criterion = PrototypicalLoss(dist_type = dist_type)
    criterion = criterion.to(device)

    # optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
    if scheduler_type == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = gamma)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = t_max)
        
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
                    loss_type=loss_type,
                    wandb=wandb)
