from utils.dataset import TrainDataset
from utils.dataset import ValidDataset
from utils.sampler import SFProtoSampler

from utils.loss import PrototypicalLoss
from utils.train import train_model
from utils.parser import createParser
from utils.transforms import Audio_Transforms
from utils.transforms import Image_Transforms
from utils.models import Model
from utils.train import evaluate_single_epoch
import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import wandb
import os


if __name__== "__main__":
    
    parser = createParser()
    namespace = parser.parse_args()
    
    # Device
    n_gpu = namespace.n_gpu
    seed_number = namespace.seed
    print("SEED {}".format(seed_number))
    
    # Validation data
    path_to_valid_dataset = namespace.path_to_valid_dataset
    path_to_valid_list = namespace.path_to_valid_list
    
    # Save
    save_dir = namespace.save_dir
    data_type = namespace.data_type # ['wav', 'rgb']
    exp_name = namespace.exp_name
    batch_size = namespace.batch_size
    print(batch_size)
    # Loaded parameters
    params =  torch.load(f'{save_dir}/{"_".join(data_type)}_{exp_name}_input_parameters')
    
    # dataset
    annotation_file = params['annotation_file']
    path_to_train_dataset = params['path_to_train_dataset']
    data_type = params['data_type']
    dataset_type = params['dataset_type']
    
    # model
    library = params['library']
    model_name = params['model_name']
    pretrained_weights=params['pretrained_weights']
    fine_tune=params['fine_tune']
    embedding_size=params['embedding_size']
    #pool = ?
    
    
    pool = params['pool']

    # sampler
    n_batch=params['n_batch']
    n_ways=params['n_ways']
    n_support=params['n_support']
    n_query=params['n_query']

    # loss
    dist_type=params['dist_type']
    loss_type=params['loss_type']
    
    # audio transform params
    sample_rate= params['sample_rate']
    sample_duration=params['sample_duration'] # seconds
    n_fft=params['n_fft'] # from Korean code
    win_length=params['win_length']
    hop_length=params['hop_length']
    window_fn=torch.hamming_window
    n_mels=params['n_mels']

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
    print(f"GPU {n_gpu}")
    audio_T = None
    rgb_T = None
    thr_T = None


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

    if 'rgb' in data_type:
        rgb_T = Image_Transforms(model_name=model_name,
                                library=library, modality="rgb", dataset_type=dataset_type,)
        rgb_T = rgb_T.transform  
        
    if 'thr' in data_type:
        thr_T = Image_Transforms(model_name=model_name,
                                library=library, modality="thr", dataset_type=dataset_type,)
        thr_T = thr_T.transform  

    # Dataset
    valid_dataset= ValidDataset(path_to_valid_dataset=path_to_valid_dataset, 
                                path_to_valid_list=path_to_valid_list, 
                                data_type=data_type,
                                dataset_type=dataset_type,
                                rgb_transform=rgb_T, 
                                thr_transform=thr_T, 
                                audio_transform=audio_T)
    
    
    valid_dataloader = DataLoader(dataset=valid_dataset,
                            batch_size=batch_size)
    
    # Build model object
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
    
    # Load model weights    
    PATH=f'{save_dir}/{"_".join(data_type)}_{exp_name}_best_eer.pth'
    model.load_state_dict(torch.load(PATH))
    model = model.to(device)
    print("Loaded weights")
    
    # Test
    logs = torch.load(f'{save_dir}/{"_".join(data_type)}_{exp_name}_logs')
    
    epoch = np.argmin(logs['val_eer'])+1
    print(f"at epoch {epoch}")
    model, val_eer, val_acc = evaluate_single_epoch(model, 
                                  valid_dataloader,
                                  epoch,
                                  device,
                                  data_type,
                                  loss_type)
    
    logs['best_test_eer'] = val_eer
    logs['best_test_acc'] = val_acc
    torch.save(logs,f'{save_dir}/{"_".join(data_type)}_{exp_name}_logs')
    




