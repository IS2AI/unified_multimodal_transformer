from speaker_verification.dataset import TrainDataset
from speaker_verification.dataset import ValidDataset
from speaker_verification.sampler import ProtoSampler

from speaker_verification.loss import PrototypicalLoss
from speaker_verification.train import train_model
from speaker_verification.parser import createParser
from speaker_verification.transforms import Audio_Transforms
from speaker_verification.transforms import Image_Transforms
from speaker_verification.models import Model
from speaker_verification.train import evaluate_single_epoch
from speaker_verification.metrics import EER_
from speaker_verification.metrics import accuracy_
import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import wandb
import os
from tqdm.auto import tqdm
from tqdm import trange
import pandas as pd 
import torch.nn.functional as F
import copy

def results_to_csv(val_eer, val_acc, data_type, save_dir, exp_name, path_to_valid_list):
    
    df = pd.DataFrame(columns = ["data_type_1", "data_type_2", "EER", "accuracy"])
    data_type = sorted(data_type)
    modalities = copy.copy(data_type)
    
    if (len(data_type) == 3):
        modalities.append("_".join([data_type[0], data_type[1]]))
        modalities.append("_".join([data_type[1], data_type[2]]))
        modalities.append("_".join([data_type[0], data_type[2]]))
        modalities.append("_".join(data_type))
    elif (len(data_type) == 2):
        modalities.append("_".join(data_type))
        
    indices = get_index_pairs(modalities, modalities)
    
    for i,j in indices:
        df.loc[len(df.index)] = [modalities[i], modalities[j], 
                                 val_eer[i,j], val_acc[i,j]]   
    if "valid" in path_to_valid_list:
        file_name = os.path.join(save_dir, exp_name+"_valid_results.csv")
    else:
        file_name = os.path.join(save_dir, exp_name+"_test_results.csv")
    df.to_csv(file_name)
    return df

def get_eer_accuracy(id1, id2, labels):
    cos_sim = F.cosine_similarity(id1, id2, dim=1)
    eer, scores = EER_(cos_sim, labels)
    accuracy = accuracy_(labels, scores)
    return eer, accuracy

def get_index_pairs(l1,l2):
    pairs = []
    for idx1 in range(len(l1)):
        for idx2 in range(len(l1)):
            pair = (idx1, idx2)
            pairs.append(pair)
    return pairs


def evaluate_bimodal_pair(model, id1, id2, labels, device):
    data1_id1,data2_id1, _ = id1
    data1_id2,data2_id2, _ = id2

    data1_id1 = data1_id1.to(device)
    data2_id1 = data2_id1.to(device)

    data1_id2 = data1_id2.to(device)
    data2_id2 = data2_id2.to(device)
    
    with torch.no_grad():
        data1_id1 = model(data1_id1)
        data2_id1 = model(data2_id1)

        data1_id2 = model(data1_id2)
        data2_id2 = model(data2_id2)

        data12_id1 = torch.mean(torch.stack([data1_id1, data2_id1]), dim=0)
        data12_id2 = torch.mean(torch.stack([data1_id2, data2_id2]), dim=0)
        id1 = [data1_id1,data2_id1,data12_id1]
        id2 = [data1_id2,data2_id2,data12_id2]

        indices = get_index_pairs(id1, id2)
        total_eer = np.zeros((len(id1), len(id2)))
        total_accuracy = np.zeros((len(id1), len(id2)))

        for i,j in indices:
            eer, accuracy = get_eer_accuracy(id1[i], id2[j], labels)
            total_eer[i,j] = eer
            total_accuracy[i,j] = accuracy

    return total_eer, total_accuracy

def evaluate_trimodal_pair(model, id1, id2, labels, device):
    data1_id1,data2_id1, data3_id1, _ = id1
    data1_id2,data2_id2, data3_id2, _ = id2

    data1_id1 = data1_id1.to(device)
    data2_id1 = data2_id1.to(device)
    data3_id1 = data3_id1.to(device)

    data1_id2 = data1_id2.to(device)
    data2_id2 = data2_id2.to(device)
    data3_id2 = data3_id2.to(device)

    with torch.no_grad():
        data1_id1 = model(data1_id1)
        data2_id1 = model(data2_id1)
        data3_id1 = model(data3_id1)

        data1_id2 = model(data1_id2)
        data2_id2 = model(data2_id2)
        data3_id2 = model(data3_id2)

        data123_id1 = torch.mean(torch.stack([data1_id1, data2_id1, data3_id1]), dim=0)
        data123_id2 = torch.mean(torch.stack([data1_id2, data2_id2, data3_id2]), dim=0)

        data12_id1 = torch.mean(torch.stack([data1_id1, data2_id1]), dim=0)
        data23_id1 = torch.mean(torch.stack([data2_id1, data3_id1]), dim=0)
        data13_id1 = torch.mean(torch.stack([data1_id1, data3_id1]), dim=0)
        
        data12_id2 = torch.mean(torch.stack([data1_id2, data2_id2]), dim=0)
        data23_id2 = torch.mean(torch.stack([data2_id2, data3_id2]), dim=0)
        data13_id2 = torch.mean(torch.stack([data1_id2, data3_id2]), dim=0)
   
        id1 = [data1_id1, data2_id1, data3_id1, data12_id1, data23_id1, data13_id1, data123_id1]
        id2 = [data1_id2, data2_id2, data3_id2, data12_id2, data23_id2, data13_id2, data123_id2]
        
        indices = get_index_pairs(id1, id2)
        total_eer = np.zeros((len(id1), len(id2)))
        total_accuracy = np.zeros((len(id1), len(id2)))

        for i,j in indices:
            eer, accuracy = get_eer_accuracy(id1[i], id2[j], labels)
            total_eer[i,j] = eer
            total_accuracy[i,j] = accuracy

    return total_eer, total_accuracy

def evaluate_multimodal_model(model,
                        val_dataloader,
                        epoch,
                        device,
                        data_type,
                        loss_type):
    model.eval()
    total_eer = []
    total_accuracy = []

    pbar = tqdm(val_dataloader, desc=f'Eval (epoch = {epoch})')

    for batch in pbar:

        data_type = sorted(data_type)
        id1, id2, labels = batch
        
        if len(data_type) == 2:
            eer, accuracy = evaluate_bimodal_pair(model, id1, id2, labels, device)
        
        elif len(data_type) == 3:
            eer, accuracy = evaluate_trimodal_pair(model, id1, id2, labels, device)

        total_eer.append(eer)
        total_accuracy.append(accuracy)

    return model, total_eer, total_accuracy


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
    exp_name = namespace.exp_name
    batch_size = namespace.batch_size

    params =  torch.load(f'{save_dir}/{exp_name}_input_parameters')
    logs =  torch.load(f'{save_dir}/{exp_name}_logs')

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
    
    data_type = sorted(data_type)
    print(f"Data type {data_type}")
    
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
                                library=library, modality="rgb")
        rgb_T = rgb_T.transform  
        
    if 'thr' in data_type:
        thr_T = Image_Transforms(model_name=model_name,
                                library=library, modality="thr")
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
    PATH=f'{save_dir}/{exp_name}_best_eer.pth'
    model.load_state_dict(torch.load(PATH))
    model = model.to(device)
    print("Loaded weights")
    
  
    epoch = np.argmin(logs['val_eer'])+1
    print(f"at epoch {epoch}")
    model, val_eer, val_acc = evaluate_multimodal_model(model, 
                                  valid_dataloader,
                                  epoch,
                                  device,
                                  data_type,
                                  loss_type)
    
    val_eer = np.array(val_eer).mean(axis=0)
    val_acc = np.array(val_acc).mean(axis=0)
    
    results_to_csv(val_eer, val_acc, data_type, save_dir, exp_name, path_to_valid_list)
    
    




