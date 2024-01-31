from utils.dataset import TrainDataset
from utils.sampler import SFProtoSampler

from utils.loss import PrototypicalLoss
from utils.train import train_model
from utils.parser import createParser
from utils.models import Model
from utils.train import evaluate_single_epoch
from utils.metrics import EER_
from utils.metrics import accuracy_
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import random
import wandb
import os
from tqdm.auto import tqdm
from tqdm import trange
import pandas as pd 
import torch.nn.functional as F
import copy
from itertools import product
from os import listdir
from os.path import isfile, join
from skimage import io
import torch.nn as nn
import torchaudio
import torchvision.transforms as T


class ValidDataset(Dataset):
    def __init__(self, path_to_valid_dataset, 
                       path_to_valid_list, 
                       data_type,
                       dataset_type,
                       rgb_transform=None, 
                       thr_transform=None, 
                       audio_transform=None,
                       num_eval=1
                ):

        super(ValidDataset, self).__init__()

        self.path_to_valid_dataset = path_to_valid_dataset
        self.data_type = data_type
        self.dataset_type = dataset_type
        self.path_to_valid_list = path_to_valid_list

        with open(self.path_to_valid_list) as f:
            self.pairs_list = f.read().splitlines()
        
        self.rgb_transform = rgb_transform
        self.thr_transform = thr_transform
        self.audio_transform = audio_transform

        self.num_eval = num_eval
        
    def __len__(self):
        return len(self.pairs_list)

    def __getitem__(self, index):

        id1_path, id2_path, label =  self._get_pair(index)

        id1 = self._get_data(id1_path)
        id2 = self._get_data(id2_path)

        return id1, id2, label

    def _get_pair(self, index):
        data_info = self.pairs_list[index].split()

        pairs_label = int(data_info[0]) # same or different: 0 or 1
        id1_path = data_info[1] # here is a path to the wav
        id2_path = data_info[2]

        return id1_path, id2_path, pairs_label

    def _get_data(self, id_path):

        label = self._get_sample_label(id_path)

        path2wav, path2rgb, path2thr = self._get_sample_path(id_path)

        data = {}
        
        if "wav" in self.data_type:
            data["wav"], sample_rate = torchaudio.load(path2wav)
            if self.audio_transform:
                data["wav"] = self.audio_transform(data["wav"], sample_rate)

        if "rgb" in self.data_type:
            if self.num_eval > 1:
                data["rgb"] = [io.imread(path2rgb_item) for path2rgb_item in path2rgb]
            else:
                data["rgb"] = io.imread(path2rgb)
            
            if self.rgb_transform:
                data["rgb"] = self.rgb_transform(data["rgb"])
            
        if "thr" in self.data_type:
            data["thr"] = io.imread(path2thr)
            if self.thr_transform:
                data["thr"] = self.thr_transform(data["thr"])
        data = dict(sorted(data.items()))
        sample = (*list(data.values()), label)
        
        return sample
    
    def _get_sample_path(self, id_path):
        
        if self.dataset_type == "SF":
            path = "/".join(self.path_to_valid_dataset.split("/")[:-1]) # path to sf_pv

            path2wav = f"{path}/{id_path}"
            path2rgb = f"{path}/" + "/".join(id_path.split("/")[:-2]) + "/" + "rgb" + "/" + id_path.split("/")[-1].split(".")[0] + "/1.jpg"
            path2thr = f"{path}/" + "/".join(id_path.split("/")[:-2]) + "/" + "thr" + "/" + id_path.split("/")[-1].split(".")[0] + "/1.jpg"
        elif self.dataset_type == "VX2":
            path = self.path_to_valid_dataset

            path2wav = f"{path}/{id_path}"
            path2rgb = f"{path}/" + "/".join(id_path.split("/")[:-2]) + "/" + "rgb" + "/"  + str(int(id_path.split("/")[-1].split(".")[0])) 
            onlyfiles = [f for f in listdir(path2rgb) if isfile(join(path2rgb, f)) and 'jpg' in f]
            onlyfiles.sort()
            if self.num_eval > 1:
                path2rgb = [path2rgb + "/" + onlyfile for onlyfile in onlyfiles]
                path2rgb = [path2rgb[i] for i in np.linspace(0, len(path2rgb) - 1, endpoint=True, num=self.num_eval, dtype=int).tolist()]
                #path2rgb = [path2rgb[i] for i in [0] * self.num_eval]
                
            else:     
                path2rgb = path2rgb + "/" + onlyfiles[0]

            path2thr = f"{path}/" + "/".join(id_path.split("/")[:-2]) + "/" + "thr" + "/" + str(int(id_path.split("/")[-1].split(".")[0]))
            onlyfiles = [f for f in listdir(path2thr) if isfile(join(path2thr, f)) and 'jpg' in f]
            path2thr = path2thr + "/" + onlyfiles[0]

        return path2wav, path2rgb, path2thr

    def _get_sample_label(self, id_path):
        
        if self.dataset_type == "SF":
            label = int(id_path.split("/")[2].split("_")[-1])
        elif self.dataset_type == "VX2":
            label = int(id_path.split('/')[0][2:])

        return label
    
class Image_Transforms:
    def __init__(self, 
                 library,
                 model_name, modality, dataset_type, num_eval):

        self.library = library
        self.model_name = model_name
        self.modality = modality
        self.dataset_type = dataset_type
        self.num_eval = num_eval
        
        if self.library == "huggingface":
            pass
        elif self.library == "timm":
            self.timm_init()
        elif self.library == "pytorch":
            self.pytorch_init()

    def timm_init(self):
        if self.model_name == "vit_base_patch16_224":
            # n_channels = 3
            if self.modality == "rgb":
                if self.dataset_type == "SF":
                    mean_val = [0.35843268, 0.27319421, 0.23963803]
                    std_val = [0.15948673, 0.13425587, 0.1222331]
                else:
                    # VoxCeleb
                    mean_val = [0.55803093, 0.38461271, 0.34251855]
                    std_val = [0.19474319, 0.15522242, 0.15067575]
            
            elif self.modality == "thr":
                mean_val = [0.94060585, 0.74481036, 0.29649508]
                std_val = [0.15892989, 0.27409379, 0.26099585]
            
            self.transform_image=T.Compose([
                T.ToPILImage(),
                T.Resize(size=(224,224)),
                T.ToTensor(), 
                T.Normalize(mean=torch.tensor(mean_val), std=torch.tensor(std_val))  
            ])
        else:
            self.transform_image = T.Compose([
                T.ToPILImage(),
                T.Resize((128,128)),
                T.ToTensor(), 
            ])
        
    def pytorch_init(self):
        self.transform_image = T.Compose([
                T.ToPILImage(),
                T.Resize((128,128)),
                T.ToTensor(), 
            ])

    # MAIN TRANSFORM FUNCTION
    def transform(self, image):
            if self.num_eval > 1  :
                return [self.transform_image(image_item) for image_item in image]    
            else:
                return self.transform_image(image)
            


class Audio_Transforms:
    def __init__(self, 
                sample_rate,
                sample_duration, # seconds
                n_fft, # from Korean code
                win_length,
                hop_length,
                window_fn,
                n_mels,
                model_name, 
                library, 
                mode
                ):

        self.sample_rate = sample_rate
        self.sample_duration = sample_duration
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.window_fn = window_fn
        self.n_mels = n_mels
        self.model_name = model_name
        self.library = library
        self.mode = mode

        self.timm_init()
        
    def timm_init(self):
       
        self.to_MelSpectrogram =  torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                window_fn=self.window_fn,
                n_mels=self.n_mels
            )
        self.instance_norm = nn.InstanceNorm1d(self.n_mels)

        if self.model_name == "vit_base_patch16_224":
            # n_channels = 1
            self.vit_transform=T.Compose([
                T.Resize(size=(224,224))
            ])


    # MAIN TRANSFORM FUNCTION
    def transform(self, signal, sample_rate):
        signal = self.test_transform(signal, sample_rate)
        inputs = self.timm_transform(signal)
        return inputs
    
    def test_transform(self, signal, sample_rate):

        # stereo --> mono
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        
        # sample_rate --> 16000
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            signal = resampler(signal)

        # normalize duration --> 3 seconds (mean duration in dataset)
        segment_length = int(self.sample_duration * sample_rate) # sample length of the audio signal
        signal_length = signal.shape[1]
        segments = []
        if signal_length < segment_length:

            num_missing_points = int(segment_length - signal_length)
            left_padding = (num_missing_points + 1) // 2
            right_padding = num_missing_points - left_padding
            
            segment = torch.zeros((signal.shape[0], segment_length))
            segment[:, left_padding:-right_padding] = signal
        #start_time = time.time()    
        start_frames = np.linspace(0, signal_length - segment_length, num=10, dtype=int)
        segments = [signal[:, start_frame:start_frame + segment_length] for start_frame in start_frames]
        # end_time = time.time()
        #print("Time: {}".format(end_time - start_time))
        return segments


    
    def timm_transform(self, audios):
        inputs = []
        for audio in audios:
            input = self.to_MelSpectrogram(audio)
            input = input+1e-6
            input = input.log()
            input = self.instance_norm(input)
            if self.model_name == "vit_base_patch16_224":
                input = input.repeat(3, 1, 1)
                input = self.vit_transform(input)
            inputs.append(input)
        #inputs = torch.stack(inputs)
        return inputs

def results_to_csv(val_eer, val_acc, data_type, save_dir, exp_name, path_to_valid_list, dataset_type):
    
    df = pd.DataFrame(columns = ["data_type_1", "data_type_2", "EER", "accuracy"])
    data_type = sorted(data_type)
    modalities = copy.copy(data_type)
    
    if (len(data_type) == 3):
        modalities.append("_".join([data_type[0], data_type[1]]))
        modalities.append("_".join([data_type[0], data_type[2]]))
        modalities.append("_".join([data_type[1], data_type[2]]))
        modalities.append("_".join(data_type))
    elif (len(data_type) == 2):
        modalities.append("_".join(data_type))
        
    indices = get_index_pairs(modalities, modalities)
    
    for i,j in indices:
        df.loc[len(df.index)] = [modalities[i], modalities[j], 
                                 val_eer[i,j], val_acc[i,j]]   
    if "valid" in path_to_valid_list:
        file_name = os.path.join(save_dir, exp_name+"{}_valid_results.csv".format(dataset_type))
    else:
        file_name = os.path.join(save_dir, exp_name+"{}_test_results.csv".format(dataset_type))
    df.to_csv(file_name)
    return df


def get_eer_accuracy(id1, id2, labels):
    scores = [F.cosine_similarity(id1_item, id2_item, dim=-1) for id1_item in id1 for id2_item in id2]
    scores = torch.stack(scores)
    cos_sim = scores.mean(dim=0)

    eer, scores = EER_(cos_sim, labels)
    accuracy = accuracy_(labels, scores)
    return eer, accuracy

def get_index_pairs(l1, l2):
    return list(product(range(len(l1)), range(len(l2))))


def preprocess_and_infer(data_list, model, device):
    
    processed_data_list = [[model(data)for data in item_list] for item_list in data_list ]
    return processed_data_list

def calculate_mean_combinations(data_list):
    mean_combinations = []
    for i in range(len(data_list)):
        for j in range(i + 1, len(data_list)):
            mean_combinations.append(torch.mean(torch.stack([data_list[i], data_list[j]]), dim=0))
    return mean_combinations

def calculate_total_eer_accuracy(id1, id2, labels):
    total_eer = np.zeros((len(id1), len(id2)))
    total_accuracy = np.zeros((len(id1), len(id2)))

    for i, j in get_index_pairs(id1, id2):
        eer, accuracy = get_eer_accuracy(id1[i], id2[j], labels)
        total_eer[i, j] = eer
        total_accuracy[i, j] = accuracy

    return total_eer, total_accuracy

def evaluate_trimodal_pair(model, id1, id2, labels, device):
    id1_data = [item.to(device) for item in id1[:-1]]
    id2_data = [item.to(device) for item in id2[:-1]]
    
    with torch.no_grad():
        processed_id1_data = preprocess_and_infer(id1_data, model, device)
        processed_id2_data = preprocess_and_infer(id2_data, model, device)

        mean_combinations_id1 = calculate_mean_combinations(processed_id1_data)
        mean_combinations_id2 = calculate_mean_combinations(processed_id2_data)

        mean_all_id1 = [torch.mean(torch.stack(processed_id1_data), dim=0)]
        mean_all_id2 = [torch.mean(torch.stack(processed_id2_data), dim=0)]

        id1_combined = processed_id1_data + mean_combinations_id1 + mean_all_id1
        id2_combined = processed_id2_data + mean_combinations_id2 + mean_all_id2

        total_eer, total_accuracy = calculate_total_eer_accuracy(id1_combined, id2_combined, labels)
    
    return total_eer, total_accuracy


def evaluate_bimodal_pair(model, id1, id2, labels, device):
    
    id1_data = [[item.to(device) for item in item_list] for item_list in id1[:-1]]
    id2_data = [[item.to(device) for item in item_list] for item_list in id2[:-1]]
    
    with torch.no_grad():
        processed_id1_data = preprocess_and_infer(id1_data, model, device)
        processed_id2_data = preprocess_and_infer(id2_data, model, device)
       
        mean_all_id1 = [[torch.mean(torch.stack([processed_id1_data[0][i], processed_id1_data[1][i]]), dim=0) for i in range(len(processed_id1_data[0]))]] 
        mean_all_id2 = [[torch.mean(torch.stack([processed_id2_data[0][i], processed_id2_data[1][i]]), dim=0) for i in range(len(processed_id2_data[0]))]] 
       
        id1_combined = processed_id1_data + mean_all_id1
        id2_combined = processed_id2_data + mean_all_id2
       
        total_eer, total_accuracy = calculate_total_eer_accuracy(id1_combined, id2_combined, labels)

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
    data_type = namespace.data_type
    dataset_type = namespace.dataset_type
    
    params =  torch.load(f'{save_dir}/{exp_name}_input_parameters')
    logs =  torch.load(f'{save_dir}/{exp_name}_logs')

    # dataset
    annotation_file = params['annotation_file']
    path_to_train_dataset = params['path_to_train_dataset']
    
    
    
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
                                    library=library,
                                    mode="test")
        audio_T = audio_T.transform

    if 'rgb' in data_type:
        rgb_T = Image_Transforms(model_name=model_name,
                                library=library, modality="rgb", dataset_type=dataset_type, num_eval=10)
        rgb_T = rgb_T.transform  
        
    if 'thr' in data_type:
        thr_T = Image_Transforms(model_name=model_name,
                                library=library, modality="thr", dataset_type=dataset_type)
        thr_T = thr_T.transform  

    # Dataset
    valid_dataset= ValidDataset(path_to_valid_dataset=path_to_valid_dataset, 
                                path_to_valid_list=path_to_valid_list, 
                                data_type=data_type,
                                dataset_type=dataset_type,
                                rgb_transform=rgb_T, 
                                thr_transform=thr_T, 
                                audio_transform=audio_T,
                                num_eval=10)
    
    
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
    
    results_to_csv(val_eer, val_acc, data_type, save_dir, exp_name, path_to_valid_list, dataset_type)
    
    




