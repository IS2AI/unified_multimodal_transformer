from speaker_verification import transforms as T

import os
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
import torch
import torchaudio

# where we load id by annotation file 
class SpeakingFacesDataset(Dataset):
    def __init__(self, annotations_file, 
                       dataset_dir, 
                       train_type, 
                       image_transform=None, 
                       audio_transform=None
                ):

        super(SpeakingFacesDataset, self).__init__()

        self.dataset_dir = f'{dataset_dir}'
        self.train_type = train_type
        self.df_all = pd.read_csv(annotations_file)
        self.df_all = self.df_all[self.df_all['train_type'] == self.train_type]
        self.df_wav = self.df_all[self.df_all['data_type'] == 'wav']
        self.labels = self.df_wav["person_id"].values

        self.image_transform = image_transform
        self.audio_transform = audio_transform


    def __len__(self):
        return len(self.df_wav)

    def __getitem__(self, index):

        sample = self._get_data(index)
        return sample

    def _get_data(self, index):

        label = self._get_sample_label(index)

        path2wav, path2rgb, path2thr = self._get_sample_path(index)

        data_wav, sample_rate = torchaudio.load(path2wav) 
        data_rgb = io.imread(path2rgb)
        data_thr = io.imread(path2thr)

        if self.audio_transform:
            data_wav = self.audio_transform(data_wav, sample_rate)
        if self.image_transform:
            data_rgb = self.image_transform(data_rgb)
            data_thr = self.image_transform(data_thr)

        return (data_wav, data_rgb, data_thr, label)
    
    def _get_sample_path(self, index):

        utterance = self.df_wav.iloc[index,1]
        session_id = self.df_wav.iloc[index,3]
        person_id = self.df_wav.iloc[index,4]

        df_sample = self.df_all[self.df_all.person_id == person_id]
        df_sample = df_sample[df_sample.utterance == utterance]
        df_sample = df_sample[df_sample.session_id == session_id]
        
        path2wav = os.path.join(self.dataset_dir,f'sub_{person_id}',
                            f'{session_id}', 'wav', 
                            df_sample.iloc[0, 0])

        path2rgb = os.path.join(self.dataset_dir,f'sub_{person_id}',
                            f'{session_id}', 'rgb', f'{utterance}',
                            df_sample.iloc[1, 0])

        path2thr = os.path.join(self.dataset_dir,f'sub_{person_id}',
                                    f'{session_id}', 'thr', f'{utterance}',
                                    df_sample.iloc[2, 0])

        return path2wav, path2rgb, path2thr

    def _get_sample_label(self, index):
        return torch.tensor(self.df_wav.iloc[index, 4])


# validation set created from valid_lists_v2.txt
class ValidDataset(Dataset):
    def __init__(self, path2dataset, # path to sf_pv
                       train_type, 
                       image_transform=None, 
                       audio_transform=None
                ):

        super(ValidDataset, self).__init__()

        self.path2dataset = path2dataset

        path2list = f'{path2dataset}/metadata/{train_type}_list_v2.txt'

        with open(path2list) as f:
            self.pairs_list = f.read().splitlines()
        
        self.image_transform = image_transform
        self.audio_transform = audio_transform

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

    def _get_data(self, id):

        label = self._get_label(id)
        path2wav, path2rgb, path2thr = self._get_sample_path(id)

        data_wav, sample_rate = torchaudio.load(path2wav) 
        data_rgb = io.imread(path2rgb)
        data_thr = io.imread(path2thr)

        if self.audio_transform:
            data_wav = self.audio_transform(data_wav, sample_rate)
        if self.image_transform:
            data_rgb = self.image_transform(data_rgb)
            data_thr = self.image_transform(data_thr)

        return (data_wav, data_rgb, data_thr, label)
    
    def _get_sample_path(self, id):

        path = "/".join(self.path2dataset.split("/")[:-1]) # path to sf_pv

        path2wav = f"{path}/{id}"
        path2rgb = f"{path}/" + "/".join(id.split("/")[:-2]) + "/" + "rgb" + "/" + id.split("/")[-1].split(".")[0] + "/1.jpg"
        path2thr = f"{path}/" + "/".join(id.split("/")[:-2]) + "/" + "thr" + "/" + id.split("/")[-1].split(".")[0] + "/1.jpg"

        return path2wav, path2rgb, path2thr

    def _get_label(self, id):
        
        label = int(id.split("/")[2].split("_")[-1])

        return label