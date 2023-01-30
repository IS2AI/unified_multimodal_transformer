from speaker_verification import transforms as T

import os
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
import torch
import torchaudio

# changed
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