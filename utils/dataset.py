import os
import pandas as pd
from skimage import io
from torch.utils.data import Dataset
import torch
import torchaudio
import numpy as np
from os import listdir
from os.path import isfile, join
import time
from sklearn.preprocessing import LabelEncoder


# Universal Dataset with VoxCeleb2, Speaking Faces or joint format 
class TrainDataset(Dataset):
    def __init__(self, annotations_file, 
                       path_to_train_dataset, 
                       data_type, 
                       dataset_type, # VX2 or SF
                       train_type,
                       rgb_transform=None,
                       thr_transform=None,
                       audio_transform=None):

        """ 
        Input parameters:
            annotations_file: annotation file joined VX2 + SF.
            path_to_train_dataset: path to the directory where all datasets are located.
            dataset_type: can be either "SF", "VX2".
            train_type: "train"
            data_type: list of data types, ex: ['wav', 'rgb', 'thr']
            rgb_transform: image transform using the mean and std of rgb images
            thr_transform: image transform using the mean and std of thr images
            audio_transform: audio transform
        """

        self.path_to_train_dataset = f'{path_to_train_dataset}'
        self.train_type = train_type
        self.data_type = data_type
        self.dataset_type = dataset_type

        self.df_all = pd.read_csv(annotations_file)
        
        self.df_all.reset_index(inplace=True)
        self.df_all = self.df_all.drop(["index"], axis=1)

        self.labels = self.df_all["new_ids"].values

        self.encoder = LabelEncoder()
        self.encoder.fit(np.unique(self.labels))
        self.labels = self.encoder.transform(self.labels)

        self.rgb_transform = rgb_transform
        self.thr_transform = thr_transform

        self.audio_transform = audio_transform

    def __len__(self):
        return len(self.df_wav)

    def __getitem__(self, index):
        sample = self._get_data(index)
        return sample

    def _get_data(self, index):
        
        label = self._get_sample_label(index)
        
        path2wav, path2rgb, path2thr = self._get_sample_path(index)
        data = {}
        
        if "wav" in self.data_type:
            if path2wav:
                data["wav"], sample_rate = torchaudio.load(path2wav)
            if self.audio_transform:
                data["wav"] = self.audio_transform(data["wav"], sample_rate)
        if "rgb" in self.data_type:
            if path2rgb:
                data["rgb"] = io.imread(path2rgb)
            if self.rgb_transform:
                data["rgb"] = self.rgb_transform(data["rgb"])
            
        if "thr" in self.data_type:
            if path2thr:
                data["thr"] = io.imread(path2thr)
            if self.thr_transform:
                data["thr"] = self.thr_transform(data["thr"])

        data = dict(sorted(data.items()))
        sample = (*list(data.values()), label)
        
        return sample

    def _get_sample_path(self, index):
        
        person_id = self.df_all.loc[index, "person_id"]
        session_id = self.df_all.loc[index, "session_id"]
        utterance = int(self.df_all.loc[index, "utterance"])
        wav_source = self.df_all.loc[index, "wav"]
        rgb_source = self.df_all.loc[index, "rgb"]    

        if self.dataset_type == "VX2":
            person_id_real = f'id{person_id:05d}'
           
        elif self.dataset_type == "SF":
            person_id_real = f'sub_{person_id}'
            thr_source = self.df_all.loc[index, "thr"]
        
        path2wav = path2rgb = path2thr = None
        for data_type in self.data_type:
            if data_type == 'wav':
                path2wav = os.path.join(self.path_to_train_dataset, person_id_real,
                                            str(session_id), 'wav', wav_source)
            elif data_type == 'rgb':
                path2rgb = os.path.join(self.path_to_train_dataset, person_id_real,
                                            f'{session_id}', 'rgb', f'{utterance}',rgb_source)
            elif data_type == 'thr':
                path2thr = os.path.join(self.path_to_train_dataset, person_id_real,
                                            f'{session_id}', 'thr', f'{utterance}',thr_source)

        return path2wav, path2rgb, path2thr
    def _get_sample_label(self, index):
        # return torch.tensor(self.df_wav.loc[index,"new_ids"])
        return torch.tensor(self.labels[index])


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


# load id by annotation file: Just Speaking Faces
class SpeakingFacesDataset(Dataset):
    def __init__(self, annotations_file, 
                       dataset_dir, 
                       train_type, 
                       data_type, # ['wav','rgb','thr']
                       rgb_transform=None,
                       thr_transform=None, 
                       audio_transform=None
                ):

        super(SpeakingFacesDataset, self).__init__()

        self.dataset_dir = f'{dataset_dir}'
        self.train_type = train_type
        self.data_type = data_type
        self.df_all = pd.read_csv(annotations_file)
        self.df_all = self.df_all[self.df_all['train_type'] == self.train_type]
        self.df_wav = self.df_all[self.df_all['data_type'] == 'wav']
        self.labels = self.df_wav["person_id"].values

        self.rgb_transform = rgb_transform
        self.thr_transform = thr_transform
        self.audio_transform = audio_transform


    def __len__(self):
        return len(self.df_wav)

    def __getitem__(self, index):

        sample = self._get_data(index)
        return sample

    def _get_data(self, index):

        label = self._get_sample_label(index)

        path2wav, path2rgb, path2thr = self._get_sample_path(index)

        data = {}
        
        if "wav" in self.data_type:
            data["wav"], sample_rate = torchaudio.load(path2wav)
            if self.audio_transform:
                data["wav"] = self.audio_transform(data["wav"], sample_rate)

        if "rgb" in self.data_type:
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