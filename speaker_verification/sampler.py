from torch.utils.data.sampler import Sampler
import numpy as np
import torch
import time

class SFProtoSampler(Sampler):

    def __init__(self, labels, n_batch, n_ways, n_support, n_query):
        self.n_batch = n_batch
        self.n_ways = n_ways
        self.n_shots = n_support
        self.n_query = n_query
        self.n_elmts = n_support + n_query

        unique_labels = np.unique(labels)
        self.indices_per_class = [torch.nonzero(torch.tensor(labels) == label).flatten() for label in unique_labels]

        
    def __iter__(self):
        for _ in range(self.n_batch):
            batch = []

            # Choose random classes
            classes = torch.randperm(len(self.indices_per_class))[:self.n_ways]

            # Choose random elements inside each class 
            batch = [self.indices_per_class[class_k][torch.randperm(len(self.indices_per_class[class_k]))[:self.n_elmts]] for class_k in classes]

            batch = torch.stack(batch).t().reshape(-1).numpy()  # [tensor([ 9, 45]), tensor([ 6, 32])] --> tensor([ 9,  6, 45, 32]) 
            yield batch # на выходе:i - class, j - element's number  [x11, x21, x12, x22]

    def __len__(self): # количество элементов в итераторе
        return self.n_batch
    

class VoxCelebProtoSampler(Sampler):
    def __init__(self, labels, n_batch, n_ways, n_support, n_query):
        
        self.n_ways = n_ways
        self.n_shots = n_support
        self.n_query = n_query
        self.n_elmts = n_support + n_query

        unique_labels = np.unique(labels)
        self.indices_per_class = [torch.nonzero(torch.tensor(labels) == label).flatten() for label in unique_labels]   
        
        self.n_batch = len(unique_labels) // self.n_ways
    
    def __iter__(self):
        
        n_classes = len(self.indices_per_class)
        classes = torch.randperm(n_classes)
        segments = [classes[i:i + self.n_ways] for i in range(0, n_classes, self.n_ways)]
        
        for segment in segments[:self.n_batch]:
            batch = [self.indices_per_class[class_k][torch.randint(len(self.indices_per_class[class_k]), (self.n_elmts,))] for class_k in segment]
            # [tensor([ 9, 45]), tensor([ 6, 32])] --> tensor([ 9,  6, 45, 32])
            batch_indices = torch.stack(batch).t().reshape(-1).numpy()
            yield batch_indices 
                
    def __len__(self): 
        return self.n_batch

class ValidSampler(Sampler):

    def __init__(self, labels):
        
        self.labels = labels
        self.unique_labels = np.unique(labels)
        
    def __iter__(self):

        for label in self.unique_labels:

            same_label_idxs = np.argwhere(self.labels == label).squeeze()
            different_label_idxs = np.argwhere(self.labels != label).squeeze()

            pairs_size = np.min((len(same_label_idxs)-1,len(different_label_idxs)))
            
            if len(different_label_idxs) > pairs_size:
                different_label_idxs = np.random.choice(different_label_idxs, pairs_size, replace=False)
            if len(same_label_idxs) > pairs_size+1:
                same_label_idxs = np.random.choice(same_label_idxs, pairs_size,replace=False)
            
            all_idxs = np.concatenate((same_label_idxs, different_label_idxs), axis=None)

            yield all_idxs

    def __len__(self): # количество элементов в итераторе
        return len(self.unique_labels)