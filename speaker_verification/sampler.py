from torch.utils.data.sampler import Sampler
import numpy as np
import torch

class SFProtoSampler(Sampler):

    def __init__(self, labels, n_batch, n_ways, n_support, n_query):
        self.n_batch = n_batch
        self.n_ways = n_ways
        self.n_shots = n_support
        self.n_query = n_query
        self.n_elmts = n_support + n_query

        unique_labels = np.unique(labels)
        self.indices_per_class = []
        for i in unique_labels:
            index = np.argwhere(labels == i).reshape(-1)
            index = torch.from_numpy(index)
            self.indices_per_class.append(index)
        
    def __iter__(self):
        for _ in range(self.n_batch):
            batch = []

            # 1.1. choose random classes
            n_classes = len(self.indices_per_class) # number of unique classes
            classes = torch.randperm(n_classes)[:self.n_ways]

            # 1.2. save indices of elements for randomly chosen classes
            for class_k in classes:
              # indexes of elements for class k
                indices_k = self.indices_per_class[class_k]

                # choose random elements inside class k
                n_elements = len(indices_k) 
                pos = torch.randperm(n_elements)[:self.n_elmts]

                # save indices of chosen elements into batch
                batch.append(indices_k[pos])
            
            # from 2d array to 1d array of indices      # class 1: [element 1, element 2], class 2: [element 1, element 2], ...]
            # t() - потому что иначе тяжело будет разделить на support и query
            batch = torch.stack(batch).t().reshape(-1)  # [tensor([ 9, 45]), tensor([ 6, 32])] --> tensor([ 9,  6, 45, 32]) 
            batch = batch.numpy()
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
        self.indices_per_class = []
        for i in unique_labels:
            index = np.argwhere(labels == i).reshape(-1)
            index = torch.from_numpy(index)
            self.indices_per_class.append(index)
        
        self.n_batch = len(unique_labels) // self.n_ways
    
    def __iter__(self):
        
        n_classes = len(self.indices_per_class)  # Number of unique classes
        classes = torch.randperm(n_classes)
        segments = [classes[i:i + self.n_ways] for i in range(0, len(classes), self.n_ways)]
        
        for segment in segments[:self.n_batch]:
            
            batch = []
            for class_k in segment:
                indices_k = self.indices_per_class[class_k]
                sampled_indices = torch.randint(low=0, high=len(indices_k), size=(self.n_elmts,))
                batch.append(indices_k[sampled_indices])

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