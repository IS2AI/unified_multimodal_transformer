import torch
import torch.nn.functional as F
import torch.nn as nn

# Loss
class PrototypicalLoss(nn.Module):
    """
        Parameters
        ----------
        dist_type : str, default = "squared_euclidean"
            Distance type to calculate. 
            It can be either "squared_euclidean" or "cosine_similarity".
    """
    def __init__(self, dist_type="squared_euclidean"):
        super(PrototypicalLoss, self).__init__()
        self.dist_type = dist_type
    
    def forward(self, data, label, n_ways, n_shots, n_query):

        # 1. Divide data on support and query
        p = n_shots * n_ways
        support, query = data[:p], data[p:]
        # 1.1. make shape [n_shots, n_ways, emb_size]
        support = support.reshape(n_shots, n_ways, -1)

        # 2. Compute prototype from support examples for each class
        prototype = self.calculate_prototype(support)

        # 3. Compute euclidean distances between query samples and prototypes
        dist = self.calculate_dist(query, prototype, dist_type=self.dist_type)

        # 4. Calculate CrossEntopyLoss(-d(q,ck))
        logits = -dist
        loss = F.cross_entropy(logits, label)

        return loss, logits
    
    def calculate_prototype(self, support):
        '''
        Calculates prototypes (aka centroids) for each sample.

        Parameters
        ----------

        support: array with shape [n_shots, n_ways, emb_size]
                Support vector from which we calculate prototypes.
                n_shots - number of support vectors in each class
                        i.e. number of elements in each sample
                n_ways - number of classes
                emb_size - embedding size
        
        Returns
        -------
        prototypes: array with shape [n_ways, emb_size]
                Prototypes(centroids) for each class.
        '''
        return support.mean(dim=0)

    def calculate_dist(self, query, prototype, dist_type):
        '''
        Calculate squared euclidean distance between each sample and prototype 
        from each class.
        i.e.
        Q = [q11, # shape: n_query*n_ways, emb_size qij - i - class, j - element
            q21,
            q31, 
            q12,
            q22,
            q32]

        Parameters
        ----------
        query: 2d tensor with shape [n_query*n_ways, emb_size]
        prototype: 2d tensor with shape [n_ways, emb_size]
        dist_type: distance type to calculate. 
                   It can be either "squared_euclidean" or "cosine_similarity".
        
        Return
        -----------
        distances: 2d tensor with shape torch.Size([n_query*n_ways, n_ways])
            d(q11, c1), d(q11, c2), d(q11, c3)
            d(q21, c1), d(q21, c2), d(q21, c3)
            d(q31, c1), d(q31, c2), d(q31, c3)

            d(q12, c1), d(q12, c2), d(q11, c3)
            d(q22, c1), d(q22, c2), d(q21, c3)
            d(q32, c1), d(q32, c2), d(q32, c3)

        '''
        # x: N x D
        # y: M x D
        n = query.size(0)
        m = prototype.size(0)
        d = query.size(1)
        # assert d == prototype.size(1)

        # with shape [n_query*n_ways, n_ways, emb_size]
        query = query.unsqueeze(1).expand(n, m, d) 
        prototype = prototype.unsqueeze(0).expand(n, m, d) 

        if dist_type == "squared_euclidean":
            distance = torch.pow(query - prototype, 2).sum(2) 
            
        elif dist_type == 'cosine_similarity':
            distance = F.cosine_similarity(query, prototype, dim=2)
        return distance 