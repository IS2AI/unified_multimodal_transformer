import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from itertools import product
import numpy as np
import torch.nn.functional as F
from itertools import combinations
import torch


def EER_(cos_sim, labels):

    cos_sim = cos_sim.cpu().numpy()
    labels = labels.cpu().numpy()

    fpr, tpr, threshold = roc_curve(labels, cos_sim, pos_label=1)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))] * 100
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    scores = cos_sim > eer_threshold

    return eer, scores


def accuracy_(labels, scores):
    errors = np.absolute(scores-labels.numpy()).sum()
    correct = len(labels) - errors
    return correct / len(labels) * 100

def accuracy_sklearn(labels, scores):
    return accuracy_score(labels, scores)

def get_eer_accuracy(id1, id2, labels, num_eval=1):
    """
    Calculates the Equal Error Rate (EER) and accuracy for the given inputs.

    Args:
        id1 (torch.Tensor): The first input tensor.
        id2 (torch.Tensor): The second input tensor.
        labels (torch.Tensor): The labels tensor.
        num_eval (int): Value that differentiates evaluation process for speaking_faces (when set to 1) and for voxceleb2(when set to 10)

    Returns:
        tuple: A tuple containing the EER and accuracy values.

    """
    if num_eval > 1:
        id1_out_lst = torch.stack(id1)
        id2_out_lst = torch.stack(id2)

        id1_out_lst_norm = id1_out_lst / id1_out_lst.norm(dim=-1)[:, :, None]
        id2_out_lst_norm = id2_out_lst / id2_out_lst.norm(dim=-1)[:, :, None]
        product_res = torch.einsum('kij, bij->kbi', id1_out_lst_norm, id2_out_lst_norm)
        # product_res shape is [10, 10, batch_size]
        res = product_res.view(-1, product_res.shape[2])
        cos_sim = res.mean(dim=0)

        eer, scores = EER_(cos_sim, labels)
        accuracy = accuracy_(labels, scores)
        return eer, accuracy
    else:
        cos_sim = F.cosine_similarity(id1, id2, dim=1)
        eer, scores = EER_(cos_sim, labels)
        accuracy = accuracy_(labels, scores)
        return eer, accuracy

def get_index_pairs(l1, l2):
    """
    Returns a list of index pairs for two input lists.

    Args:
        l1 (list): The first input list.
        l2 (list): The second input list.

    Returns:
        list: A list of index pairs, where each pair consists of the indices from l1 and l2.

    """
    return list(product(range(len(l1)), range(len(l2))))

def calculate_total_eer_accuracy(id1, id2, labels, num_eval=1):
    """
    Calculate the total equal error rate (EER) and accuracy for pairs of IDs.

    Args:
        id1 (list): List of IDs for the first set of samples.
        id2 (list): List of IDs for the second set of samples.
        labels (list): List of labels for the samples.
        num_eval (int): Value that differentiates evaluation process for speaking_faces (when set to 1) and for voxceleb2(when set to 10).

    Returns:
        tuple: A tuple containing two numpy arrays - total_eer and total_accuracy.
            - total_eer: A 2D numpy array representing the EER values for each pair of IDs.
            - total_accuracy: A 2D numpy array representing the accuracy values for each pair of IDs.
    """
    total_eer = np.zeros((len(id1), len(id2)))
    total_accuracy = np.zeros((len(id1), len(id2)))

    for i, j in get_index_pairs(id1, id2):
        eer, accuracy = get_eer_accuracy(id1[i], id2[j], labels, num_eval)
        total_eer[i, j] = eer
        total_accuracy[i, j] = accuracy

    return total_eer, total_accuracy


def calculate_mean_combinations(data_list, num_eval=1):
    """
    Calculates the mean of each pair of tensors in the given data_list.

    Args:
        data_list (list): A list of tensors.
        num_eval (int): Value that differentiates evaluation process for speaking_faces (when set to 1) and for voxceleb2(when set to 10).

    Returns:
        list: A list of tensors, where each tensor is the mean of a pair of tensors from data_list.
    """
    if num_eval > 1:
        mean_combinations = []
        for i in range(len(data_list)):
            for j in range(i + 1, len(data_list)):
                mean_combinations.append(torch.mean(torch.stack([data_list[i], data_list[j]]), dim=0))
        return mean_combinations
    else:
        return [torch.mean(torch.stack(pair), dim=0) for pair in combinations(data_list, 2)]