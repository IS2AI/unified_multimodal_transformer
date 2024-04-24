from tqdm.auto import tqdm
from tqdm import trange
import torch
from utils.metrics import EER_
from utils.metrics import get_eer_accuracy
from utils.metrics import calculate_mean_combinations
from utils.metrics import calculate_total_eer_accuracy

from utils.metrics import accuracy_
from utils.data_processing import preprocess_and_infer,results_to_csv, preprocess_and_infer_train
from timeit import default_timer as timer
from itertools import product
import numpy as np

import pandas as pd 




def evaluate_pair(model, id1, id2, labels, device, modalities):
    """
    Evaluates a pair of multimodal data, supporting both bi-modal and tri-modal cases.

    Args:
        model: The multimodal model.
        id1: First data point (list of tensors).
        id2: Second data point (list of tensors).
        labels: Ground truth labels.
        device: Device to use (e.g., 'cuda').
        modalities: Number of modalities (2 or 3).

    Returns:
        total_eer: Total Equal Error Rate.
        total_accuracy: Total accuracy.
    """

    id1_data = [item.to(device) for item in id1[:-1]]
    id2_data = [item.to(device) for item in id2[:-1]]
    
    with torch.no_grad():
        processed_id1_data = preprocess_and_infer(id1_data, model, device)
        processed_id2_data = preprocess_and_infer(id2_data, model, device)
        
        mean_all_id1 = [torch.mean(torch.stack(processed_id1_data), dim=0)]
        mean_all_id2 = [torch.mean(torch.stack(processed_id2_data), dim=0)]
        
 
        mean_combinations_id1 = calculate_mean_combinations(processed_id1_data)
        mean_combinations_id2 = calculate_mean_combinations(processed_id2_data)
        
        id1_combined = processed_id1_data + mean_all_id1 + mean_combinations_id1
        id2_combined = processed_id2_data + mean_all_id2 + mean_combinations_id2

        
        total_eer, total_accuracy = calculate_total_eer_accuracy(id1_combined, id2_combined, labels)
    

    return total_eer, total_accuracy     
        
def evaluate(model, val_dataloader, epoch, device, data_type, loss_type, mode):
    """
    Evaluate the performance of a model on a validation dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        val_dataloader (torch.utils.data.DataLoader): The validation dataloader.
        epoch (int): The current epoch number.
        device (torch.device): The device to perform the evaluation on.
        data_type (list): A list of data types to evaluate. Each data type represents a different modality.
        loss_type (str): The type of loss function to use.
        mode (str): The evaluation mode. Can be either "train" or "evaluate".

    Returns:
        tuple: A tuple containing the model, the average equal error rate (EER), and the average accuracy.
            If mode is "train" or (mode is "evaluate" and len(data_type) == 1), the tuple contains the model, avg_eer, and avg_accuracy.
            If mode is "evaluate" and len(data_type) > 1, the tuple contains the model, total_eer, and total_accuracy.
    """
    
    model.eval()
    data_type_len = len(data_type)
    
    if mode == "train" or (mode == 'evaluate' and data_type_len == 1):
        total_eer = 0
        total_accuracy = 0
        
        pbar = tqdm(val_dataloader, desc=f'Eval (epoch = {epoch})')
        
        for batch in pbar:
            id1, id2, labels = batch

            id1_out = preprocess_and_infer_train(id1, model, device, data_type_len)
            id2_out = preprocess_and_infer_train(id2, model, device, data_type_len)
            
            if data_type_len > 1:
                id1_out = id1_out.view(data_type_len, -1, id1_out.size(-1)).mean(dim=0)
                id2_out = id2_out.view(data_type_len, -1, id2_out.size(-1)).mean(dim=0)
                
            eer,accuracy = get_eer_accuracy(id1_out, id2_out, labels)
            total_eer += eer
            total_accuracy += accuracy
            
        avg_eer = total_eer / len(val_dataloader)
        print("\nAverage val eer: {}".format(avg_eer))
        avg_accuracy = total_accuracy / len(val_dataloader)
        print("\nAverage val accuracy: {}".format(avg_accuracy))
        return model, avg_eer, avg_accuracy
    
    if mode == 'evaluate':
        total_eer = []
        total_accuracy = []
        
        pbar = tqdm(val_dataloader, desc=f'Eval (epoch = {epoch})')
        
        for batch in pbar:
            
            data_type = sorted(data_type)
            id1, id2, labels = batch
            
            eer,accuracy = evaluate_pair(model, id1, id2, labels, device, modalities=len(data_type))
            total_eer.append(eer)
            total_accuracy.append(accuracy)
        
        return model, total_eer, total_accuracy
        
        

        




