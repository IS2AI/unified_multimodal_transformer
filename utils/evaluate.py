from tqdm.auto import tqdm
import torch
import numpy as np
import pandas as pd

from utils.metrics import get_eer_accuracy, calculate_mean_combinations, calculate_total_eer_accuracy, accuracy_
from utils.data_preprocessing import preprocess_and_infer, results_to_csv, preprocess_and_infer_train

def evaluate_pair(model, id1, id2, labels, device, dataset_type, num_eval, modalities):
    """
    Evaluates a pair of multimodal data, supporting both bi-modal and tri-modal cases.

    Args:
        model: The multimodal model.
        id1: First data point (list of tensors).
        id2: Second data point (list of tensors).
        labels: Ground truth labels.
        device: Device to use (e.g., 'cuda').
        modalities: Number of modalities (1 or 2). Used for VX2 dataset.

    Returns:
        total_eer: Total Equal Error Rate.
        total_accuracy: Total accuracy.
    """
    
    id1 = [t.to(device) for t in id1] if dataset_type == "SF" else id1
    id2 = [t.to(device) for t in id2] if dataset_type == "SF" else id2

    with torch.no_grad():
        if dataset_type == "SF" or (dataset_type == "VX2" and modalities == 3):
            processed_id1 = preprocess_and_infer(id1[:-1], model, device)
            processed_id2 = preprocess_and_infer(id2[:-1], model, device)

            id1_combined = processed_id1  + calculate_mean_combinations(processed_id1) + [torch.mean(torch.stack(processed_id1), dim=0)]
            id2_combined = processed_id2  + calculate_mean_combinations(processed_id2) + [torch.mean(torch.stack(processed_id2), dim=0)] 

            return calculate_total_eer_accuracy(id1_combined, id2_combined, labels)

        if dataset_type == "VX2":
            if modalities == 1:
                data_id1_lst, _ = id1 # was data_id1_lst,_
                data_id2_lst, _ = id2
                
                id1_out = [model(data.to(device)) for data in data_id1_lst]
                id2_out = [model(data.to(device)) for data in data_id2_lst]
                return get_eer_accuracy(id1_out, id2_out, labels, num_eval=10)

            if modalities == 2:
                processed_id1 = preprocess_and_infer([[t.to(device) for t in l] for l in id1[:-1]], model, device, num_eval=10)
                processed_id2 = preprocess_and_infer([[t.to(device) for t in l] for l in id2[:-1]], model, device, num_eval=10)
                id1_combined = processed_id1 + [[torch.mean(torch.stack([processed_id1[0][i], processed_id1[1][i]]), dim=0) for i in range(len(processed_id1[0]))]]
                id2_combined = processed_id2 + [[torch.mean(torch.stack([processed_id2[0][i], processed_id2[1][i]]), dim=0) for i in range(len(processed_id2[0]))]]
                return calculate_total_eer_accuracy(id1_combined, id2_combined, labels, num_eval=10)

    raise ValueError(f"Unsupported dataset type: {dataset_type}")

def evaluate(model, val_dataloader, epoch, num_eval, device, data_type, loss_type, mode, save_dir, exp_name, path_to_valid_list, dataset_type):
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
    total_eer = [] if mode == 'test' and data_type_len > 1 else 0
    total_accuracy = [] if mode == 'test' and data_type_len > 1 else 0

    with tqdm(val_dataloader, desc=f'Eval (epoch = {epoch})') as pbar:
        for batch in pbar:
            id1, id2, labels = batch
            
            if mode == "train" or (mode == 'test' and data_type_len == 1 and dataset_type == "SF"):
                id1_out = preprocess_and_infer_train(id1, model, device, data_type_len)
                id2_out = preprocess_and_infer_train(id2, model, device, data_type_len)

                if data_type_len > 1:
                    id1_out = id1_out.view(data_type_len, -1, id1_out.size(-1)).mean(dim=0)
                    id2_out = id2_out.view(data_type_len, -1, id2_out.size(-1)).mean(dim=0)
                eer, accuracy = get_eer_accuracy(id1_out, id2_out, labels)
                total_eer += eer
                total_accuracy += accuracy
                
            elif mode == 'test':
                eer, accuracy = evaluate_pair(model, id1, id2, labels, device, dataset_type, num_eval, data_type_len)
                if isinstance(total_eer, list):
                    total_eer.append(eer)
                    total_accuracy.append(accuracy)
                else:
                    total_eer += eer
                    total_accuracy += accuracy

    if mode == "train" or (mode == 'test' and data_type_len == 1):
        avg_eer = total_eer / len(val_dataloader)
        print("\nAverage val eer: {}".format(avg_eer))

        avg_accuracy = total_accuracy / len(val_dataloader)
        print("\nAverage val accuracy: {}".format(avg_accuracy))

        return model, avg_eer, avg_accuracy
    
    if mode == 'test' and data_type_len > 1:
        val_eer = np.array(total_eer).mean(axis=0)
        val_acc = np.array(total_accuracy).mean(axis=0)
        results_to_csv(val_eer, val_acc, data_type, save_dir, exp_name, path_to_valid_list, dataset_type)
    return model, total_eer, total_accuracy
