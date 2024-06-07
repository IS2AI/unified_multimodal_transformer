import pandas as pd 
from utils.metrics import get_index_pairs
import copy
import os
import torch

def results_to_csv(val_eer, val_acc, data_type, save_dir, exp_name, path_to_valid_list, dataset_type):
    """
    Saves the evaluation results to a CSV file.

    Args:
        val_eer: Array of equal error rates.
        val_acc: Array of accuracies.
        data_type: List of data types.
        save_dir: Directory to save the CSV file.
        exp_name: Experiment name.
        path_to_valid_list: Path to the validation list.
        dataset_type: Type of dataset (e.g., 'train', 'test').

    Returns:
        df: DataFrame containing the evaluation results.
    """
    df = pd.DataFrame(columns=["data_type_1", "data_type_2", "EER", "accuracy"])
    data_type = sorted(data_type)
    modalities = copy.copy(data_type)

    if len(data_type) == 3:
        modalities.append("_".join([data_type[0], data_type[1]]))
        modalities.append("_".join([data_type[0], data_type[2]]))
        modalities.append("_".join([data_type[1], data_type[2]]))
        modalities.append("_".join(data_type))
    elif len(data_type) == 2:
        modalities.append("_".join(data_type))

    indices = get_index_pairs(modalities, modalities)

    for i, j in indices:
        df.loc[len(df.index)] = [modalities[i], modalities[j], val_eer[i, j], val_acc[i, j]]

    if "valid" in path_to_valid_list:
        file_name = os.path.join(save_dir, exp_name + "{}_valid_results.csv".format(dataset_type))
    else:
        file_name = os.path.join(save_dir, exp_name + "{}_test_results.csv".format(dataset_type))

    df.to_csv(file_name)
    return df


def preprocess_and_infer(data_list, model, device, num_eval=1):
    """
    Preprocesses the given data list and performs inference using the provided model on each data item.

    Args:
        data_list (list): A list of data items to be preprocessed and inferred.
        model: The model to be used for inference.
        device: The device on which the inference should be performed.
        num_eval: TODO(yuliya) write that we need 10x10 arguments for vox_celeb

    Returns:
        list: A list of processed data items after inference.
    """
    if num_eval > 1:
        processed_data_list = [ [model(data) for data in item_list] for item_list in data_list ]
    else:
        processed_data_list = [model(data) for data in data_list]
    return processed_data_list

def preprocess_and_infer_train(id, model, device, data_type_len):
    """
    Preprocesses the input data and performs inference using the provided model.

    Args:
        id (torch.Tensor or list of torch.Tensor): The input data ID(s).
        model (torch.nn.Module): The model used for inference.
        device (torch.device): The device to perform the inference on.
        data_type_len (int): The length of the data type.

    Returns:
        torch.Tensor: The output of the model after performing inference.
    """
    data_id = torch.cat(id, dim=0).to(device) if data_type_len > 1 else id[0].to(device)
    with torch.no_grad():
        return model(data_id)