from tqdm import tqdm
from tqdm import trange
import gc
import torch
import torch.nn.functional as F

from speaker_verification.loss import PrototypicalLoss
from speaker_verification.metrics import EER_
from speaker_verification.metrics import accuracy_
from timeit import default_timer as timer

def train_model(model, 
                train_dataloader, 
                valid_dataloader,
                train_sampler,
                criterion,
                optimizer,
                scheduler,
                device,
                num_epochs,
                save_dir,
                exp_name,
                modality='rgb'):

    logs = {}
    logs['train_loss'] = []
    logs['train_acc'] = []
    logs['val_eer'] = []
    logs['val_acc'] = []
    logs['best_acc'] = 0.0
    logs['best_eer'] = float('inf')

    logs['train_time_min'] = []
    logs['eval_time_min'] = []
    logs['epoch_time_min'] = []

    

    for epoch in trange(num_epochs, desc="Epoch"):
        start = timer()
        start_train = timer()
        
        model, train_loss, train_acc = train_singe_epoch(model, 
                                  train_dataloader,
                                  epoch, 
                                  train_sampler.n_ways, 
                                  train_sampler.n_shots, 
                                  train_sampler.n_query,
                                  criterion,
                                  optimizer,
                                  device,
                                  modality)
        
        end_train = timer()
        logs['train_time_min'].append((end_train - start_train)/60)
        
        start_val = timer()
        
        model, val_eer, val_acc = evaluate_single_epoch(model, 
                                  valid_dataloader,
                                  epoch,
                                  device,
                                  modality)

        end_val = timer()
        logs['eval_time_min'].append((end_val - start_val)/60)

        scheduler.step()

        logs['train_loss'].append(train_loss)
        logs['train_acc'].append(train_acc)
        logs['val_eer'].append(val_eer)
        logs['val_acc'].append(val_acc)


        if logs['best_eer'] > val_eer:
            logs['best_eer'] = val_eer
            torch.save(model.state_dict(), f"{save_dir}/{modality}_{exp_name}_best_eer.pth")
            print("Best eer model saved at epoch {}".format(epoch))

        if logs['best_acc'] < val_acc:
            logs['best_acc'] = val_acc
            torch.save(model.state_dict(), f"{save_dir}/{modality}_{exp_name}_best_acc.pth")
            print("Best acc model saved at epoch {}".format(epoch))
        
        end = timer()
        print("Time elapsed:",(end - start)/60," minutes")
        logs['epoch_time_min'].append((end - start)/60)

        torch.save(logs,f'{save_dir}/{modality}_{exp_name}_logs')
    
    del logs
    gc.collect()

    return model

def train_singe_epoch(model,
                      train_dataloader, 
                      epoch, 
                      n_ways, 
                      n_shots, 
                      n_query,
                      criterion,
                      optimizer,
                      device,
                      modality):

    model.train()
    pbar = tqdm(train_dataloader, desc=f'Train (epoch = {epoch})', leave=False)  

    total_loss = 0
    total_acc = 0
    for batch in pbar:

        if modality == "rgb":
            # data_wav, data_rgb, data_thr, label
            _,rgb, _, _ = batch # we do not use labels from dataset
            data = rgb.to(device)
        elif modality == "thr":
            # data_wav, data_rgb, data_thr, label
            _,_, thr, _ = batch # we do not use labels from dataset
            data = thr.to(device)
        elif modality == "wav":
            wav, _, _, _ = batch # we do not use labels from dataset
            data = wav.to(device)
            
        data = model(data)

        label = torch.arange(n_ways).repeat(n_query)
        label = label.to(device)

        loss, logits = criterion(data, label, n_ways, n_shots, n_query)

        pred = F.softmax(logits,dim=1).argmax(dim=1)
        accuracy = (pred == label).sum()/len(label) * 100
        
        total_loss += loss.item()
        total_acc += accuracy.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    avg_loss = total_loss / len(train_dataloader)
    avg_acc = total_acc / len(train_dataloader)

    print()
    print(f"Average train loss: {avg_loss}")
    print(f"Average train accuracy: {avg_acc}")

    return model, avg_loss, avg_acc

def evaluate_single_epoch(model,
                        val_dataloader,
                        epoch, 
                        device, 
                        modality='rgb'):
    model.eval()
    total_eer = 0
    total_accuracy = 0

    pbar = tqdm(val_dataloader, desc=f'Eval (epoch = {epoch})')

    for batch in pbar:

        id1, id2, labels = batch

        wav_id1, rgb_id1, thr_id1, _ = id1
        wav_id2, rgb_id2, thr_id2, _ = id2

        if modality == "rgb":
            data_id1 = rgb_id1.to(device)
            data_id2 = rgb_id2.to(device)

        elif modality == "thr":
            data_id1 = thr_id1.to(device)
            data_id2 = thr_id2.to(device)

        elif modality == "wav":
            data_id1 = wav_id1.to(device)
            data_id2 = wav_id2.to(device)

        with torch.no_grad():
            id1_out = model(data_id1)
            id2_out = model(data_id2)

            cos_sim = F.cosine_similarity(id1_out, id2_out, dim=1)
            eer, scores = EER_(cos_sim, labels)
            accuracy = accuracy_(labels, scores)

            total_eer += eer
            total_accuracy += accuracy
    
    avg_eer = total_eer / len(val_dataloader)
    print("\nAverage val eer: {}".format(avg_eer))

    avg_accuracy = total_accuracy / len(val_dataloader)
    print("\nAverage val accuracy: {}".format(avg_accuracy))

    return model, avg_eer, avg_accuracy
    

# when we do not use lists and create pairs with ValidSampler
def evaluate_single_epoch_1(model,
                          val_dataloader,
                          epoch,
                          device):
    model.eval()
    total_eer = 0
    total_accuracy = 0

    pbar = tqdm(val_dataloader, desc=f'Eval (epoch = {epoch})')
    for batch in pbar:
        _, data_rgb, _, _ = batch
        data = data_rgb.to(device)

        with torch.no_grad():
            data = model(data)
            
            pairs_size = (data.shape[0]-1) // 2
            main_id = data[0].repeat(pairs_size*2).reshape(pairs_size*2, data.shape[1])
            compare_id = data[1:]
            labels = torch.cat((torch.ones(pairs_size), torch.zeros(pairs_size)))

            cos_sim = F.cosine_similarity(main_id, compare_id)
            eer, scores = EER_(cos_sim, labels)
            accuracy = accuracy_(labels, scores)
            
            total_eer += eer
            total_accuracy += accuracy

    avg_eer = total_eer / len(val_dataloader)
    print("\nAverage val eer: {}".format(avg_eer))

    avg_accuracy = total_accuracy / len(val_dataloader)
    print("\nAverage val accuracy: {}".format(avg_accuracy))

    return model, avg_eer, avg_accuracy