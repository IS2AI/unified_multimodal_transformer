from tqdm.auto import tqdm
from tqdm import trange
import gc
import torch
import torch.nn.functional as F
import random

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
                data_type,
                loss_type,
                wandb=None):

    logs = {}
    logs['train_loss'] = []
    logs['train_acc'] = []
    logs['val_eer'] = []
    logs['val_acc'] = []
    logs['best_acc'] = 0.0
    logs['best_eer'] = float('inf')
    logs['lr'] = []
    logs['train_time_min'] = []
    logs['eval_time_min'] = []
    logs['epoch_time_min'] = []

    for epoch in trange(num_epochs, desc="Epoch"):
        start = timer()
        start_train = timer()


        if loss_type == "metric_learning":
            model, train_loss, train_acc = train_singe_epoch(model=model, 
                                    train_dataloader=train_dataloader,
                                    epoch=epoch, 
                                    n_ways=train_sampler.n_ways, 
                                    n_shots=train_sampler.n_shots, 
                                    n_query=train_sampler.n_query,
                                    criterion=criterion,
                                    optimizer=optimizer,
                                    device=device,
                                    data_type=data_type,
                                    loss_type=loss_type)

        elif loss_type == "classification":
            model, train_loss, train_acc = train_singe_epoch(model=model, 
                                    train_dataloader=train_dataloader,
                                    epoch=epoch, 
                                    n_ways=None, 
                                    n_shots=None, 
                                    n_query=None,
                                    criterion=criterion,
                                    optimizer=optimizer,
                                    device=device,
                                    data_type=data_type,
                                    loss_type=loss_type)
        
        end_train = timer()
        logs['train_time_min'].append((end_train - start_train)/60)
        
        start_val = timer()
        
        model, val_eer, val_acc = evaluate_single_epoch(model, 
                                  valid_dataloader,
                                  epoch,
                                  device,
                                  data_type,
                                  loss_type)

        end_val = timer()
        logs['eval_time_min'].append((end_val - start_val)/60)

        scheduler.step()
        logs['lr'].append(optimizer.param_groups[0]["lr"])
        logs['train_loss'].append(train_loss)
        logs['train_acc'].append(train_acc)
        logs['val_eer'].append(val_eer)
        logs['val_acc'].append(val_acc)


        if logs['best_eer'] > val_eer:
            logs['best_eer'] = val_eer
            torch.save(model.state_dict(), f"{save_dir}/{data_type[0]}_{exp_name}_best_eer.pth")
            print("Best eer model saved at epoch {}".format(epoch))

        if logs['best_acc'] < val_acc:
            logs['best_acc'] = val_acc
            torch.save(model.state_dict(), f"{save_dir}/{data_type[0]}_{exp_name}_best_acc.pth")
            print("Best acc model saved at epoch {}".format(epoch))
        
        end = timer()
        print("Time elapsed:",(end - start)/60," minutes")
        logs['epoch_time_min'].append((end - start)/60)

        torch.save(logs,f'{save_dir}/{data_type[0]}_{exp_name}_logs')
        
        if wandb:
            wandb.log({"train_loss": train_loss, 
                   "train_acc": train_acc,
                   "val_eer": val_eer,
                   "val_acc": val_acc,
                   })
    
    del logs
    gc.collect()

    return model

def train_unimodal(data, label, device, model, criterion, loss_type, n_ways, n_shots, n_query):
    data = data.to(device)
    data = model(data)

    if loss_type == "metric_learning":
        label = torch.arange(n_ways).repeat(n_query)
        label = label.to(device)
        loss, logits = criterion(data, label, n_ways, n_shots, n_query)
        pred = F.softmax(logits,dim=1).argmax(dim=1)

    elif loss_type == "classification":
        label = label.to(device)
        loss = F.cross_entropy(data, label)
        pred = torch.argmax(F.softmax(data,dim=1), dim=1)

    accuracy = (pred == label).sum()/len(label) * 100
    return loss, accuracy


def train_unimodal_mix(data, device, criterion, n_ways, n_shots, n_query):

    label = torch.arange(n_ways).repeat(n_query)
    label = label.to(device)
    loss, logits = criterion(data, label, n_ways, n_shots, n_query)
    pred = F.softmax(logits,dim=1).argmax(dim=1)
    accuracy = (pred == label).sum()/len(label) * 100

    return loss, accuracy


def train_bimodal_mix(data1, data2, label, device, model, criterion, loss_type, n_ways, n_shots, n_query):
    # Note in loss: support, query = data[:p], data[p:] 
    total_loss = 0
    total_accuracy = 0
    
    if loss_type == "metric_learning":
        p = n_shots * n_ways
        data1 = data1.to(device)
        data1 = model(data1)           
        data2 = data2.to(device)
        data2 = model(data2)
        data12 = torch.mean(torch.stack([data1, data2]), dim=0)
        data = torch.zeros_like(data1)

        data_lst = [data1,data2,data12]
        pairs = [(idx1, idx2) for idx1 in range(len(data_lst)) for idx2 in range(len(data_lst))]
        
        for i,j in pairs:
            data[p:] = data_lst[i][p:]
            data[:p] = data_lst[j][:p]
            loss, accuracy = train_unimodal_mix(data, device, criterion, n_ways, n_shots, n_query)
            total_loss += loss
            total_accuracy += accuracy
            
        total_accuracy = total_accuracy/len(pairs)
        
    return total_loss, total_accuracy

def train_trimodal_mix(data1, data2, data3, label, device, model, criterion, loss_type, n_ways, n_shots, n_query):
    # Note in loss: support, query = data[:p], data[p:] 
    total_loss = 0
    total_accuracy = 0
    
    if loss_type == "metric_learning":
        p = n_shots * n_ways
        data1 = data1.to(device)
        data1 = model(data1)           
        data2 = data2.to(device)
        data2 = model(data2)
        data3 = data3.to(device)
        data3 = model(data3)

        data12 = torch.mean(torch.stack([data1, data2]), dim=0)
        data23 = torch.mean(torch.stack([data2, data3]), dim=0)
        data13 = torch.mean(torch.stack([data1, data3]), dim=0)
        data123 = torch.mean(torch.stack([data1, data2, data3]), dim=0)
        
        data = torch.zeros_like(data1)

        data_lst = [data1, data2, data3, data12, data23, data13, data123]
        
        pairs = [(idx1, idx2) for idx1 in range(len(data_lst)) for idx2 in range(len(data_lst))]
        
        for i,j in pairs:
            data[p:] = data_lst[i][p:]
            data[:p] = data_lst[j][:p]
            loss, accuracy = train_unimodal_mix(data, device, criterion, n_ways, n_shots, n_query)
            total_loss += loss
            total_accuracy += accuracy
            
        total_accuracy = total_accuracy/len(pairs)
        
    return total_loss, total_accuracy



def train_singe_epoch(model,
                      train_dataloader, 
                      epoch, 
                      n_ways, 
                      n_shots, 
                      n_query,
                      criterion,
                      optimizer,
                      device,
                      data_type,
                      loss_type):

    model.train()
    pbar = tqdm(train_dataloader, desc=f'Train (epoch = {epoch})', leave=False)  

    total_loss = 0
    total_acc = 0
    for batch in pbar:

        data_type = sorted(data_type)

        if len(data_type) == 1:
            data, label = batch
            loss, accuracy = train_unimodal(data, label, device, model, criterion,  
                                            loss_type, n_ways, n_shots, n_query)
            
        elif len(data_type) == 2: # two modalities
            data1, data2, label = batch
            loss, accuracy = train_bimodal_mix(data1, data2, label, device, model, criterion,
                          loss_type, n_ways, n_shots, n_query)
            
        elif len(data_type) == 3: # three modalities
            data1, data2, data3, label = batch
            loss, accuracy = train_trimodal_mix(data1, data2, data3, label, device, model, criterion,
                          loss_type, n_ways, n_shots, n_query)  
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
                        data_type,
                        loss_type):
    model.eval()
    total_eer = 0
    total_accuracy = 0

    pbar = tqdm(val_dataloader, desc=f'Eval (epoch = {epoch})')

    for batch in pbar:

        data_type = sorted(data_type)
        id1, id2, labels = batch

        if len(data_type) == 1:
            data_id1, _ = id1
            data_id2, _ = id2

            data_id1 = data_id1.to(device)
            data_id2 = data_id2.to(device)
       
            with torch.no_grad():

                if loss_type == "metric_learning":
                    id1_out = model(data_id1)
                    id2_out = model(data_id2)
                elif loss_type == "classification":
                    id1_out = model.pretrained_model(data_id1)
                    id2_out = model.pretrained_model(data_id2)

                cos_sim = F.cosine_similarity(id1_out, id2_out, dim=1)
                eer, scores = EER_(cos_sim, labels)
                accuracy = accuracy_(labels, scores)
                
                total_eer += eer
                total_accuracy += accuracy    
        
        elif len(data_type) == 2:
            data1_id1,data2_id1, _ = id1
            data1_id2,data2_id2, _ = id2

            data1_id1 = data1_id1.to(device)
            data2_id1 = data2_id1.to(device)
       
            data1_id2 = data1_id2.to(device)
            data2_id2 = data2_id2.to(device)
       
            with torch.no_grad():

                if loss_type == "metric_learning":
                    data1_id1 = model(data1_id1)
                    data2_id1 = model(data2_id1)
                    
                    data1_id2 = model(data1_id2)
                    data2_id2 = model(data2_id2)
                    
                    id1_out = torch.mean(torch.stack([data1_id1, data2_id1]), dim=0)
                    id2_out = torch.mean(torch.stack([data1_id2, data2_id2]), dim=0)
                    
                cos_sim = F.cosine_similarity(id1_out, id2_out, dim=1)

                eer, scores = EER_(cos_sim, labels)
                accuracy = accuracy_(labels, scores)

                total_eer += eer
                total_accuracy += accuracy
        
        elif len(data_type) == 3:
            data1_id1,data2_id1,data3_id1, _ = id1
            data1_id2,data2_id2,data3_id2, _ = id2

            data1_id1 = data1_id1.to(device)
            data2_id1 = data2_id1.to(device)
            data3_id1 = data3_id1.to(device)
            
            data1_id2 = data1_id2.to(device)
            data2_id2 = data2_id2.to(device)
            data3_id2 = data3_id2.to(device)
            
            with torch.no_grad():

                if loss_type == "metric_learning":
                    data1_id1 = model(data1_id1)
                    data2_id1 = model(data2_id1)
                    data3_id1 = model(data3_id1)
                    
                    data1_id2 = model(data1_id2)
                    data2_id2 = model(data2_id2)
                    data3_id2 = model(data3_id2)
                    
                    id1_out = torch.mean(torch.stack([data1_id1, data2_id1, data3_id1]), dim=0)
                    id2_out = torch.mean(torch.stack([data1_id2, data2_id2, data3_id2]), dim=0)
                    
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