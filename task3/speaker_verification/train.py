from tqdm import tqdm
from tqdm import trange
import gc
import torch
import torch.nn.functional as F

from speaker_verification.metrics import EER_
from speaker_verification.metrics import accuracy_


def train_model(model, 
                train_dataloader, 
                valid_dataloader,
                train_sampler,
                criterion,
                optimizer,
                scheduler,
                device,
                num_epochs,
                save_dir):

    logs = {}
    logs['train_loss'] = []
    logs['train_acc'] = []
    logs['val_eer'] = []
    logs['val_acc'] = []
    logs['best_acc'] = 0.0

    for epoch in trange(num_epochs, desc="Epoch"):
        
        model, train_loss, train_acc = train_singe_epoch(model, 
                                  train_dataloader,
                                  epoch, 
                                  train_sampler.n_ways, 
                                  train_sampler.n_shots, 
                                  train_sampler.n_query,
                                  criterion,
                                  optimizer,
                                  device)
        
        model, val_eer, val_acc = evaluate_single_epoch(model, 
                                  valid_dataloader,
                                  epoch,
                                  device)
        scheduler.step()

        logs['train_loss'].append(train_loss)
        logs['train_acc'].append(train_acc)
        logs['val_eer'].append(val_eer)
        logs['val_acc'].append(val_acc)

        if logs['best_acc'] < val_acc:
            logs['best_acc'] = val_acc

        torch.save(logs,f'{save_dir}/logs')
    
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
                      device):
    model.train()
    pbar = tqdm(train_dataloader, desc=f'Train (epoch = {epoch})', leave=False)  

    total_loss = 0
    total_acc = 0
    for batch in pbar:

        _, data_rgb, _, _ = batch # we do not use labels from dataset
        data = data_rgb.to(device)
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
    print("\nAverage train loss: {}".format(avg_loss))

    avg_acc = total_acc / len(train_dataloader)
    print("\nAverage train accuracy: {}".format(avg_acc))

    return model, avg_loss, avg_acc


def evaluate_single_epoch(model,
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