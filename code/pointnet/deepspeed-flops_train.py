"""This script is the training script for a PointNet."""
import datetime
import numpy as np
from typing import List
import os
import random
import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from model import PointNet, pointnetloss
from utils import PointCloudDataSet, SamplePoints, TREE_SPECIES, classes_to_tensor

from deepspeed.profiling.flops_profiler import FlopsProfiler

def fix_random_seed(seed: int, device=None) -> None:
    """Fix random seeds for reproducibility of all experiments."""
    random.seed(seed) # Python pseudo-random generator
    os.environ['PYTHONHASHSEED'] = str(seed) # Python
    np.random.seed(seed) # NumPy
    torch.manual_seed(seed) # PyTorch
    if device == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    # For now, do not set torch.backends.cudnn.deterministic to True and cudnn.benchmark to False (it is faster without).
    
def create_dataloader(data_folder : str, validation_percentage = 0.15, verbose=True):
    """Return two dataloaders, one for train and one for validation (with validation_percentage of all data samples)."""
    transformations = transforms.Compose(
        [SamplePoints(1024, sample_method="random")])
    # It would be also possible to sample the farthest points:
    # transformations = transforms.Compose([SamplePoints(1024, sample_method = "farthest_points")])
    
    data = PointCloudDataSet(data_folder, train=True,
                             transform=transformations)
    dataset_size = len(data)
    idx = list(range(dataset_size))
    split = int(np.floor(validation_percentage * dataset_size))
    np.random.shuffle(idx)
    train_idx, val_idx = idx[split:], idx[:split]
    
    if verbose:
        print(f"Training is done with {len(train_idx)} samples.")
        print(f"Validation is done with {len(val_idx)} samples.")
    
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    
    train_loader = DataLoader(data, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(data, batch_size=batch_size, sampler=val_sampler)
    
    return train_loader, val_loader
    
def train(model, train_loader, val_loader, optimizer, num_training_epochs = 100, saved_models_path=None):
    """The training loop."""
    for epoch in range(num_training_epochs):
        model.train()
        running_loss = 0.0

        prof = FlopsProfiler(model)

        profile_step = 5
        print_profile= True

        for i, batch in enumerate(train_loader):

            # Load data.
            labels = batch["label"]
            labels = classes_to_tensor(labels).to(device)
            points = batch["points"].to(device)
            
            optimizer.zero_grad()

            # start profiling at training step "profile_step"
            if i == profile_step:
                prof.start_profile()
            
            predictions = model(points)

            # end profiling and print output
            if i == profile_step: # if using multi nodes, check global_rank == 0 as well
                prof.stop_profile()
                flops = prof.get_total_flops()
                macs = prof.get_total_macs()
                params = prof.get_total_params()
                if print_profile:
                    prof.print_model_profile(profile_step=profile_step)
                prof.end_profile()

            predicted_labels, embedding3x3, embedding64x64 = predictions
            
            loss = pointnetloss(predicted_labels, labels, embedding3x3, embedding64x64, device=device)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 500 == 499: # print every n-th minibatch
                print(f"Epoch {epoch + 1} | running loss {running_loss/500}")
                running_loss = 0.0
            
        # Validation.
        model.eval()
        if epoch % 5 == 4: # get validation accuracy every 5 epochs
            with torch.no_grad():
                total_pred = total_correct = 0
                for i, batch in enumerate(val_loader):
                    # Load data.
                    labels = batch["label"]
                    labels = classes_to_tensor(labels).to(device)
                    points = batch["points"].to(device)
                    
                    predicted_labels, _, _ = model(points)
                    predicted = torch.argmax(predicted_labels.data, 1)
                    
                    total_correct += torch.sum(labels == predicted).item() 
                    total_pred += len(labels)
                    
                percent_correct = total_correct / total_pred * 100
                print(f"Epoch {epoch + 1} | validation accuracy {percent_correct}")

        if saved_models_path:
            model_name = "tree_pointnet"
            curr_time = "{0:%Y-%m-%d--%H-%M-%S}".format(datetime.datetime.now())
            save_model_in_file = model_name + "-epoch-" + str(epoch) + "-time-" + curr_time + ".pt"
            torch.save(model.state_dict(), os.path.join(saved_models_path, save_model_in_file))
        
    if saved_models_path:
        model_name = "tree_pointnet"
        curr_time = "{0:%Y-%m-%d--%H-%M-%S}".format(datetime.datetime.now())
        save_model_in_file = model_name + "-epoch-" + "final" + "-time-" + curr_time + ".pt"
        torch.save(model.state_dict(), os.path.join(saved_models_path, save_model_in_file))
        
    return model

if __name__ == "__main__":
    print("Start training")
    
    ### LEARNING PARAMETERS ###
    
    n_classes = len(TREE_SPECIES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU available?
    print(f"Training with: {device}")
    
    saved_models_path = "./saved_models"
    if not os.path.exists(saved_models_path):
        os.makedirs(saved_models_path)
        print(f"Created path for models in {saved_models_path}")
    
    learning_rate = 0.001
    batch_size = 32
    num_training_epochs = 100
    
    
    ### FIX SEEDS ###
    
    fix_random_seed(26, device=device)

    ### DATA ####
    
    data_folder = "/scratch/projects/workshops/forest/synthetic_trees_ten"
    train_loader, val_loader = create_dataloader(data_folder)
    
    ### MODEL ####
    
    model = PointNet(n_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    prof = FlopsProfiler(model)
    profile_step = 5
    print_profile= True

    ### TRAINING LOOP ###
    model = train(model, train_loader, val_loader, optimizer, num_training_epochs = num_training_epochs, saved_models_path=saved_models_path)
    
    print("Finished training.")