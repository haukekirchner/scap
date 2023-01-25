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

from deepspeed.profiling.flops_profiler import FlopsProfiler

from model import PointNet, pointnetloss
from utils import PointCloudDataSet, SamplePoints, TREE_SPECIES, classes_to_tensor
import sys, getopt

from torch.utils.tensorboard import SummaryWriter

from torch.profiler import profile, record_function, ProfilerActivity

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
    
def create_dataloader(data_folder : str,batch_size, validation_percentage = 0.15, verbose=True, do_sample_points = True):
    """Return two dataloaders, one for train and one for validation (with validation_percentage of all data samples)."""
    
    if do_sample_points:
        transformations = transforms.Compose(
            [SamplePoints(1024, sample_method="random")])
        # It would be also possible to sample the farthest points:
        # transformations = transforms.Compose([SamplePoints(1024, sample_method = "farthest_points")])
        
        data = PointCloudDataSet(data_folder, train=True,
                             transform=transformations)
    else:
        data = PointCloudDataSet(data_folder, train=True)

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
    
def train(model, train_loader, val_loader, optimizer, device, num_training_epochs = 100, saved_models_path=None, logdir=None, tensorboard = False, deepspeed = False, profiler_torch = False):
    
    if deepspeed:
        flop_prof = FlopsProfiler(model)

        profile_step = 5
        print_profile= True

    """The training loop."""
    
    if tensorboard:
        loss_idx_value = 0
    
    for epoch in range(num_training_epochs):
        model.train()
        running_loss = 0.0
        
        if profiler_torch:
            # profiler #####################################################
            prof = torch.profiler.profile(
                    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(logdir + "/profiler"),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True)

            prof.start()
            ################################################################
        
        for i, batch in enumerate(train_loader):
        
            if deepspeed:
                if i == profile_step:
                    flop_prof.start_profile()

            #if profiler_torch:
            #    # profiler #############################################
            #    if i >= (1 + 1 + 3) * 2:
            #        break
                ########################################################
               
            # Load data.
            labels = batch["label"]
            labels = classes_to_tensor(labels).to(device)
            points = batch["points"].to(device)
            optimizer.zero_grad()
                
            predictions = model(points)
            predicted_labels, embedding3x3, embedding64x64 = predictions
            
            loss = pointnetloss(predicted_labels, labels, embedding3x3, embedding64x64, device=device)
            
            if deepspeed:
                # end profiling and print output
                if i == profile_step: # if using multi nodes, check global_rank == 0 as well
                    flop_prof.stop_profile()
                    flops = flop_prof.get_total_flops()
                    macs = flop_prof.get_total_macs()
                    params = flop_prof.get_total_params()
                    if print_profile:
                        flop_prof.print_model_profile(profile_step=profile_step)
                    flop_prof.end_profile()

            loss.backward()
            optimizer.step()

            if profiler_torch:
                # profiler
                prof.step()  # Need to call this at the end of each step to notify profiler of steps' boundary.
                ##########

            running_loss += loss.item()
            
            if tensorboard:
                writer.add_scalar("Loss/Minibatches", running_loss, loss_idx_value)
                loss_idx_value += 1
            if i % 500 == 499: # print every n-th minibatch
                print(f"Epoch {epoch + 1} | running loss {running_loss/500}")
                running_loss = 0.0
        
        # Write loss for epoch
        if tensorboard:
            writer.add_scalar("Loss/Epochs", running_loss, epoch)
        

        if profiler_torch:
            prof.stop()

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
                
                if tensorboard:
                    writer.add_scalar("Accuracy", percent_correct, epoch)
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

def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
         raise ValueError

if __name__ == "__main__":


    # Initialize the SummaryWriter for TensorBoard
    # Its output will be written to ./runs/

    batch_size = 32
    do_sample_points = False
    
    logdir = ''
    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv,"hl:t:p:d:s:b:",["logdir=","tensorboard=","profiler_torch","deepspeed=","sample_points=","batch_size="])
    for opt, arg in opts:
        if opt == '-h':
            print ('train.py -l <logdir>')
            sys.exit()
        elif opt in ("-l", "--logdir"):
            logdir = arg
        elif opt in ("-t", "--tensorboard"):
            tensorboard = str_to_bool(arg)
        elif opt in ("-p", "--profiler_torch"):
            profiler_torch = str_to_bool(arg)
        elif opt in ("-d", "--deepspeed"):
            deepspeed = str_to_bool(arg)
        elif opt in ("-s", "--sample_points"):
            do_sample_points = str_to_bool(arg)
        elif opt in ("-b", "--batch_size"):
            print(arg)
            print(type(arg))
            batch_size = int(arg)
    print ('logdir is ', logdir)

    print("Tensorboard : ", tensorboard, " ", type(tensorboard))
    print("profiler_torch : ", profiler_torch, " ", type(profiler_torch))
    print("deepspeed : ", deepspeed, " ", type(deepspeed))

    if tensorboard:
        writer = SummaryWriter(log_dir=logdir + "/tensorboard")

    print("Start training")
    
    ### LEARNING PARAMETERS ###
    
    n_classes = len(TREE_SPECIES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU available?
    print(f"Training with: {device}")
    
    #saved_models_path = "./saved_models"
    saved_models_path = None
    #if not os.path.exists(saved_models_path):
    #    os.makedirs(saved_models_path)
    #    print(f"Created path for models in {saved_models_path}")
    
    learning_rate = 0.001
    num_training_epochs = 15
    
    
    ### FIX SEEDS ###
    
    fix_random_seed(26, device=device)

    ### DATA ####
    
    if do_sample_points:
        data_folder = "/scratch/projects/workshops/forest/synthetic_trees_ten"
    else:
        data_folder = "/scratch/projects/workshops/forest/synthetic_trees_ten_sampled"

    train_loader, val_loader = create_dataloader(data_folder, do_sample_points=do_sample_points, batch_size=batch_size)
    
    ### MODEL ####
    
    model = PointNet(n_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    ### TRAINING LOOP ###
    model = train(model, train_loader, val_loader, optimizer, num_training_epochs = num_training_epochs, saved_models_path=saved_models_path, logdir=logdir, device=device, tensorboard = tensorboard, deepspeed = deepspeed, profiler_torch = profiler_torch)
    
    print("Finished training.")
