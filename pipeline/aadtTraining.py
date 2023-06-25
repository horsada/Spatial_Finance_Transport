import pandas as pd
import numpy as np
import os
import torch
import random
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt
import albumentations as album
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import random_split
from sklearn.metrics import accuracy_score
import plotly.express as px
import torchmetrics
from torchmetrics import MeanAbsolutePercentageError
from glob import glob
import plotly.graph_objs as go
import plotly.io as pio

learning_rate = 0.01
batch_size = 256
epochs = 3
loss_fn = nn.MSELoss()
#optimizer = torch.optim.Adam(nn_model.parameters(), lr=learning_rate)

NORMALISED = True

MAPE = MeanAbsolutePercentageError()

def zip_lists_by_name_attribute(list1, list2):
    """
    Zips two lists together based on the 'name' attribute of the elements.

    Args:
        list1 (list): The first list to be zipped.
        list2 (list): The second list to be zipped.

    Returns:
        list: The zipped list of tuples, where each tuple contains elements from list1 and list2
              that have the same 'name' attribute value.
    """
    # Create a dictionary to store the elements of list1 with the 'name' attribute as the key
    dict1 = {elem.name: elem for elem in list1}

    # Create a dictionary to store the elements of list2 with the 'name' attribute as the key
    dict2 = {elem.name: elem for elem in list2}

    # Get the set of keys that are common to both dictionaries
    common_keys = set(dict1.keys()) & set(dict2.keys())

    # Zip the common elements from list1 and list2 together
    zipped_list = [(dict1[key], dict2[key]) for key in common_keys]

    return zipped_list

def data_loading(AADT_PROCESSED_PATH):

    pattern = os.path.join(AADT_PROCESSED_PATH, 'aadt_*.csv')

    processed_aadt_file_paths = [os.path.join(AADT_PROCESSED_PATH, os.path.basename(x)) for x in glob(pattern)]
    print("Processed aadt files: {}".format(processed_aadt_file_paths))

    processed_aadt_df_list = []
    processed_aadt_df_test_list = []

    for i in range(len(processed_aadt_file_paths)):
        processed_aadt_df = pd.read_csv(processed_aadt_file_paths[i])
        processed_aadt_df = processed_aadt_df.loc[:, ~processed_aadt_df.columns.str.contains('^Unnamed')]

        processed_aadt_df['site_name'] = processed_aadt_df['site_name'].astype(str)

        processed_aadt_df.name = processed_aadt_df.iloc[0]['Local Authority']+'_'+processed_aadt_df.iloc[0]['site_name'].replace('/', '_')

        processed_aadt_df_list.append(processed_aadt_df)

    return processed_aadt_df_list

class CustomDataset(Dataset):
    def __init__(self, df, normalised=True):

        self.name = df.name

        self.labels = torch.tensor(df['all_motor_vehicles'].values.astype('float32')).unsqueeze(1)
        self.labels_cars_and_taxis = torch.tensor(df['cars_and_taxis'].values.astype('float32')).unsqueeze(1)
        self.labels_buses_and_coaches = torch.tensor(df['buses_and_coaches'].values.astype('float32')).unsqueeze(1)
        self.labels_lgvs = torch.tensor(df['lgvs'].values.astype('float32')).unsqueeze(1)
        self.labels_all_hgvs = torch.tensor(df['all_hgvs'].values.astype('float32')).unsqueeze(1)

        #self.speed_data = pd.read_csv(SPEED_DATA_PATH)
        #self.road_width = pd.read_csv(ROAD_WIDTH_PATH)
        self.hour = torch.tensor(df['hour'].values.astype('float32')).unsqueeze(1)
        self.avg_mph = torch.tensor(df['avg_mph'].values.astype('float32')).unsqueeze(1)
        self.day = torch.tensor(df['day'].values.astype('float32')).unsqueeze(1)
        self.month = torch.tensor(df['month'].values.astype('float32')).unsqueeze(1)

        if normalised:
            vehicle_types = ['0-520cm_normalised', '521-660cm_normalised', '661-1160cm_normalised', '1160+cm_normalised', 'total_volume_normalised']
        else:
            vehicle_types = ['0-520cm', '521-660cm', '661-1160cm', '1160+cm', 'total_volume']

        self.small_vehicle = torch.tensor(df[vehicle_types[0]].values.astype('float32')).unsqueeze(1)
        self.mid_vehicle = torch.tensor(df[vehicle_types[1]].values.astype('float32')).unsqueeze(1)
        self.large_vehicle = torch.tensor(df[vehicle_types[2]].values.astype('float32')).unsqueeze(1)
        self.very_large_vehicle = torch.tensor(df[vehicle_types[3]].values.astype('float32')).unsqueeze(1)
        self.vehicle_count = torch.tensor(df[vehicle_types[4]].values.astype('float32')).unsqueeze(1)

        self.x = torch.concat((self.vehicle_count, self.small_vehicle, self.mid_vehicle, self.large_vehicle, self.very_large_vehicle, self.avg_mph, self.day, self.month, self.hour), dim=-1)
        self.y = torch.concat((self.labels, self.labels_cars_and_taxis, self.labels_buses_and_coaches, self.labels_lgvs, self.labels_all_hgvs), dim=-1)

        # Reshape y to have shape (batch_size, 5)
        self.y = self.y.view(-1, 5)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.name, self.x[idx], self.y[idx]
    

    
class NeuralNetwork(nn.Module):
    def __init__(self, name):
        super(NeuralNetwork, self).__init__()

        self.name = name
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(9, 7),
            nn.Linear(7,7),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(7,5),
            nn.ReLU()
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class Hyperparameters:
    def __init__(self):
        self.learning_rate = 0.01
        self.batch_size = 256
        self.epochs = 3
        self.loss_fn = nn.MSELoss()

hyperparams = Hyperparameters()

def run_epoch(ep_id, action, loader, model, optimizer, criterion, early_stopper):
    losses = [] # Keep list of accuracies to track progress
    is_training = action == "train" # True when action == "train", else False

    # Looping over all batches
    for batch_idx, batch in enumerate(loader):
        dl_name, x, y = batch

        # Assert we are training the correct model with the correct dataset!
        assert dl_name[0] == model.name

        # Resetting the optimizer gradients
        optimizer.zero_grad()

        # Setting model to train or test
        with torch.set_grad_enabled(is_training):

            # Feed batch to model
            logits = model(x)

            # Determine the batch size and reshape logits and y accordingly
            batch_size = logits.size(0)
            logits = logits.view(batch_size, -1)
            y = y.view(batch_size, -1)

            #print("y shape: {}".format(y.shape))
            #print("logits shape: {}".format(logits.shape))
            #print("y values: {}".format(y))
            #print("logits values: {}".format(logits))

            # Calculate the loss based on predictions and real labels
            loss = criterion(logits, y)

            #print("loss: {}".format(loss.item()))
            mape_loss = MAPE(logits, y)

            # If training, perform backprop and update weights
            if is_training:
                loss.backward()
                optimizer.step()

            # Append current batch accuracy
            losses.append(mape_loss.detach().numpy())

            # Print some stats every 50th batch
            if batch_idx % 50 == 0:
                print(f"{action.capitalize()}, Epoch: {ep_id+1}, Batch {batch_idx}: Loss = {loss.item()}")

        if not is_training:
            if early_stopper.early_stop(mape_loss.detach().numpy()):
                print("Entered Early Stopping")
                break

    # Return accuracies to main loop
    return losses

def train(epochs, train_dl, val_dl, model, optimizer, criterion, early_stopper):

    # Keep lists of accuracies to track performance on train and test sets
    train_losses = []
    val_losses = []

    # Looping over epochs
    for epoch in range(epochs):

        # Looping over train set and training
        train_loss = run_epoch(epoch, "train", train_dl, model, optimizer, criterion, early_stopper=early_stopper)

        # Looping over test set
        val_loss = run_epoch(epoch, "val", val_dl, model, optimizer, criterion, early_stopper=early_stopper)

        # Collecting stats
        train_losses += train_loss
        val_losses += val_loss

    return train_losses, val_losses

def main(epochs, train_dl_list, val_dl_list, model_list):
    all_train_losses = []
    all_val_losses = []

    for (model, train_dl, val_dl) in zip(model_list, train_dl_list, val_dl_list):

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        loss_fn = nn.MSELoss()

        early_stopper = EarlyStopper(patience=3, min_delta=10)

        train_losses, val_losses = train(epochs, train_dl, val_dl, model, optimizer=optimizer, criterion=loss_fn, early_stopper=early_stopper)

        all_train_losses.append((model.name, train_losses))
        all_val_losses.append((model.name, val_losses))

    return all_train_losses, all_val_losses


def aadt_training(AADT_PROCESSED_PATH, NN_MODEL_PATH):

    processed_aadt_df_list = data_loading(AADT_PROCESSED_PATH)
                 
    dataset_list = []

    for df in processed_aadt_df_list:

        custom_dataset = CustomDataset(df, normalised=NORMALISED)
        dataset_list.append(custom_dataset)

    nn_model_list = []

    for df in processed_aadt_df_list:
        nn_model = NeuralNetwork(name=df.name)
        nn_model_list.append(nn_model)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.ones_(m.weight)
            m.bias.data.fill_(0.01)

    for nn_model in nn_model_list:
        nn_model.apply(init_weights)

    train_dataloader_list = []
    val_dataloader_list = []

    for dataset in dataset_list:
        train_split = 0.8
        train_data, val_data = random_split(dataset, [train_split, 1-train_split])

        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, sampler=None,
                        batch_sampler=None, num_workers=0, collate_fn=None,
                        pin_memory=False, drop_last=False, timeout=0,
                        worker_init_fn=None,
                        persistent_workers=False)

        val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, sampler=None,
                        batch_sampler=None, num_workers=0, collate_fn=None,
                        pin_memory=False, drop_last=False, timeout=0,
                        worker_init_fn=None,
                        persistent_workers=False)

        train_dataloader_list.append(train_dataloader)
        val_dataloader_list.append(val_dataloader)

    all_train_losses, all_val_losses = main(epochs=epochs, 
                    train_dl_list=train_dataloader_list, val_dl_list=val_dataloader_list, 
                    model_list=nn_model_list)
    
    for nn_model in nn_model_list:

        print("nn model name: {}".format(nn_model.name))

        if 'Blackburn with Darwen' in nn_model.name:
            nn_model.name = nn_model.name.replace('Blackburn with Darwen', 'blackburn')

        print("saving model to: {}".format(NN_MODEL_PATH+"nn_model_{}.pth".format(nn_model.name.lower())))
        torch.save(nn_model.state_dict(), NN_MODEL_PATH+"nn_model_{}.pth".format(nn_model.name.lower()))

    return True