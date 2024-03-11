import pandas as pd 
from sklearn.model_selection import train_test_split
from dataset_class import MINISTTrainDataset, MINISTValDataset

import numpy as np 
from torch.utils.data import DataLoader, Dataset

def get_loaders(train_df_dir, test_df_dir, batch_size):
    train_df = pd.read_csv(train_df_dir)
    test_df = pd.read_csv(test_df_dir)

    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=26)

    train_dataset = MINISTTrainDataset(images=train_df.iloc[:, 1:].values.astype(np.uint8),
                                       labels=train_df.iloc[:, 0].values,
                                       indicies=train_df.index.values)
    val_dataset = MINISTValDataset(images=val_df.iloc[:, 1:].values.astype(np.uint8),
                                       labels=val_df.iloc[:, 0].values,
                                       indicies=val_df.index.values)
    test_dataset = MINISTValDataset(images=test_df.iloc[:, 1:].values.astype(np.uint8),
                                       labels=test_df.iloc[:, 0].values.astype(np.uint8),
                                       indicies=test_df.index.values)
    
    # put them into dataloader
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                                  shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                  shuffle=False)
    
    return train_dataloader, val_dataloader, test_dataloader
    
    