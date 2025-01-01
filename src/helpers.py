# Necessary Imports
import numpy as np
import torch

# Function to read labels from text file
def read_labels(label_path):
    # Initialize an empty list to store the label data
    labels = []
    
    # Open and read the label file
    with open(label_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Split the line into individual elements
            elements = line.strip().split()
            # Convert elements to float
            elements = [float(e) for e in elements]
            # Append the elements as a list to the labels list
            labels.append(elements)
    
    # Convert the labels list to a NumPy array
    labels_array = np.array(labels)
    
    return labels_array

# Collate function for dataloader
def collate_fn(batch):
    im, label, path = zip(*batch)
    for i, lb in enumerate(label):
        lb[:, 0] = i
    return torch.stack(im, 0), torch.cat(label, 0), path