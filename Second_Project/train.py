import argparse
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import torch.utils.data
import numpy as np
import datetime
from collections import OrderedDict
import utility_functions as uf
import model_functions as mf

parser = argparse.ArgumentParser(
    description = 'train.py parser'
)

# Getting the arguments
parser.add_argument('data_dir', action="store", default="./flowers/")
parser.add_argument('--arch', action="store", default="vgg16")
parser.add_argument('--save_dir', action="store", default="./checkpoint.pth")
parser.add_argument('--learning_rate', action="store", type=float, default=0.001)
parser.add_argument('--hidden_units', action="store", type=int, default=4096)
parser.add_argument('--epochs', action="store", default=10, type=int)
parser.add_argument('--gpu', action="store_true", default="gpu", dest="gpu")

# Parsing the arguments
args = parser.parse_args()
data_directory = args.data_dir
arch = args.arch
save_directory = args.save_dir
learning_rate = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
gpu = args.gpu

# Selecting device, epochs
device = ['cuda' if gpu else 'cpu'][0]
print(device)
epoch_num = [epochs if epochs else 10][0]

# Defining main function
def main():
    train_loader, valid_loader, test_loader, train_datasets = uf.load_data(data_directory)
    model, criterion, optimiser = mf.model_setup(device, arch, learning_rate, hidden_units)

    print_interval = 20
    train_loss = 0
    train_accuracy = 0
    validation_losses = []
    training_losses = []


    for epoch in range(epoch_num):
        batch_count = 0
        train_loss = 0

        # Switch to training mode
        model.train()

        for images, labels in train_loader:
            start = datetime.datetime.now()
            batch_count += 1

            # Move data to the GPU
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            logits = model(images)
            loss = criterion(logits, labels)

            # Backpropagation
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            # Compute metrics
            probabilities = torch.exp(logits)
            top_probs, top_classes = probabilities.topk(1, dim=1)
            matches = (top_classes == labels.view(*top_classes.shape)).type(torch.FloatTensor)
            accuracy = matches.mean()

            # Reset optimiser gradient and track metrics
            optimiser.zero_grad()
            train_loss += loss.item()
            train_accuracy += accuracy.item()

            training_losses.append(train_loss / print_interval)
            
            # Validate the model every 'print_interval' batches
            if batch_count % print_interval == 0:
                print(f'\nEpoch {epoch + 1}/{epoch_num} || Batch {batch_count}')

                # Print training metrics
                print(f'Training Loss: {train_loss / print_interval:.3f}')
                print(f'Training Accuracy: {train_accuracy / print_interval * 100:.2f}%')

                validation_losses = mf.validation(model, criterion, device, valid_loader, validation_losses)

                # Reset metrics and switch back to training mode
                train_loss = 0
                train_accuracy = 0
                model.train()
                
                # Calculate elapsed time
                epoch_time = datetime.datetime.now() - start
                print(f'Time Training Batch: {epoch_time.seconds // 60:.0f}m {epoch_time.seconds % 60:.0f}s')
                print('-' * 25)

    mf.save_model(model, criterion, optimiser, train_datasets,  arch, learning_rate, hidden_units, epochs = epoch_num)

        

if __name__== "__main__":
    main()