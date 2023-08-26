import torch
from torch import nn
import torch.utils.data
from collections import OrderedDict
from torchvision import models

def model_import(arch = 'vgg16'):
    if arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif arch == 'resnet152':
        model = models.resnet152(pretrained=True)
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)
    return model


def model_setup(device, arch = 'vgg16', learning_rate = 0.001, hidden_units = 4096):
    model = model_import(arch)
    
    # Let's redefine the classifier
    model.classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(25088, hidden_units)),
                                ('relu1', nn.ReLU()),
                                ('dropout1', nn.Dropout(p = 0.2)),
                                ('fc2', nn.Linear(hidden_units, 2048)),
                                ('relu2', nn.ReLU()),
                                ('dropout2', nn.Dropout(p = 0.2)),
                                ('fc3', nn.Linear(2048, 1024)),
                                ('relu3', nn.ReLU()),
                                ('fc4', nn.Linear(1024, 102)),
                                ('output', nn.LogSoftmax(dim =1))
                                ]))
    
    # Let's move model to gpu
    model = model.to(device)

    # Let's define the criterion as Negative Log Likelihood Loss
    criterion = nn.NLLLoss()

    # Let's optimize the parameters
    optimiser = torch.optim.Adam(model.classifier.parameters(), learning_rate)

    return model, criterion, optimiser

def validation(model, criterion, device, valid_loader, validation_losses):
    # Initialize validation metrics
    validation_loss = 0
    validation_accuracy = 0

    # Switch to evaluation mode, no gradient calculation
    model.eval()

    with torch.no_grad():
        for val_images, val_labels in valid_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)

            # Forward pass for validation
            val_logits = model(val_images)
            val_loss = criterion(val_logits, val_labels)
            val_probs = torch.exp(val_logits)
            top_val_probs, top_val_classes = val_probs.topk(1, dim=1)
            val_matches = (top_val_classes == val_labels.view(*top_val_classes.shape)).type(torch.FloatTensor)
            val_accuracy = val_matches.mean()

            # Track validation metrics
            validation_loss += val_loss.item()
            validation_accuracy += val_accuracy.item()

    # Print validation metrics
    print(f'Validation Loss: {validation_loss / len(valid_loader):.3f}')
    print(f'Validation Accuracy: {validation_accuracy / len(valid_loader) * 100:.2f}%')
    
    # Track training and validation metrics
    validation_losses.append(validation_loss / len(valid_loader))

    return validation_losses

def save_model(model, criterion, optimiser, train_datasets,  arch = 'vgg16', learning_rate = 0.001, hidden_units = 4096, epochs = 10):
    model.class_to_idx = train_datasets.class_to_idx
    model.to('cpu')
    torch.save({
                'criterion': criterion,
                'optimiser':optimiser,
                'classifier': model.classifier,
                'arch': arch,
                'learning_rate':learning_rate,
                'hidden_units':hidden_units,
                'epochs':epochs,
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx},
                'checkpoint.pth')
    print('The model has been successfully trained and saved.')

def loading_model(file_path):
    checkpoint = torch.load(file_path)
    model = models.vgg16(pretrained = True)
        
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    criterion = checkpoint['criterion']
    optimiser = checkpoint['optimiser']
    arch = checkpoint['optimiser']
    learning_rate = checkpoint['learning_rate']
    hidden_units = checkpoint['hidden_units']
    epochs = checkpoint['epochs']
    
    for param in model.parameters(): 
        param.requires_grad = False 
    
    return model, criterion, optimiser, arch, learning_rate, hidden_units, epochs