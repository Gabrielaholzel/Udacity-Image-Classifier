import torch
from torchvision import datasets, transforms
import torch.utils.data
from PIL import Image

def load_data(data_directory):
    train_data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])
                                                ])

    valid_data_transforms = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])
                                                ])

    test_data_transforms = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])
                                                ])

    train_datasets = datasets.ImageFolder(data_directory + '/train', transform = train_data_transforms)
    valid_datasets = datasets.ImageFolder(data_directory + '/valid', transform = valid_data_transforms)
    test_datasets = datasets.ImageFolder(data_directory + '/test', transform = test_data_transforms)

    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size = 64, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(valid_datasets, batch_size = 64, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size = 64, shuffle = True)

    return train_loader, valid_loader, test_loader, train_datasets

def process_image(image_path):
    # Open the image using PIL
    pil_image = Image.open(image_path).convert('RGB')

    # Define transformations
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Apply the transformations to the image
    tensor_image = preprocess(pil_image)

    # Convert the tensor to a NumPy array and transpose dimensions
    np_image = tensor_image.numpy()

    return np_image