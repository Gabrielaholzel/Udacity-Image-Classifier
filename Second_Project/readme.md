# Second Project
## Image Classifier
### Imports and Installations
This section begins by importing the necessary Python libraries and modules. Here are some key imports:

* [`torch`][torch] and [`torchvision`][torchvision]: PyTorch is a popular deep learning framework, and torchvision is its library for handling computer vision tasks.
* [`matplotlib.pyplot`][plt]: This library is used for plotting graphs and visualizing images.
* [`PIL.Image`][pil]: The Python Imaging Library (PIL) is used for opening, manipulating, and saving images.
* [`json`][json]: This module is used for handling JSON data.
* [`collections.OrderedDict`][ordereddict]: An ordered dictionary is used for preserving the order of items in dictionaries.
* [`numpy`][numpy]: This module provides support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

```python
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torch.utils.data
import json
import numpy as np
import time
import datetime
from collections import OrderedDict
import seaborn as sns
from PIL import Image
```

### Load the Data
In this section, the notebook explains how to load and preprocess image data. It uses a dataset of flower images, typically used for image classification tasks. Key steps include:

* **Loading and preprocessing the data**: Data augmentation and normalization are applied to the training data, while validation and testing data are only normalized.
```python
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
```

* **Creating data loaders**: PyTorch's DataLoader is used to efficiently load and iterate through the data during training and testing.
```
train_datasets = datasets.ImageFolder(train_dir, transform = train_data_transforms)
valid_datasets = datasets.ImageFolder(valid_dir, transform = valid_data_transforms)
test_datasets = datasets.ImageFolder(test_dir, transform = test_data_transforms)

train_loader = torch.utils.data.DataLoader(train_datasets, batch_size = 64, shuffle = True)
valid_loader = torch.utils.data.DataLoader(valid_datasets, batch_size = 64, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_datasets, batch_size = 64, shuffle = True)
```

### Build and Train the Classifier
This section focuses on building and training a deep neural network classifier using PyTorch. Key steps include:

* **Defining the neural network architecture**: In this example, a pre-trained deep learning model (VGG16) is used as the base architecture.
* **Building the custom classifier**: A custom classifier is added to the pre-trained model to suit the specific classification task (flower classification).
* **Specifying loss and optimizer functions**: The choice of loss function (criterion) and optimizer (e.g., Adam) is explained.
* **Training the classifier**: The training loop is detailed, including forward and backward passes, and model checkpointing is implemented to save the best model weights.

```python
def model_building():
    model = models.vgg16(pretrained=True)

    # Let's freeze the model's params so they stay static
    for params in model.parameters():
        params.requires_grad = False

    # Let's redefine the classifier
    classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(25088, 4096)),
                                ('relu1', nn.ReLU()),
                                ('dropout1', nn.Dropout(p = 0.2)),
                                ('fc2', nn.Linear(4096, 2048)),
                                ('relu2', nn.ReLU()),
                                ('dropout2', nn.Dropout(p = 0.2)),
                                ('fc3', nn.Linear(2048, 1024)),
                                ('relu3', nn.ReLU()),
                                ('fc4', nn.Linear(1024, 102)),
                                ('output', nn.LogSoftmax(dim =1))
                                ]))
    
    model.classifier = classifier

    # Let's move model to gpu
    model = model.to(device)

    # Let's define the criterion as Negative Log Likelihood Loss
    criterion = nn.NLLLoss()

    # Let's optimize the parameters
    optimiser = torch.optim.Adam(model.classifier.parameters(), 0.001)

    return model, criterion, optimiser
```

```python
start = datetime.datetime.now()
model, criterion, optimiser = model_building()
time_elapsed = (datetime.datetime.now() - start).seconds
print(f"Time for building the model: {time_elapsed//60:.0f}m {time_elapsed % 60:.0f}s")
```
> Time for building the model: 0m 16s

To see the code developed for testing and validating the model, please navigate to [this notebook][notebook]. The goal is to reach at least 70% of accuracy. The last few batches were as follows: 
```python
Epoch 10/10 || Batch 60
Training Loss: 0.809
Training Accuracy: 77.81%
Validation Loss: 0.509
Validation Accuracy: 87.18%
Time Training Batch: 0m 18s
-------------------------

Epoch 10/10 || Batch 80
Training Loss: 0.753
Training Accuracy: 80.00%
Validation Loss: 0.472
Validation Accuracy: 89.12%
Time Training Batch: 0m 17s
-------------------------

Epoch 10/10 || Batch 100
Training Loss: 0.795
Training Accuracy: 79.06%
Validation Loss: 0.499
Validation Accuracy: 87.47%
Time Training Batch: 0m 16s
-------------------------
```


### Save and Load the Model
This section explains how to save and load the trained model using PyTorch. The saved model checkpoint contains information about the model architecture, optimizer, and state dict (trained weights).

```python
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
```

```python
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
```

### Inference for Classification
This section is about writing a function to use a trained network for inference. Key steps include:

* **Preprocessing input images**: Similar preprocessing techniques are applied to input images as during training.
```python
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
```
To check the function, a new function was provided. This function converts a PyTorch tensor and displays it in the notebook. If `process_image` function worked, running the output through the provided function should return the original image (except for the cropped out portions).
```python
image_path = valid_dir + '/101/image_07951.jpg'
img = process_image(image_path)
imshow(img)
```

![imshow](https://github.com/Gabrielaholzel/Udacity-Image-Classifier/blob/af95cde5a63daa22cc51b1eb499e2b8a740ee1bb/Second_Project/imshow.png)

```python
process_image(image_path)
processed_image = Image.open(image_path)
processed_image
```
![imshow](https://github.com/Gabrielaholzel/Udacity-Image-Classifier/blob/d9b5666c8ca9c2fce3588c26b47582980fda7eec/Second_Project/process_image.png)

* **Making predictions**: The model is used to predict the class probabilities for a given input image.
This method should takes a path to an image and a model checkpoint, then return the probabilities and classes.

```python
def predict(image_path, model, topk=5):
    # Process the image using the process_image function
    image = process_image(image_path)
    # image = image.transpose((1, 2, 0))
    image_tensor = torch.FloatTensor(image).unsqueeze(0)  # Add batch dimension

    # Set the model to evaluation mode
    model.eval()

    # Use the model to make predictions
    with torch.no_grad():
        output = model(image_tensor)

    # Calculate class probabilities and retrieve top K classes
    probabilities, indices = torch.topk(torch.softmax(output, dim=1), topk)
    probabilities = probabilities.squeeze().numpy()
    indices = indices.squeeze().numpy()

    # Convert indices to class labels using the model's class_to_idx attribute
    idx_to_class = {idx: class_label for class_label, idx in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in indices]

    return probabilities, top_classes
```

* **Mapping class indices to class names**: A mapping between class indices and class names is used to interpret model outputs.

```python
model = uploaded_model 
img = process_image(image_path)
imshow(img)
plt.show()
probs, classes = predict(image_path, model, 5)

class_names = [cat_to_name [item] for item in classes]

plt.figure(figsize = (6,10))
plt.subplot(2,1,2)
sns.barplot(x=probs, y=class_names, color= 'green');
plt.show(
```

As a result of the previous code, we get the following prediction:
![imshow](https://github.com/Gabrielaholzel/Udacity-Image-Classifier/blob/af95cde5a63daa22cc51b1eb499e2b8a740ee1bb/Second_Project/imshow.png)
![imshow](https://github.com/Gabrielaholzel/Udacity-Image-Classifier/blob/d9b5666c8ca9c2fce3588c26b47582980fda7eec/Second_Project/prediction.png)





[//]: ()
[torch]: <https://pytorch.org/>
[torchvision]: <https://pytorch.org/vision/stable/index.html>
[plt]: <https://matplotlib.org/3.5.3/api/pyplot_summary.html>
[pil]: <https://pillow.readthedocs.io/en/latest/reference/Image.html>
[json]: <https://docs.python.org/3/library/json.html>
[ordereddict]: <https://docs.python.org/3/library/collections.html>
[numpy]: <https://numpy.org/>
[notebook]: <https://github.com/Gabrielaholzel/Udacity-Image-Classifier/blob/main/Second_Project/Image%20Classifier%20Project.ipynb>
