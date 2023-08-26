import argparse
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import torch.utils.data
import numpy as np
import json
import utility_functions as uf
import model_functions as mf

parser = argparse.ArgumentParser(
    description = 'predict.py parser'
)

# Getting the arguments
parser.add_argument('input', action="store", type = str, default='flowers/valid/101/image_07951.jpg')
parser.add_argument('checkpoint', action="store", type = str, default='./checkpoint.pth')
parser.add_argument('--top_k', action="store", type=int, default=1)
parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
parser.add_argument('--gpu', action="store_true", default="gpu")

# Parsing the arguments
args = parser.parse_args()
image_path = args.input
checkpoint_file = args.checkpoint
top_k = args.top_k
category_names = args.category_names
gpu = args.gpu

# Selecting device, topk_num, json_file
device = ['cuda' if gpu else 'cpu'][0]
topk_num = [top_k if top_k else 1][0]
json_file = [category_names if category_names else 'cat_to_name.json'][0]



def main():
    model, _, _, _, _, _, _ = mf.loading_model('checkpoint.pth')
    model


    with open(json_file, 'r') as f:
        cat_to_name = json.load(f)

    np_image = uf.process_image(image_path)
    image_tensor = torch.FloatTensor(np_image).unsqueeze(0)

    # Set the model to evaluation mode
    model.eval()

    # Use the model to make predictions
    with torch.no_grad():
        output = model(image_tensor)

    # Calculate class probabilities and retrieve top K classes
    probabilities, indices = torch.topk(torch.softmax(output, dim=1), top_k)
    probabilities = probabilities.squeeze().numpy()
    indices = indices.squeeze().numpy()

    # Convert indices to class labels using the model's class_to_idx attribute
    idx_to_class = {idx: class_label for class_label, idx in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in indices]
    mapping = {val: key for key, val in model.class_to_idx.items()}
    classes = [mapping[item] for item in indices]
    classes = np.array(classes)

    class_names = [cat_to_name[item] for item in classes]
    
    print(f'The top {topk_num} classes are:')
    for prob, class_name in zip(probabilities,class_names):
        print(f'{class_name} with a probability of {prob:.2f}%')

if __name__== "__main__":
    main()