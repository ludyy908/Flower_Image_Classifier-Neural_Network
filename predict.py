import argparse
import numpy as np
import torch
import json

from collections import OrderedDict
 
from torch import nn
from torchvision import models


from PIL import Image





def get_command_line_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image', type=str, help='path to the image to predict')
    parser.add_argument('--topk', type=int, default = 5, help='Top classes to return')
    parser.add_argument('--checkpoint', type=str, default = './model_checkpoint.pth', help='Saved Checkpoint') 
    parser.add_argument('--gpu', default='False', action='store_true', help='Where to use gpu or cpu')
    parser.add_argument('--labels', type=str, default='./cat_to_name.json', help='file for label names')
    parser.add_argument('--arch', type=str, default='vgg16', help='chosen model')

    return parser.parse_args()

    
def load_checkpoint(filepath, arch):
        
    checkpoint = torch.load(filepath)
    model =  build_model(arch)

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    print("Loaded checkpoint with arch {}".format(checkpoint['structure']))
    return model

def build_model(arch):
     
    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    else:
        raise TypeError("The arch specified is not supported.")
    
    # Freezing parameters so we don't backpropagate through them 
    for parameters in model.parameters():
        parameters.requires_grad = False
  
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    image = Image.open(image)
    
    # Getting the size of the image
    width, height = image.size
    
    
    # Determining the aspect ratio of the image
    aspect_ratio = width / height

    # Calculating the new dimensions based on the shortest side (256)
    if width < height:
        new_width = 256
        new_height = int(256 / aspect_ratio)
    else:
        new_width = int(256 * aspect_ratio)
        new_height = 256

    # Resizing the image while preserving the aspect ratio
    image = image.resize((new_width, new_height), Image.ANTIALIAS)
    
   
    # Defining the size of the crop (224x224)
    crop_size = 224

    # Calculating the coordinates to crop the center portion
    left = (width - crop_size) / 2
    top = (height - crop_size) / 2
    right = (width + crop_size) / 2
    bottom = (height + crop_size) / 2

    # Cropping the center portion
    cropped_image = image.crop((left, top, right, bottom))
    
    # Convertting colour channel from 0-255 to 0-1
    np_image = np.array(cropped_image)/255
    
    # Normalizing the color channels
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalized_image = (np_image - mean) / std

    # Reordering the dimensions to have color channels as the first dimension
    normalized_image = normalized_image.transpose((2, 0, 1))

    # Converting the NumPy array to a PyTorch tensor
    tensor = torch.tensor(normalized_image, dtype=torch.float)
    
    return tensor
    

def predict(image_path, model, cat_to_name, gpu, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Processing image
    image = process_image(image_path)
    image = image.unsqueeze(0)

    # Setting available device
    if gpu and torch.cuda.is_available():
        device = torch.device("cuda")    
        model.to(device)
        image = image.to(device)
    else:
        device = torch.device("cpu")    
        model.to(device)
        image = image.to(device)
    
    model.eval()
    
    with torch.no_grad():
        ps = torch.exp(model(image))
   
    flower_cat = {cat: cat_to_name[i] for i, cat in model.class_to_idx.items()}    
    
    top_ps, top_classes = ps.topk(topk, dim=1)
    predicted = [flower_cat[i] for i in top_classes.tolist()[0]]

    return top_ps.tolist()[0], predicted
               
               
   
               
def classify_image(image, model, gpu, top_k, cat_to_name):
    with open(cat_to_name, 'r') as f:
        cat_to_name = json.load(f)
   
    # Getting actual image label
    cat_idx = image.split('/')[-2]
    flower_name = cat_to_name[cat_idx]
    
    # Predicting image
    top_ps, top_classes = predict(image, model, cat_to_name, gpu, top_k)
    
    top_flower = top_classes[0]


    if top_flower == flower_name:
        classification = "Correctly Classified" 
    elif flower_name in top_classes:
        classification = "In Top 5 Prediction" 
    else: 
        classification = "Incorrectly Classified" 

    print('\n--Top Predictions:--\n')

    for idx, (class_name, probability) in enumerate(zip(top_classes, top_ps), start=1):
        print(f"{idx}. Class: {class_name}\t\tProbability: {probability:.4f}")
               
    print("\nActual Class = {} \t\t Classification = {}".format(flower_name, classification))
               
              
               
def main():
    args = get_command_line_args()
    
    if (args.gpu and not torch.cuda.is_available()):
        raise Exception("--gpu option enabled, but no GPU detected.")
    
    top_k = args.topk
    image_path = args.image
          
    model = load_checkpoint(args.checkpoint, args.arch)
    classify_image(image_path, model, args.gpu, top_k, args.labels)
    
main()
    