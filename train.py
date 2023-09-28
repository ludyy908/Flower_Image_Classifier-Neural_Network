import argparse
import time
import torch


from torch import nn, optim
from torchvision import datasets, models, transforms
from collections import OrderedDict



# Defining command line arguments
def get_command_line_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type = str, default = './flowers', help = 'Images Folder.')
    parser.add_argument('--arch', default = 'vgg16', type = str, help = 'Model architecture.')
    parser.add_argument('--save_dir', type = str, default = './model_checkpoint.pth', help = 'Save trained model to file.')
    parser.add_argument('--epochs', type = int, default = '8', help = 'Number of epochs for training.')
    parser.add_argument('--hidden_units', type = int, default = '2048', help = 'Number of hidden units.')
    parser.add_argument('--learning_rate', type=float, default = '0.003', help = 'Model training learning rate.')
    parser.add_argument('--gpu', default = 'False', help = 'Use GPU if available (True/False).')
 
    return parser.parse_args()

# Loading images
def get_images(directory):
    data_dir = directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    return train_dir, test_dir, valid_dir

# Transforming images
def loading_data(directory):
    data_transforms = {
        'train' : transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomRotation(45),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        ]),
        'test' : transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'validate' : transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    
    #get images folders
    train_dir, test_dir, valid_dir = get_images(directory)
    
    # Load the datasets with ImageFolder
    image_datasets = {
        'train' : datasets.ImageFolder(train_dir, transform = data_transforms['train']),
        'test' : datasets.ImageFolder(test_dir, transform = data_transforms['test']),
        'validate' : datasets.ImageFolder(valid_dir, transform = data_transforms['validate'])
        }

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        x : torch.utils.data.DataLoader(image_datasets[x], batch_size = 64, shuffle = True)
        for x in list(image_datasets.keys())} 
    
    class_to_idx = image_datasets['train'].class_to_idx
    
    return dataloaders, class_to_idx
    
    
 # Building model classifier based on pre-trained model 
def build_model(arch, hidden_units):
     
    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_features = model.classifier.in_features
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_features = model.classifier[0].in_features
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
        input_features = model.classifier[0].in_feature
    else:
        raise TypeError("The arch specified is not supported.")
       
    output_features = 102 
    dropout_prob = 0.2
    
    # Freezing parameters so we don't backpropagate through them 
    for parameters in model.parameters():
        parameters.requires_grad = False

    # Defining a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
    classifier_model = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_features, hidden_units)),
        ('relu1', nn.ReLU(inplace = True)),
        ('dropout1', nn.Dropout(p=dropout_prob)),
        ('fc2', nn.Linear(hidden_units, output_features)),
        ('output', nn.LogSoftmax(dim = 1))
    ]))
    
    print('Network architecture:', arch)

    model.classifier = classifier_model
  
    return model
    
# Training model classifier
def train(model, data, epoch, learning_rate, gpu = True):

    criterion = nn.NLLLoss()
    weight_decay = 1e-4
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Moving the model to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) 
    
    
    print('Number of epochs:', epoch)
    print('Learning rate:', learning_rate)
    
    # Initialize a learning rate scheduler (optional)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    print('Training:')
    time_start = time.time()
    for i in range(epoch):
        steps = 0
        batch = 30
        running_loss = 0
        for inputs, labels in (data['train']):
            inputs, labels = inputs.to(device), labels.to(device)  
            
            #Turning model to train mode
            model.train()
            
            # Clearing gradients
            optimizer.zero_grad()
            
            # Forward pass, then backward pass, then update weights  
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            steps += 1
            
            # Training validation
            if (steps % batch) == 0:
                accuracy, validation_loss = validation(model, data['validate'], criterion, device)
                print ('--------------')
                print('Epoch: {}/{}'.format(i + 1, epoch), 
                      '\tTraining Loss: {:.4f}'.format(running_loss/batch),  #len(train_data)
                      '\tValidation Loss: {}'.format(round(validation_loss, 4)),
                      '\tValidation Accuracy: {}'.format(round(accuracy, 4))
                     )
                running_loss = 0
        # Optionally, step the learning rate scheduler
        scheduler.step()
          
    time_end = time.time()-time_start
    print("Model Training Coomplete.")
    
    print("\nTotal Elapsed Runtime:",
          str(int((time_end/3600)))+":"+str(int((time_end%3600)/60))+":"
          +str(int((time_end%3600)%60)) )
          
    validation_test(model, data['test'], criterion, device)
          
    return model, optimizer, criterion
          

def validation(model, valid_data, criterion, device):
    correct = 0
    total = 0
    validation_loss = 0
    accuracy = 0
    
    # Setting the model in evaluation mode
    model.eval()
    
    with torch.no_grad():
        for inputs, labels in valid_data:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            
            # Calculating the loss
            batch_loss = criterion(outputs, labels)  
            
            # Accumulating the loss
            validation_loss += batch_loss.item()  
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total  # Calculating accuracy
    validation_loss /= len(valid_data)  # Calculating the average loss
    
    return accuracy, validation_loss

    
    
# Validation on the test set
def validation_test(model, test_data, criterion, device):
    
    correct = 0
    total = 0
    accuracy = 0
    test_loss = 0
    
    print('\nTesting:')
    # Setting the model in evaluation mode
    model.eval()
    
    with torch.no_grad():
        for inputs, labels in test_data:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            
            # Calculating the loss
            batch_loss = criterion(outputs, labels)  
            test_loss += batch_loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            # Calculating the accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Calculating accuracy and average test loss
    accuracy = correct / total  
    average_test_loss = test_loss / len(test_data)
    
    print('Test Loss: {:.3f}'.format(average_test_loss),
          '\nAccuracy: {:.2%}'.format(accuracy)
         )
          
# Saving model
def save_model(model, arch, checkpoint_dir, optimizer, class_to_idx, ):
    model.class_to_idx = class_to_idx
    checkpoint = {'classifier': model.classifier,
              'state_dict': model.state_dict(),
              'input_size': model.classifier[0].in_features,
              'output_size': model.classifier[3].out_features,
              'structure': arch,
              'optimizer': optimizer.state_dict(),
              'class_to_idx': class_to_idx
             }

    torch.save(checkpoint, checkpoint_dir)
    print("Checkpoint saved as {}.".format(checkpoint_dir))
          
     
def main():
    args = get_command_line_args()
    
    print("You Are Now Creating a Deep Learning Model.\n")
    data_loaders, class_to_idx = loading_data(args.dir)
    model = build_model(args.arch, args.hidden_units)
    model, optimizer, criterion = train(model, data_loaders, args.epochs, args.learning_rate, args.gpu)
    save_model(model, args.arch, args.save_dir, optimizer, class_to_idx)
    
    print("\nAll Done.")
          
          
    
if __name__ == "__main__":
    main()
          
          
             
          