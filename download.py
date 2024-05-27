import yaml
import os
import requests
import tarfile
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config
 


image_size = (224, 224)

def load_and_split_data(data_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(image_size),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the train dataset
    trainval_dataset =datasets.OxfordIIITPet(root=data_dir, split="trainval", download=True, transform=transform)

    # Load the test dataset
    test_dataset =datasets.OxfordIIITPet(root=data_dir, split = "test", download=True, transform=transform)
    
    return trainval_dataset, test_dataset



if __name__ == "__main__":
    config_file = "config.yaml"
    config = load_config(config_file)
    destination_folder = config['data']['local_dir']
    trainval_dataset, test_dataset = load_and_split_data(destination_folder)

    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Save the split datasets to pickle files
    import pickle
    with open('data/trainval_dataset.pkl', 'wb') as f:
        pickle.dump(trainval_dataset, f)
    with open('data/test_dataset.pkl', 'wb') as f:
        pickle.dump(test_dataset, f)
