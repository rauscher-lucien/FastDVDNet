import os
import sys
sys.path.append(os.path.join(".."))

import torch
import numpy as np
import tifffile
import argparse
import glob
from torchvision import transforms
import matplotlib.pyplot as plt

from model import *
from transforms import *
from utils import *
from dataset import *


def load(checkpoints_dir, model, epoch, optimizer=None, device='cpu'):
    """
    Load the model and optimizer states.

    :param dir_chck: Directory where checkpoint files are stored.
    :param netG: The Generator model (or any PyTorch model).
    :param epoch: Epoch number to load.
    :param optimG: The optimizer for the Generator model.
    :param device: The device ('cpu' or 'cuda') to load the model onto.
    :return: The model, optimizer, and epoch, all appropriately loaded to the specified device.
    """

    # Ensure optimG is not None; it's better to explicitly check rather than using a mutable default argument like []
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())  # Or whatever default you prefer

    checkpoint_path = os.path.join(checkpoints_dir, f'model_epoch{epoch:04d}.pth')
    dict_net = torch.load(checkpoint_path, map_location=device)

    print(f'Loaded {epoch}th network')

    model.load_state_dict(dict_net['netG'])
    # Ensure the optimizer state is also loaded to the correct device
    optimizer.load_state_dict(dict_net['optimG'])

    # If the model and optimizer are expected to be used on a GPU, explicitly move them after loading.
    model.to(device)
    # Note: Optimizers will automatically move their tensors to the device of the parameters they optimize.
    # So, as long as the model parameters are correctly placed, the optimizer's tensors will be as well.

    return model, optimizer, epoch


def main():


        # Check if the script is running on the server by looking for the environment variable
    if os.getenv('RUNNING_ON_SERVER') == 'true':

        parser = argparse.ArgumentParser(description='Process data directory.')

        parser.add_argument('--data_dir', type=str, help='Path to the data directory')
        parser.add_argument('--project_name', type=str, help='Name of the project')
        parser.add_argument('--inference_name', type=str, help='Name of the inference')
        parser.add_argument('--load_epoch', type=int, default=1, 
                            help='Epoch number from which to continue training (default: 1)')
    

        # Parse arguments
        args = parser.parse_args()

        # Now you can use args.data_dir as the path to your data
        data_dir = args.data_dir
        project_name = args.project_name 
        inference_name = args.inference_name 
        load_epoch = args.load_epoch
        project_dir = os.path.join('/g', 'prevedel', 'members', 'Rauscher', 'projects', 'FastDVDNet')

        print(f"Using data directory: {data_dir}")
        print(f"Project name: {project_name}")
        print(f"Load epoch: {load_epoch}")
    else:
        # If not running on the server, perhaps use a default data_dir or handle differently
        data_dir = os.path.join('C:\\', 'Users', 'rausc', 'Documents', 'EMBL', 'data', 'only_two_dataset_not_similar', 'good_sample_unidentified')
        project_dir = os.path.join('C:\\', 'Users', 'rausc', 'Documents', 'EMBL', 'projects', 'FastDVDNet')
        project_name = 'only_two_dataset_not_similar-test_1'
        inference_name = 'inference_300-good_sample_unidentified'
        load_epoch = 300

    #********************************************************#


    #********************************************************#

    results_dir = os.path.join(project_dir, project_name, 'results')
    checkpoints_dir = os.path.join(project_dir, project_name, 'checkpoints')

    # Make a folder to store the inference
    inference_folder = os.path.join(results_dir, inference_name)
    os.makedirs(inference_folder, exist_ok=True)



    # check if GPU is accessible
    if torch.cuda.is_available():
        print("\nGPU will be used.")
        device = torch.device("cuda:0")
    else:
        print("\nCPU will be used.")
        device = torch.device("cpu")

    mean, std = load_normalization_params(checkpoints_dir)
    
    inf_transform = transforms.Compose([
        Normalize(mean, std),
        CropToMultipleOf16Video(),
        ToTensorVideo(),
    ])

    inv_inf_transform = transforms.Compose([
        BackTo01Range(),
        ToNumpyVideo()
    ])

    inf_dataset = N2NVideoDataset(
        data_dir,
        transform=inf_transform
    )

    batch_size = 8
    print("Dataset size:", len(inf_dataset))
    inf_loader = torch.utils.data.DataLoader(
        inf_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )


    model = FastDVDnet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model, optimizer, _ = load(checkpoints_dir, model, load_epoch, optimizer)

    model = model.to(device)


    print("starting inference")

    with torch.no_grad():
        model.eval()

        # Initialize list to store numpy arrays for output images
        output_images = []

        for batch, data in enumerate(inf_loader):
            input_stack = data[0].to(device)  # Assuming data is a tensor with shape [B, 4, H, W]

            # Generate the output images
            output_img = model(input_stack)
            output_img_np = inv_inf_transform(output_img)  # Convert output to numpy, adjust shape
            # Assuming output_img_np is of shape [B, H, W, 1] for single output image
            for img in output_img_np:
                output_images.append(img)

            print(f'BATCH {batch+1}/{len(inf_loader)}')

    # Clip output images to the 0-1 range
    output_images_clipped = [np.clip(img, 0, 1) for img in output_images]

    # Stack and save output images
    output_stack = np.stack(output_images_clipped, axis=0).squeeze(-1)  # Remove channel dimension if single channel
    filename = f'output_stack-{project_name}-{inference_name}.TIFF'
    tifffile.imwrite(os.path.join(inference_folder, filename), output_stack)

    print("Output TIFF stack created successfully.")


if __name__ == '__main__':
    main()


