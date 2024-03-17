import os
import sys
sys.path.append(os.path.join(".."))

import torch
import numpy as np
import tifffile
import glob
from torchvision import transforms
import matplotlib.pyplot as plt

from new_model_3 import *
from transforms import *
from utils import *
from dataset import *


def load(dir_chck, netG, epoch, optimG=[]):

    dict_net = torch.load('%s/model_epoch%04d.pth' % (dir_chck, epoch))

    print('Loaded %dth network' % epoch)

    netG.load_state_dict(dict_net['netG'])
    optimG.load_state_dict(dict_net['optimG'])

    return netG, optimG, epoch

def main():

    #********************************************************#

    # project_dir = os.path.join('Z:\\', 'members', 'Rauscher', 'projects', 'OCM_denoising-n2n_training')
    project_dir = os.path.join('C:\\', 'Users', 'rausc', 'Documents', 'EMBL', 'projects', 'OCM_denoising-FastDVDNet')
    data_dir = os.path.join('C:\\', 'Users', 'rausc', 'Documents', 'EMBL', 'data', 'test_data_2')
    name = 'test-new_pooling-sigmoid'
    inference_name = 'inference_50'
    load_epoch = 50


    #********************************************************#

    results_dir = os.path.join(project_dir, name, 'results')
    checkpoints_dir = os.path.join(project_dir, name, 'checkpoints')

    # Make a folder to store the inference
    inference_folder = os.path.join(results_dir, inference_name)
    os.makedirs(inference_folder, exist_ok=True)
    
    ## Load image stack for inference
    filenames = glob.glob(os.path.join(data_dir, "*.TIFF"))
    print("Following file will be denoised:  ", filenames[0])



    # check if GPU is accessible
    if torch.cuda.is_available():
        print("\nGPU will be used.")
        device = torch.device("cuda:0")
    else:
        print("\nCPU will be used.")
        device = torch.device("cpu")

    min, max = load_min_max_params(data_dir=data_dir)
    
    inf_transform = transforms.Compose([
        MinMaxNormalizeVideo(min, max),
        CropToMultipleOf16Video(),
        ToTensorVideo(),
    ])

    inv_inf_transform = transforms.Compose([
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


    model = FastDVDnet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model, optimizer, st_epoch = load(checkpoints_dir, model, load_epoch, optimizer)


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
    tifffile.imwrite(os.path.join(inference_folder, 'output_stack.TIFF'), output_stack)

    print("Output TIFF stack created successfully.")


if __name__ == '__main__':
    main()


