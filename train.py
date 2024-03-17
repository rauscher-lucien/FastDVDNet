import os
import torch
import matplotlib.pyplot as plt
import pickle
import time
import logging

from torchvision import transforms

from utils import *
from transforms import *
from dataset import *
from model import *


class Trainer:

    def __init__(self, data_dict):

        self.results_dir = data_dict['results_dir']

        self.train_results_dir = os.path.join(self.results_dir, 'train')
        os.makedirs(self.train_results_dir, exist_ok=True)

        self.checkpoints_dir = data_dict['checkpoints_dir']

        self.data_dir = data_dict['data_dir']

        self.num_epoch = data_dict['num_epoch']
        self.batch_size = data_dict['batch_size']

        self.num_freq_disp = data_dict['num_freq_disp']
        self.num_freq_save = data_dict['num_freq_save']

        self.train_continue = data_dict['train_continue']
        self.load_epoch = data_dict['load_epoch']

        # check if we have a gpu
        if torch.cuda.is_available():
            print("GPU is available")
            self.device = torch.device("cuda:0")
        else:
            print("GPU is not available")
            self.device = torch.device("cpu")


    def save(self, dir_chck, net, optimG, epoch):
        if not os.path.exists(dir_chck):
            os.makedirs(dir_chck)

        torch.save({'netG': net.state_dict(),
                    'optimG': optimG.state_dict()},
                   '%s/model_epoch%04d.pth' % (dir_chck, epoch))
        

    def load(self, dir_chck, net, epoch, optimG=[]):

        dict_net = torch.load('%s/model_epoch%04d.pth' % (dir_chck, epoch))

        print('Loaded %dth network' % epoch)

        net.load_state_dict(dict_net['netG'])
        optimG.load_state_dict(dict_net['optimG'])

        return net, optimG, epoch
    

    def train(self):

        ### transforms ###

        print(self.data_dir)
        start_time = time.time()
        min, max = compute_global_min_max_and_save(self.data_dir)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")

        transform_train = transforms.Compose([
            MinMaxNormalize(min, max),
            RandomCrop(output_size=(64,64)),
            ToTensor()
        ])

        transform_inv_train = transforms.Compose([
            ToNumpy()
        ])


        ### make dataset and loader ###

        dataset_train = N2NSliceDataset2(root_folder_path=self.data_dir, 
                                    transform=transform_train)

        loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2)

        num_train = len(dataset_train)
        num_batch_train = int((num_train / self.batch_size) + ((num_train % self.batch_size) != 0))


        ### initialize network ###

        net = UNet(nch_in = 1,
                   nch_out = 1
                   ).to(self.device)

        # init_weights(net, init_type='kaiming', init_gain=1e-20)
        init_weights(net, init_type='normal', init_gain=0.02)

        N2N_loss = nn.L1Loss().to(self.device).to(self.device)

        params = net.parameters()

        optimG = torch.optim.Adam(params, lr=1e-3, betas=(0.5, 0.999))

        st_epoch = 0
        if self.train_continue == 'on':
            print(self.checkpoints_dir)
            net, optimG, st_epoch = self.load(self.checkpoints_dir, net, self.load_epoch, optimG)


        for epoch in range(st_epoch + 1, self.num_epoch + 1):

            net.train()  # Set the network to training mode

            loss_train = []  # Initialize a list to store losses for each batch

            for batch, data in enumerate(loader_train, 1):
                
                def should(freq):
                    return freq > 0 and (batch % freq == 0 or batch == num_batch_train)

                # Assuming 'data' is a tuple of (input_batch, target_batch)
                # and each element in 'data' has a shape of [1, 8, 1, 64, 64] due to batch_size=1 in DataLoader
                # Squeeze the unnecessary outer batch dimension
                input_img, target_img = [x.squeeze(0).to(self.device) for x in data]

                # Forward pass: compute the network output on input_img
                output_img = net(input_img)

                # Reset gradients for the current batch
                optimG.zero_grad()

                # Compute loss using the output from the network and the target_img
                loss = N2N_loss(output_img, target_img)

                # Backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()

                # Perform a single optimization step (parameter update)
                optimG.step()

                # Store the loss for this batch
                loss_train.append(loss.item())  # .item() converts a single-element tensor to a scalar


                logging.info('TRAIN: EPOCH %d: BATCH %04d/%04d: LOSS: %.4f'
                             % (epoch, batch, num_batch_train, np.mean(loss_train)))

                
                if should(self.num_freq_disp):

                    input_img = transform_inv_train(input_img)[..., 0]
                    target_img = transform_inv_train(target_img)[..., 0]
                    output_img = transform_inv_train(output_img)[..., 0]

                    # plot_intensity_distribution(input_img)
                    # plot_intensity_distribution(target_img)
                    # plot_intensity_distribution(output_img)

                    # input_img = np.clip(input_img, 0, 1)
                    # target_img = np.clip(target_img, 0, 1)
                    # output_img = np.clip(output_img, 0, 1)

                    for j in range(target_img.shape[0]):

                        name = num_batch_train * (batch - 1) + j
                        fileset = {'name': name,
                                   'input': "%04d-%04d-input.png" % (batch, j),
                                   'output': "%04d-%04d-output.png" % (batch, j),
                                   'target': "%04d-%04d-label.png" % (batch, j)}


                        plt.imsave(os.path.join(self.train_results_dir, fileset['input']), input_img[j, :, :], cmap='gray')
                        plt.imsave(os.path.join(self.train_results_dir, fileset['output']), output_img[j, :, :], cmap='gray')
                        plt.imsave(os.path.join(self.train_results_dir, fileset['target']), target_img[j, :, :], cmap='gray')
            
            if (epoch % self.num_freq_save) == 0:
                self.save(self.checkpoints_dir, net, optimG, epoch)