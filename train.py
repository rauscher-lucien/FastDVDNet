import os
import torch
import matplotlib.pyplot as plt
import pickle
import time
import logging

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


from utils import *
from transforms import *
from dataset import *
from model import *


class Trainer:

    def __init__(self, data_dict):

        self.train_data_dir = data_dict['train_data_dir']
        self.val_data_dir = data_dict['val_data_dir']
        self.project_dir = data_dict['project_dir']
        self.project_name = data_dict['project_name']

        self.results_dir, self.checkpoints_dir = create_result_dir(self.project_dir, self.project_name)
        self.train_results_dir, self.val_results_dir = create_train_val_dir(self.results_dir)

        self.num_epoch = data_dict['num_epoch']
        self.batch_size = data_dict['batch_size']
        self.lr = data_dict['lr']

        self.num_freq_disp = data_dict['num_freq_disp']
        self.num_freq_save = data_dict['num_freq_save']

        self.train_continue = data_dict['train_continue']
        self.load_epoch = data_dict['load_epoch']

        self.device = get_device()

        self.writer = SummaryWriter(self.results_dir + '/tensorboard_logs')



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

        print(self.train_data_dir)
        start_time = time.time()
        mean, std = compute_global_mean_and_std(self.train_data_dir, self.checkpoints_dir)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")

        transform_train = transforms.Compose([
            Normalize(mean, std),
            RandomCropVideo(output_size=(64,64)),
            ToTensorVideo()
        ])

        transform_inv_train = transforms.Compose([
            ToNumpyVideo()
        ])


        ### make dataset and loader ###

        ## prepare dataset
        crop_tiff_depth_to_divisible(self.train_data_dir, self.batch_size)
        crop_tiff_depth_to_divisible(self.val_data_dir, self.batch_size)

        train_dataset = DatasetLoadAll(root_folder_path=self.train_data_dir,
                                    transform=transform_train)
        
        val_dataset = DatasetLoadAll(root_folder_path=self.val_data_dir,
                                    transform=transform_train)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2)
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2)

        num_train = len(train_dataset)
        num_batch_train = int((num_train / self.batch_size) + ((num_train % self.batch_size) != 0))


        ### initialize network ###

        model = FastDVDnet().to(self.device)

        criterion = nn.MSELoss(reduction='sum').to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), self.lr)

        st_epoch = 0
        if self.train_continue == 'on':
            print(self.checkpoints_dir)
            model, optimizer, st_epoch = self.load(self.checkpoints_dir, model, self.load_epoch, optimizer)

            model = model.to(self.device)

        for epoch in range(st_epoch + 1, self.num_epoch + 1):

            for batch, data in enumerate(train_loader, 0):

                def should(freq):
                    return freq > 0 and (batch % freq == 0 or batch == num_batch_train)

                # Pre-training step
                model.train()

                # When optimizer = optim.Optimizer(net.parameters()) we only zero the optim's grads
                optimizer.zero_grad()

                input_stack, target_img = [x.squeeze(0).to(self.device) for x in data]

                input_stack = input_stack.to(self.device)
                target_img = target_img.to(self.device)

                output_img = model(input_stack)

                loss = criterion(output_img, target_img)
                self.writer.add_scalar('Loss/train', loss.item(), epoch * num_batch_train + batch)
                loss.backward()
                optimizer.step()

                logging.info('TRAIN: EPOCH %d: BATCH %04d/%04d: LOSS: %.4f'
                             % (epoch, batch, num_batch_train, loss))
                

                if should(self.num_freq_disp):

                    input_img_np = transform_inv_train(input_stack)
                    target_img_np = transform_inv_train(target_img)
                    output_img_np = transform_inv_train(output_img)

                    num_frames = input_stack.shape[-1]

                    for j in range(target_img_np.shape[0]):
                        base_filename = f"sample{j:03d}"

                        for frame_idx in range(num_frames):
                            input_frame_filename = os.path.join(self.train_results_dir, f"{base_filename}_input_frame{frame_idx}.png")
                            plt.imsave(input_frame_filename, input_img_np[j, :, :, 0, frame_idx], cmap='gray')

                        target_filename = os.path.join(self.train_results_dir, f"{base_filename}_target.png")
                        output_filename = os.path.join(self.train_results_dir, f"{base_filename}_output.png")

                        plt.imsave(target_filename, target_img_np[j, :, :, 0], cmap='gray')
                        plt.imsave(output_filename, output_img_np[j, :, :, 0], cmap='gray')


            model.eval()  # Set model to evaluation mode
            val_loss = 0.0
            with torch.no_grad():  # Disable gradient computation
                for data in val_loader:
                    input_stack, target_img = [x.squeeze(0).to(self.device) for x in data]
                    output_img = model(input_stack)
                    loss = criterion(output_img, target_img)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            self.writer.add_scalar('Loss/val', avg_val_loss, epoch)
            print(f'Validation Loss: {avg_val_loss:.4f}')


            if (epoch % self.num_freq_save) == 0:
                self.save(self.checkpoints_dir, model, optimizer, epoch)
 
        self.writer.close()