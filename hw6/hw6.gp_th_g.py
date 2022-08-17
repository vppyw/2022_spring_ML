# import module
import os
import glob
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torch import autograd
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np
import logging
from tqdm import tqdm


# seed setting
def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(0x6666)
workspace_dir = '.'

# prepare for CrypkoDataset

class CrypkoDataset(Dataset):
    def __init__(self, fnames, transform):
        self.transform = transform
        self.fnames = fnames
        self.num_samples = len(self.fnames)

    def __getitem__(self,idx):
        fname = self.fnames[idx]
        img = torchvision.io.read_image(fname)
        img = self.transform(img)
        return img

    def __len__(self):
        return self.num_samples

def get_dataset(root):
    fnames = glob.glob(os.path.join(root, '*'))
    compose = [
        transforms.ToPILImage(),
        transforms.Resize((96, 96)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
    transform = transforms.Compose(compose)
    dataset = CrypkoDataset(fnames, transform)
    return dataset

# Generator

class Generator(nn.Module):
    """
    Input shape: (batch, in_dim)
    Output shape: (batch, 3, 64, 64)
    """
    def __init__(self, in_dim, feature_dim=96):
        super().__init__()

        #input: (batch, z_dim)
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, feature_dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(feature_dim * 8 * 4 * 4),
            nn.ReLU()
        )
        self.l2 = nn.Sequential(
            self.dconv_bn_relu(feature_dim * 8, feature_dim * 4),               #(batch, feature_dim * 16, 8, 8)
            self.dconv_bn_relu(feature_dim * 4, feature_dim * 2),               #(batch, feature_dim * 16, 16, 16)
            self.dconv_bn_relu(feature_dim * 2, feature_dim),                   #(batch, feature_dim * 16, 32, 32)
        )
        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, 3, kernel_size=5, stride=2,
                               padding=2, output_padding=1, bias=False),
            nn.Tanh()
        )
        self.apply(weights_init)
    def dconv_bn_relu(self, in_dim, out_dim):
        return nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=5, stride=2,
                               padding=2, output_padding=1, bias=False),        #double height and width
            nn.BatchNorm2d(out_dim),
            nn.ReLU(True)
        )
    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2(y)
        y = self.l3(y)
        return y

# Discriminator
class Discriminator(nn.Module):
    """
    Input shape: (batch, 3, 64, 64)
    Output shape: (batch)
    """
    def __init__(self, in_dim, feature_dim=96):
        super(Discriminator, self).__init__()

        #input: (batch, 3, 64, 64)
        """
        NOTE FOR SETTING DISCRIMINATOR:

        Remove last sigmoid layer for WGAN
        """
        self.l1 = nn.Sequential(
            nn.Conv2d(in_dim, feature_dim, kernel_size=4, stride=2, padding=1), #(batch, 3, 32, 32)
            nn.LeakyReLU(0.2),
            self.conv_bn_lrelu(feature_dim, feature_dim * 2),                   #(batch, 3, 16, 16)
            self.conv_bn_lrelu(feature_dim * 2, feature_dim * 4),               #(batch, 3, 8, 8)
            self.conv_bn_lrelu(feature_dim * 4, feature_dim * 8),               #(batch, 3, 4, 4)
            nn.Conv2d(feature_dim * 8, 1, kernel_size=4, stride=1, padding=0),
            # nn.Sigmoid()
            nn.Dropout(0.1),
        )
        self.apply(weights_init)
    def conv_bn_lrelu(self, in_dim, out_dim):
        """
        NOTE FOR SETTING DISCRIMINATOR:

        You can't use nn.Batchnorm for WGAN-GP
        Use nn.InstanceNorm2d instead
        """

        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 4, 2, 1),
            nn.InstanceNorm2d(out_dim, affine = True),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
        )
    def forward(self, x):
        y = self.l1(x)
        y = y.view(-1)
        return y

# setting for weight init function
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class TrainerGAN():
    def __init__(self, config):
        self.config = config

        self.G = Generator(self.config["z_dim"])
        self.D = Discriminator(3)

        self.loss = nn.BCELoss()

        """
        NOTE FOR SETTING OPTIMIZER:

        GAN: use Adam optimizer
        WGAN: use RMSprop optimizer
        WGAN-GP: use Adam optimizer
        """
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=self.config["lr"], betas=(0.5, 0.999))
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=self.config["lr"], betas=(0.5, 0.999))

        self.dataloader = None
        self.log_dir = os.path.join(self.config["workspace_dir"], 'logs')
        self.ckpt_dir = os.path.join(self.config["workspace_dir"], 'checkpoints')
        self.last_G_cktp = ''
        self.last_D_cktp = ''
        self.output_dir = os.path.join(self.config["workspace_dir"], 'output')

        FORMAT = '%(asctime)s - %(levelname)s: %(message)s'
        logging.basicConfig(level=logging.INFO,
                            format=FORMAT,
                            datefmt='%Y-%m-%d %H:%M')

        self.steps = 0
        self.z_samples = Variable(torch.randn(100, self.config["z_dim"])).to(self.config["device"])

    def prepare_environment(self):
        """
        Use this funciton to prepare function
        """
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        # update dir by time
        time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.log_dir = os.path.join(self.log_dir, time+f'_{self.config["model_type"]}')
        self.ckpt_dir = os.path.join(self.ckpt_dir, time+f'_{self.config["model_type"]}')
        self.output_dir = os.path.join(self.output_dir, time+f'_{self.config["model_type"]}')
        os.makedirs(self.log_dir)
        os.makedirs(self.ckpt_dir)
        os.makedirs(self.output_dir)

        # create dataset by the above function
        dataset = get_dataset(os.path.join(self.config["workspace_dir"], 'faces'))
        # dataset += get_dataset(os.path.join(self.config["workspace_dir"], 'faces2'))
        self.dataloader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True, num_workers=2)

        # model preparation
        self.G = self.G.to(self.config["device"])
        self.D = self.D.to(self.config["device"])
        self.G.train()
        self.D.train()

    def gp(self, r_imgs, f_imgs):
        """
        Implement gradient penalty function
        """
        eta = torch.FloatTensor(r_imgs.size(0),1,1,1).uniform_(0,1)
        eta = eta.expand(*r_imgs.shape).to(self.config['device'])
        inter = eta * r_imgs + ((1 - eta) * f_imgs).to(self.config['device'])
        inter = Variable(inter, requires_grad = True)
        prob_inter = self.D(inter)
        gd = autograd.grad(outputs = prob_inter, inputs = inter,
                            grad_outputs = torch.ones(prob_inter.size()).to(self.config['device']),
                            create_graph = True, retain_graph = True)[0]
        return ((gd.norm(2, dim=1) - 1) ** 2).mean() * self.config['gp_lambda']

    def train(self):
        """
        Use this function to train generator and discriminator
        """
        self.prepare_environment()

        for e, epoch in enumerate(range(self.config["n_epoch"])):
            progress_bar = tqdm(self.dataloader)
            progress_bar.set_description(f"Epoch {e+1}")
            for i, data in enumerate(progress_bar):
                imgs = data.to(self.config["device"])
                bs = imgs.size(0)

                # *********************
                # *    Train D        *
                # *********************
                z = Variable(torch.randn(bs, self.config["z_dim"])).to(self.config["device"])
                r_imgs = Variable(imgs).to(self.config["device"])
                f_imgs = self.G(z)
                r_label = torch.ones((bs)).to(self.config["device"])
                f_label = torch.zeros((bs)).to(self.config["device"])


                # Discriminator forwarding
                r_logit = self.D(r_imgs)
                f_logit = self.D(f_imgs)

                """
                NOTE FOR SETTING DISCRIMINATOR LOSS:

                GAN:
                    loss_D = (r_loss + f_loss)/2
                WGAN:
                    loss_D = -torch.mean(r_logit) + torch.mean(f_logit)
                WGAN-GP:
                    gradient_penalty = self.gp(r_imgs, f_imgs)
                    loss_D = -torch.mean(r_logit) + torch.mean(f_logit) + gradient_penalty
                """
                # Loss for discriminator
                # r_loss = self.loss(r_logit, r_label)
                # f_loss = self.loss(f_logit, f_label)
                loss_D = -torch.mean(r_logit) + torch.mean(f_logit) + self.gp(r_imgs, f_imgs)

                # Discriminator backwarding
                self.D.zero_grad()
                loss_D.backward()
                self.opt_D.step()

                """
                NOTE FOR SETTING WEIGHT CLIP:

                WGAN: below code
                """
                # for p in self.D.parameters():
                #     p.data.clamp_(-self.config["clip_value"], self.config["clip_value"])



                # *********************
                # *    Train G        *
                # *********************
                if self.steps % self.config["n_critic"] == 0:
                    # Generate some fake images.
                    z = Variable(torch.randn(bs, self.config["z_dim"])).to(self.config["device"])
                    f_imgs = self.G(z)

                    # Generator forwarding
                    f_logit = self.D(f_imgs)

                    """
                    NOTE FOR SETTING LOSS FOR GENERATOR:

                    GAN: loss_G = self.loss(f_logit, r_label)
                    WGAN: loss_G = -torch.mean(self.D(f_imgs))
                    WGAN-GP: loss_G = -torch.mean(self.D(f_imgs))
                    """
                    # Loss for the generator.
                    # loss_G = self.loss(f_logit, r_label)
                    loss_G = -torch.mean(self.D(f_imgs))

                    # Generator backwarding
                    self.G.zero_grad()
                    loss_G.backward()
                    self.opt_G.step()

                if self.steps % 10 == 0:
                    progress_bar.set_postfix(loss_G=loss_G.item(), loss_D=loss_D.item())
                self.steps += 1

            self.G.eval()
            f_imgs_sample = (self.G(self.z_samples).data + 1) / 2.0
            filename = os.path.join(self.log_dir, f'Epoch_{epoch+1:03d}.jpg')
            torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
            logging.info(f'Save some samples to {filename}.')

            # Show some images during training.
            # grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
            # plt.figure(figsize=(10,10))
            # plt.imshow(grid_img.permute(1, 2, 0))
            # plt.show()

            self.G.train()

            if (e+1) % 5 == 0 or e == 0 or e == self.config['n_epoch'] - 1:
                # Save the checkpoints.
                torch.save(self.G.state_dict(), os.path.join(self.ckpt_dir, f'G_{e}.pth'))
                torch.save(self.D.state_dict(), os.path.join(self.ckpt_dir, f'D_{e}.pth'))
                self.last_G_cktp = os.path.join(self.ckpt_dir, f'G_{e}.pth') 
                self.last_D_cktp = os.path.join(self.ckpt_dir, f'D_{e}.pth') 

        logging.info('Finish training')

    def inference(self, G_path=None, D_path=None, n_generate=1000, n_output=30, show=False):
        if G_path != None:
            self.last_G_cktp = G_path
        self.G.load_state_dict(torch.load(self.last_G_cktp))
        self.G.to(self.config["device"])
        self.G.eval()
        if D_path != None:
            self.last_D_cktp = D_path
        self.D.load_state_dict(torch.load(self.last_D_cktp))
        self.D.to(self.config["device"])
        self.D.eval()

        dataset = get_dataset(os.path.join(self.config["workspace_dir"], 'faces'))
        # dataset += get_dataset(os.path.join(self.config["workspace_dir"], 'faces2'))
        self.dataloader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True, num_workers=2)

        for e, epoch in enumerate(range(self.config["d_epoch_num"])):
            self.G.eval()
            self.D.train()
            progress_bar = tqdm(self.dataloader)
            progress_bar.set_description(f"Discriminator Epoch {e+1}")
            for i, data in enumerate(progress_bar):
                imgs = data.to(self.config["device"])
                bs = imgs.size(0)

                z = Variable(torch.randn(bs, self.config["z_dim"])).to(self.config["device"])
                r_imgs = Variable(imgs).to(self.config["device"])
                f_imgs = self.G(z)
                r_label = torch.ones((bs)).to(self.config["device"])
                f_label = torch.zeros((bs)).to(self.config["device"])

                r_logit = self.D(r_imgs)
                f_logit = self.D(f_imgs)
                loss_D = -torch.mean(r_logit) + torch.mean(f_logit) + self.gp(r_imgs, f_imgs)

                self.D.zero_grad()
                loss_D.backward()
                self.opt_D.step()

        self.G.eval()
        self.D.eval()
        print('Start Generate')
        count = 1
        while count <= n_generate:
            z = Variable(torch.randn(self.config["batch_size"], self.config["z_dim"])).to(self.config["device"])
            imgs = (self.G(z).data + 1) / 2.0
            logits = self.D(imgs)
                
            for logit, img in zip(logits, imgs):
                if logit > self.config["d_threshold"] * logits.std() + logits.mean():
                    torchvision.utils.save_image(img, os.path.join(self.output_dir, f'{count}_{logit}.jpg'))
                    count += 1
                if count > n_generate:
                    break

        if show:
            row, col = n_output//10 + 1, 10
            grid_img = torchvision.utils.make_grid(imgs[:n_output].cpu(), nrow=row)
            plt.figure(figsize=(row, col))
            plt.imshow(grid_img.permute(1, 2, 0))
            plt.show()

config = {
    "model_type": "WGAN-GP-TH-G_b64_n100_c2_z512_l50_dn3_dth17_sz96",
    "batch_size": 64,
    "lr": 1e-4,
    "n_epoch": 150,
    "n_critic": 2,
    "z_dim": 512,
    "clip_value":0.005,
    "gp_lambda":50,
    "d_epoch_num":3,
    "d_threshold":1.7,
    "workspace_dir": workspace_dir, # define in the environment setting
    "device":"cuda:6"
}

trainer = TrainerGAN(config)
trainer.train()
trainer.inference()
