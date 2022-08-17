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

same_seeds(0x666)
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
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
    transform = transforms.Compose(compose)
    dataset = CrypkoDataset(fnames, transform)
    return dataset

# Generator
class AddNoise(nn.Module):
    def __init__(self, factor=1e-7):
        super(AddNoise, self).__init__()
        self.factor = factor

    def forward(self, x, n):
        """
        x: (bs, channel, h, w)
        n: (bs, n_dim)
        output: (bs, channel, h, w)
        """
        _n = n.reshape([-1])[:x.reshape([-1]).size(0)].reshape_as(x)
        return x + self.factor * _n * torch.rand(size=x.shape).to(x.device)

class AdaMod(nn.Module):
    def __init__(self, x_c, x_h, x_w, w_dim):
        super(AdaMod, self).__init__()
        self.y1 = nn.Linear(w_dim, x_c * x_h * x_w)
        self.y2 = nn.Linear(w_dim, x_c * x_h * x_w)

    def forward(self, x, w):
        """
        x: (bs, channel, h, w)
        w: (bs, w_dim)
        output: (bs, channel, h, w)
        """
        return torch.reshape(self.y1(w), x.shape) * x \
                + torch.reshape(self.y2(w), x.shape)

class AdaNorm(nn.Module):
    def __init__(self, ax = (2, 3), eps = 1e-5):
        super(AdaNorm, self).__init__()
        self.ax = ax
        self.eps = eps

    def forward(self, x):
        """
        input: (bs, channel, h, w)
        output: (bs, channel, h, w)
        """
        mean = torch.mean(x, dim = self.ax, keepdim = True)
        diff = x - mean
        var = torch.mean(torch.square(diff), dim = self.ax, keepdim = True)
        return diff * torch.rsqrt(var + self.eps)

class Map(nn.Module):
    def __init__(self, in_dim, out_dim, emb_dim, n_layer=5):
        super(Map, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, emb_dim, bias = False),
            *[nn.Linear(emb_dim, emb_dim) for _ in range(n_layer)],
            nn.Linear(emb_dim, out_dim, bias = False),
        )

    def forward(self, z):
        """
        z: (bs, in_dim)
        output: (bs, out_dim)
        """
        return self.fc(z)

class StyleBlock(nn.Module):
    def __init__(self, x_shape, w_dim, out_dim, upsample=False, noise=False):
        super(StyleBlock, self).__init__()
        self.norm = AdaNorm()
        self.ada_mod = AdaMod(x_shape[0], x_shape[1], x_shape[2], w_dim)
        self.upsample = upsample
        if self.upsample:
            self.up_layer = nn.Upsample(scale_factor = 2, mode = "bilinear")
        else:
            self.up_layer = None
        self.conv = nn.Conv2d(in_channels = x_shape[0],\
                                out_channels = out_dim,\
                                kernel_size = 3, stride = 1, padding = 1)
        self.noise = noise
        self.add_noise = AddNoise()

    def forward(self, input):
        """
        x: (bs, channel, h, w)
        w: (bs, w_dim)
        n: (bs, n_dim)
        output: (bs, out_dim, h, w)
        output(upsample = True): (bs, out_dim, 2 * w, 2 * h)
        output w: (bs, w_dim)
        output n: (bs, n_dim)
        """
        x, w, n = input
        x = self.ada_mod(x, w)
        if self.upsample:
            x = self.up_layer(x)
        x = self.conv(x)
        if self.noise:
            x = self.add_noise(x, n)
        x = self.norm(x)
        x = nn.LeakyReLU(0.2)(x)
        return (x, w, n)

class Generator(nn.Module):
    """
    Input shape: (batch, z_dim)
    Output shape: (batch, 3, 64, 64)
    """
    def __init__(self, z_dim, w_dim, emb_dim):
        super(Generator, self).__init__()
        self.map = Map(in_dim=z_dim, out_dim=w_dim, emb_dim=emb_dim)
        self.model = nn.Sequential(
            StyleBlock(x_shape=torch.Size([512, 2, 2]), w_dim=w_dim, out_dim=1024),
            StyleBlock(x_shape=torch.Size([1024, 2, 2]), w_dim=w_dim, out_dim=1024, upsample=True),
            StyleBlock(x_shape=torch.Size([1024, 4, 4]), w_dim=w_dim, out_dim=256),
            StyleBlock(x_shape=torch.Size([256, 4, 4]), w_dim=w_dim, out_dim=256, upsample=True),
            StyleBlock(x_shape=torch.Size([256, 8, 8]), w_dim=w_dim, out_dim=256),
            StyleBlock(x_shape=torch.Size([256, 8, 8]), w_dim=w_dim, out_dim=64, upsample=True),
            StyleBlock(x_shape=torch.Size([64, 16, 16]), w_dim=w_dim, out_dim=64),
            StyleBlock(x_shape=torch.Size([64, 16, 16]), w_dim=w_dim, out_dim=16, upsample=True),
            StyleBlock(x_shape=torch.Size([16, 32, 32]), w_dim=w_dim, out_dim=16),
            StyleBlock(x_shape=torch.Size([16, 32, 32]), w_dim=w_dim, out_dim=8, upsample=True),
            StyleBlock(x_shape=torch.Size([8, 64, 64]), w_dim=w_dim, out_dim=3),
        )

    def forward(self, z):
        bs = z.size(0)
        x = torch.ones((bs, 512, 2, 2)).to(z.device)
        w = self.map(z)
        n = torch.rand((bs, 70000)).to(z.device)
        x, _, _ = self.model((x, w, n))
        return x

# Discriminator
class Discriminator(nn.Module):
    """
    Input shape: (batch, 3, 64, 64)
    Output shape: (batch)
    """
    def __init__(self, in_dim, feature_dim=64):
        super(Discriminator, self).__init__()

        #input: (batch, 3, 64, 64)
        self.l1 = nn.Sequential(
            nn.Conv2d(in_dim, feature_dim, kernel_size=4, stride=2, padding=1), #(batch, 3, 32, 32)
            nn.LeakyReLU(0.2),
            self.conv_bn_lrelu(feature_dim, feature_dim * 2),                   #(batch, 3, 16, 16)
            self.conv_bn_lrelu(feature_dim * 2, feature_dim * 4),               #(batch, 3, 8, 8)
            self.conv_bn_lrelu(feature_dim * 4, feature_dim * 8),               #(batch, 3, 4, 4)
            nn.Conv2d(feature_dim * 8, 1, kernel_size=4, stride=1, padding=0),
            nn.Dropout(0.17),
        )
        self.apply(weights_init)

    def conv_bn_lrelu(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 4, 2, 1),
            nn.InstanceNorm2d(out_dim, affine = True),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1)
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

        self.G = Generator(self.config["z_dim"], self.config["w_dim"], self.config["emb_dim"])
        self.D = Discriminator(3)

        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=self.config["lr"], betas=(0.5, 0.999))
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=self.config["lr"], betas=(0.5, 0.999))

        self.dataloader = None
        self.log_dir = os.path.join(self.config["workspace_dir"], 'logs')
        self.ckpt_dir = os.path.join(self.config["workspace_dir"], 'checkpoints')
        self.last_cktp = ''
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
        self.log_dir = os.path.join(self.log_dir, f'{self.config["model_type"]}_'+time)
        self.ckpt_dir = os.path.join(self.ckpt_dir, f'{self.config["model_type"]}_'+time)
        self.output_dir = os.path.join(self.output_dir, f'{self.config["model_type"]}_'+time)
        os.makedirs(self.log_dir)
        os.makedirs(self.ckpt_dir)
        os.makedirs(self.output_dir)

        # create dataset by the above function
        dataset = get_dataset(os.path.join(self.config["workspace_dir"], 'faces'))
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

                loss_D = -torch.mean(r_logit) + torch.mean(f_logit) + self.gp(r_imgs, f_imgs)

                # Discriminator backwarding
                self.D.zero_grad()
                loss_D.backward()
                self.opt_D.step()

                # *********************
                # *    Train G        *
                # *********************
                if self.steps % self.config["n_critic"] == 0:
                    # Generate some fake images.
                    z = Variable(torch.randn(bs, self.config["z_dim"])).to(self.config["device"])
                    f_imgs = self.G(z)

                    # Generator forwarding
                    f_logit = self.D(f_imgs)
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

            self.G.train()

            if (e+1) % 5 == 0 or e == 0 or e == self.config["n_epoch"] - 1:
                # Save the checkpoints.
                torch.save(self.G.state_dict(), os.path.join(self.ckpt_dir, f'G_{e}.pth'))
                torch.save(self.D.state_dict(), os.path.join(self.ckpt_dir, f'D_{e}.pth'))
                self.last_cktp = os.path.join(self.ckpt_dir, f'G_{e}.pth') 

        logging.info('Finish training')

    def inference(self, G_path=None, n_generate=1000, n_output=30, show=False):
        """
        1. G_path is the path for Generator ckpt
        2. You can use this function to generate final answer
        """
        self.G.load_state_dict(torch.load(self.last_cktp))
        self.G.to(self.config["device"])
        self.G.eval()
        z = Variable(torch.randn(n_generate, self.config["z_dim"])).to(self.config["device"])
        imgs = (self.G(z).data + 1) / 2.0

        for i in range(n_generate):
            torchvision.utils.save_image(imgs[i], os.path.join(self.output_dir, f'{i+1}.jpg'))

        if show:
            row, col = n_output//10 + 1, 10
            grid_img = torchvision.utils.make_grid(imgs[:n_output].cpu(), nrow=row)
            plt.figure(figsize=(row, col))
            plt.imshow(grid_img.permute(1, 2, 0))
            plt.show()

config = {
    "model_type": "Style-WGAN-GP",
    "batch_size": 32,
    "lr": 1e-4,
    "n_epoch": 30,
    "n_critic": 1,
    "z_dim": 512,
    "w_dim": 1024,
    "emb_dim": 512,
    "gp_lambda":10,
    "workspace_dir": workspace_dir, # define in the environment setting
    # "device":"cpu",
    "device":"cuda:7",
}

trainer = TrainerGAN(config)
trainer.train()
trainer.inference()
