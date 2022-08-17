import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.optim import Adam, AdamW
from qqdm import qqdm, format_str
import pandas as pd

test = np.load('data/testingset.npy', allow_pickle=True)

print(test.shape)

def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(0x1725)
device = "cuda:2" if torch.cuda.is_available() else "cpu"

# Model Architecture
class multi_autoencoder(nn.Module):
    def __init__(self):
        super(multi_autoencoder, self).__init__()
        self.resnet = nn.Sequential(
            models.resnet34(),
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )
        self.convext = nn.Sequential(
            models.convnext_base(),
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )
        self.decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 64 * 64 * 3),
            nn.Dropout(0.2),
            nn.Tanh(),
        )

    def forward(self, x):
        bs = x.size(0)
        cat_out = torch.cat((
            self.resnet(x),
            self.convext(x),
        ), dim = 1)
        x = self.decoder(cat_out).reshape((bs, 3, 64, 64))
        return x

# Dataset
class CustomTensorDataset(TensorDataset):
    """
    TensorDataset with support of transforms.
    """
    def __init__(self, tensors):
        self.tensors = tensors
        if tensors.shape[-1] == 3:
            self.tensors = tensors.permute(0, 3, 1, 2)

        self.transform = transforms.Compose([
          transforms.Lambda(lambda x: x.to(torch.float32)),
          transforms.Lambda(lambda x: 2. * x/255. - 1.),
        ])

    def __getitem__(self, index):
        x = self.tensors[index]

        if self.transform:
            # mapping images to [-1.0, 1.0]
            x = self.transform(x)

        return x

    def __len__(self):
        return len(self.tensors)

# Model
model_type = 'multi_res_convext'
model = multi_autoencoder().to(device)

eval_batch_size = 1

# Inference
# build testing dataloader
data = torch.tensor(test, dtype=torch.float32)
test_dataset = CustomTensorDataset(data)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=eval_batch_size, num_workers=1)
eval_loss = nn.MSELoss(reduction='none')

# load trained model
checkpoint_path = f'best_model_{model_type}.pt'
model = torch.load(checkpoint_path)
model.eval()

anomality = list()
from PIL import Image

with torch.no_grad():
  for i, data in enumerate(test_dataloader):
    img = data.float().to(device)
    transforms.ToPILImage()(img[0]).show()
    output = model(img)
    transforms.ToPILImage()(output[0]).show()
    input()
    loss = eval_loss(output, img).sum([1, 2, 3])
    anomality.append(loss)
