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

train = np.load('data/trainingset.npy', allow_pickle=True)
test = np.load('data/testingset.npy', allow_pickle=True)

print(train.shape)
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
device = "cuda:4" if torch.cuda.is_available() else "cpu"

# Model Architecture
class Permute(nn.Module):
    def __init__(self, perm):
        super(Permute, self).__init__()
        self.perm = perm

    def forward(self, x):
        return x.permute(*self.perm)

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(x.size(0), *self.shape)

class attn_autoencoder(nn.Module):
    def __init__(self):
        super(attn_autoencoder, self).__init__()
        self.attn_encoder = nn.Sequential(
            self.conv(3, 8),
            self.attn(in_dim=8,w=32),
            self.conv(8, 16),
            self.attn(in_dim=16,w=16),
            self.conv(16, 32),
            self.attn(in_dim=32,w=8),
            self.conv(32, 64, act=nn.Tanh),
            nn.Dropout(0.1),
        )

        self.decoder = nn.Sequential(
            self.trans_conv(64, 32),
            self.attn(in_dim=32,w=8),
            self.trans_conv(32, 16),
            self.attn(in_dim=16,w=16),
            self.trans_conv(16, 8),
            self.attn(in_dim=8,w=32),
            self.trans_conv(8, 3, act=nn.Tanh),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        bs = x.size(0)
        attn_out = self.attn_encoder(x)
        x = self.decoder(attn_out)
        return x
    
    def conv(self, in_dim, out_dim, act=nn.ReLU):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_dim,
                out_channels=out_dim,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(out_dim),
            act(),
        )
        
    def trans_conv(self, in_dim, out_dim, act=nn.ReLU):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_dim,
                out_channels=out_dim,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(out_dim),
            act(),
        )

    def attn(self, in_dim, w):
        return nn.Sequential(
            Permute([0, 2, 3, 1]),
            Reshape([-1, in_dim]),
            nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=in_dim,
                    nhead=2
                ),
                num_layers=1,
            ),
            Reshape([w, w, in_dim]),
            Permute([0, 3, 1, 2]),
        )
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

# Training hyperparameters
num_epochs = 300
batch_size = 128
learning_rate = 2e-4

# Build training dataloader
x = torch.from_numpy(train)
train_dataset = CustomTensorDataset(x)

train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

# Model
model_type = 'attn'   # selecting a model type from {'cnn', 'fcn', 'vae', 'resnet'}
model = attn_autoencoder().to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train
best_loss = np.inf
model.train()

qqdm_train = qqdm(range(num_epochs), desc=format_str('bold', 'Description'))
for epoch in qqdm_train:
    tot_loss = list()
    for data in train_dataloader:

        # ===================loading=====================
        img = data.float().to(device)

        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)

        tot_loss.append(loss.item())
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================save_best====================
    mean_loss = np.mean(tot_loss)
    if mean_loss < best_loss:
        best_loss = mean_loss
        torch.save(model, 'best_model_{}.pt'.format(model_type))
    # ===================log========================
    qqdm_train.set_infos({
        'epoch': f'{epoch + 1:.0f}/{num_epochs:.0f}',
        'loss': f'{mean_loss:.4f}',
    })
    # ===================save_last========================
    torch.save(model, 'last_model_{}.pt'.format(model_type))

eval_batch_size = 128

# Inference
# build testing dataloader
data = torch.tensor(test, dtype=torch.float32)
test_dataset = CustomTensorDataset(data)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=eval_batch_size, num_workers=1)
eval_loss = nn.MSELoss(reduction='none')

# load trained model
checkpoint_path = f'last_model_{model_type}.pt'
model = torch.load(checkpoint_path)
model.eval()

# prediction file
out_file = f'prediction_{model_type}.csv'

anomality = list()
with torch.no_grad():
  for i, data in enumerate(test_dataloader):
    img = data.float().to(device)
    output = model(img)
    loss = eval_loss(output, img).sum([1, 2, 3])
    anomality.append(loss)

anomality = torch.cat(anomality, axis=0)
anomality = torch.sqrt(anomality).reshape(len(test), 1).cpu().numpy()

df = pd.DataFrame(anomality, columns=['score'])
df.to_csv(out_file, index_label = 'ID')
