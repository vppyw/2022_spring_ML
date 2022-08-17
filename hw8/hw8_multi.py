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
device = "cuda:1" if torch.cuda.is_available() else "cpu"

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

class multi_autoencoder(nn.Module):
    def __init__(self):
        super(multi_autoencoder, self).__init__()
        self.linear_encoder = nn.Sequential(
            nn.Linear(64 * 64 * 3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Dropout(0.1),
        )

        self.cnn_encoder = nn.Sequential(
            nn.LayerNorm([3, 64, 64]),
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LayerNorm([8, 64, 64]),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.LayerNorm([8, 32, 32]),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LayerNorm([16, 32, 32]),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.LayerNorm([16, 16, 16]),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LayerNorm([32, 16, 16]),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.LayerNorm([32, 8, 8]),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LayerNorm([32, 8, 8]),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.LayerNorm([16, 4, 4]),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LayerNorm([8, 4, 4]),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.decoder = nn.Sequential(
            nn.Linear(160, 320),
            nn.ReLU(),
            nn.Linear(320, 640),
            nn.ReLU(),
            nn.Linear(640, 1280),
            nn.ReLU(),
            nn.Linear(1280, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 64 * 64 * 3),
            nn.Dropout(0.1),
            nn.Tanh(),
        )

    def forward(self, x):
        bs = x.size(0)
        linear_out = self.linear_encoder(x.view(bs, -1))
        cnn_out = self.cnn_encoder(x).view(bs, -1)
        cat_out = torch.cat((linear_out, cnn_out), dim = 1)
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

# Training hyperparameters
num_epochs = 400
batch_size = 1024
learning_rate = 1e-4

# Build training dataloader
x = torch.from_numpy(train)
train_dataset = CustomTensorDataset(x)

train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

# Model
model_type = 'multi_fc_cnn'
model = multi_autoencoder().to(device)

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

eval_batch_size = 1024

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
