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
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.net = models.resnet34()
        self.fc = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.fc(self.net(x).view(x.size(0), -1))

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
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.Dropout(0.1),
        )

        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            nn.Dropout(0.1),
        )

        self.decoder = nn.Sequential(
            nn.InstanceNorm1d(320),
            nn.Linear(320, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 64 * 64 * 3),
            nn.Dropout(0.1),
            nn.Tanh(),
        )
        
    def forward(self, x, fake=False):
        if not fake:
            bs = x.size(0)
            linear_out = self.linear_encoder(x.view(bs, -1))
            cnn_out = self.cnn_encoder(x).view(bs, -1)
            cat_out = torch.cat((linear_out, cnn_out), dim = 1)
            img = self.decoder(cat_out).reshape((bs, 3, 64, 64))
        else:
            bs = x.size(0)
            img = self.decoder(torch.randn((bs, 320)).to(device)).reshape((bs, 3, 64, 64))
        return img

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
num_epochs = 1000
num_classify_epochs = 25
batch_size = 1024
learning_rate = 1e-4

# Build training dataloader
x = torch.from_numpy(train)
train_dataset = CustomTensorDataset(x)

train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

# Model
model_type = 'multi_fc_cnn_class'   # selecting a model type from {'cnn', 'fcn', 'vae', 'resnet'}
model_ae  = multi_autoencoder().to(device)
model_class = Classifier().to(device)

# Loss and optimizer
criterion_ae = nn.MSELoss()
optimizer_ae = torch.optim.Adam(model_ae.parameters(), lr=learning_rate)
criterion_class = nn.CrossEntropyLoss()
optimizer_class = torch.optim.Adam(model_class.parameters(), lr=learning_rate) 

# Train
best_loss = np.inf

log_file = open(f'{model_type}.log', 'w')
qqdm_train = qqdm(range(num_epochs), desc=format_str('bold', 'Description'))
model_ae.train()
for epoch in qqdm_train:
    tot_ae_loss = list()
    for data in train_dataloader:
        img = data.float().to(device)
        out_ae = model_ae(img)
        loss_ae = criterion_ae(out_ae, img)
        tot_ae_loss.append(loss_ae.item())
        optimizer_ae.zero_grad()
        loss_ae.backward()
        optimizer_ae.step()

    qqdm_train.set_infos({
        'epoch': f'{epoch + 1:.0f}/{num_epochs:.0f}',
        'ae_loss': f'{np.mean(tot_ae_loss):.4e}',
    })
    log_file.write(
        f'ae epoch: {epoch + 1:.0f}/{num_epochs:.0f}, ae_loss: {np.mean(tot_ae_loss):.4e}\n'
    )
    torch.save(model_ae, 'last_model_ae_{}.pt'.format(model_type))

qqdm_train = qqdm(range(num_classify_epochs), desc=format_str('bold', 'Description'))
checkpoint_path = f'last_model_ae_{model_type}.pt'
model_ae = torch.load(checkpoint_path)
model_ae.eval()
model_class.train()
for epoch in qqdm_train:
    tot_class_loss = list()
    tot_r_acc = list()
    tot_f_acc = list()
    for data in train_dataloader:
        img = data.float().to(device)
        r_logits = model_class(model_ae(img))
        f_logits = model_class(model_ae(img, fake=True))
        r_loss = criterion_class(r_logits, torch.Tensor([[1, 0] for _ in range(r_logits.size(0))]).to(device))
        f_loss = criterion_class(f_logits, torch.Tensor([[0, 1] for _ in range(f_logits.size(0))]).to(device))
        loss_class = r_loss + f_loss
        tot_class_loss.append(loss_class.item())
        optimizer_class.zero_grad()
        loss_class.backward()
        optimizer_class.step()
        tot_r_acc.append(1- r_logits.argmax(dim=1).sum().item() / r_logits.size(0))
        tot_f_acc.append(f_logits.argmax(dim=1).sum().item() / f_logits.size(0))

    qqdm_train.set_infos({
        'epoch': f'{epoch + 1:.0f}/{num_epochs:.0f}',
        'r_acc': f'{np.mean(tot_r_acc):.4f}',
        'f_acc': f'{np.mean(tot_f_acc):.4f}',
        'class_loss': f'{np.mean(tot_class_loss):.4e}',
    })
    log_file.write(
        f'classify epoch: {epoch + 1:.0f}/{num_epochs:.0f}, r_acc: {np.mean(tot_r_acc):.4f}, f_acc: {np.mean(tot_f_acc):.4f}, class_loss: {np.mean(tot_class_loss):.4e}\n'
    )
    torch.save(model_class, 'last_model_class_{}.pt'.format(model_type))

log_file.close()
eval_batch_size = 1024

# Inference
# build testing dataloader
data = torch.tensor(test, dtype=torch.float32)
test_dataset = CustomTensorDataset(data)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=eval_batch_size, num_workers=1)

# load trained model
checkpoint_path = f'last_model_ae_{model_type}.pt'
model_ae = torch.load(checkpoint_path)
model_ae.eval()
checkpoint_path = f'last_model_class_{model_type}.pt'
model_class = torch.load(checkpoint_path)
model_class.eval()

# prediction file
out_file = f'prediction_{model_type}.csv'

anomality = list()
with torch.no_grad():
  for i, data in enumerate(test_dataloader):
    img = data.float().to(device)
    output = model_class(model_ae(img))
    anomality.append(output.softmax(dim=1).narrow(1, 1, 1))

anomality = torch.cat(anomality, axis=0)
anomality = torch.sqrt(anomality).reshape(len(test), 1).cpu().numpy()

df = pd.DataFrame(anomality, columns=['score'])
df.to_csv(out_file, index_label = 'ID', float_format='%.20f')
