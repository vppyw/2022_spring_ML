import os
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
device="cuda:5" if torch.cuda.is_available() else "cpu"

# Model Architecture
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            models.resnet34(),
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Dropout(0.1),
        )
    def forward(self, x):
        return self.net(x)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(6, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 12, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.enc_out_1 = nn.Sequential(
            nn.Conv2d(24, 24, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.enc_out_2 = nn.Sequential(
            nn.Conv2d(24, 24, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.decoder = nn.Sequential(
		    nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            nn.ReLU(),
		    nn.ConvTranspose2d(24, 24, 3, stride=1, padding=1),
            nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),
            nn.ReLU(),
		    nn.ConvTranspose2d(12, 12, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),
            nn.Tanh(),
            nn.Dropout(0.1),
        )

    def encode(self, x):
        h1 = self.encoder(x)
        return self.enc_out_1(h1), self.enc_out_2(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(device)
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, gen=False):
        if not gen:
            mu, logvar = self.encode(x)
            z = self.reparametrize(mu, logvar)
            return self.decode(z), mu, logvar
        else:
            return self.decode(torch.randn((x.size(0), 48, 8, 8)).to(device))

def loss_vae(recon_x, x, mu, logvar, criterion):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    mse = criterion(recon_x, x)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return mse + KLD

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
num_epochs = 100
batch_size = 1024
learning_rate = 1e-4
num_classify_epoch = 50

# Build training dataloader
x = torch.from_numpy(train)
train_dataset = CustomTensorDataset(x)

train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

# Model
model_type = 'vae'   # selecting a model type from {'cnn', 'fcn', 'vae', 'resnet'}
model = VAE().to(device)
classifier = Classifier().to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
c_criterion = nn.CrossEntropyLoss()
c_optimizer = torch.optim.Adam(classifier.parameters(), lr=1.5e-4)


# Train
best_loss = np.inf
model.train()

qqdm_train = qqdm(range(num_epochs), desc=format_str('bold', 'Description'))
log = open(f'{model_type}.log', 'w')
model.train()
for epoch in qqdm_train:
    tot_loss = list()
    for data in train_dataloader:
        img = data.float().to(device)
        output = model(img)
        loss = loss_vae(output[0], img, output[1], output[2], criterion)
        tot_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    mean_loss = np.mean(tot_loss)
    if mean_loss < best_loss:
        best_loss = mean_loss
        torch.save(model, 'best_model_{}.pt'.format(model_type))
    # ===================log========================
    qqdm_train.set_infos({
        'epoch': f'{epoch + 1:.0f}/{num_epochs:.0f}',
        'loss': f'{mean_loss:.4f}',
    })
    log.write(
        f'vae loss: {mean_loss:.4f}\n'
    )
    # ===================save_last========================
    torch.save(model, 'last_model_{}.pt'.format(model_type))

model.eval()
classifier.train()
qqdm_classify = qqdm(range(num_classify_epoch), desc=format_str('bold', 'Description'))
for epoch in qqdm_classify:
    tot_r_acc = list()
    tot_f_acc = list()
    for data in train_dataloader:
        img = data.float().to(device)
        r_output, _, _ = model(img)
        f_output = model(x = img, gen = True)
        r_logit = classifier(r_output)
        f_logit = classifier(f_output)
        r_loss = c_criterion(r_logit, torch.Tensor([[1, 0] for _ in range(r_logit.size(0))]).to(device))
        f_loss = c_criterion(f_logit, torch.Tensor([[0, 1] for _ in range(r_logit.size(0))]).to(device))
        tot_r_acc.append(1 - r_logit.argmax(dim = 1).sum().item() / r_logit.size(0))
        tot_f_acc.append(f_logit.argmax(dim = 1).sum().item() / f_logit.size(0))
        loss = r_loss + f_loss
        c_optimizer.zero_grad()
        loss.backward()
        c_optimizer.step()

    qqdm_classify.set_infos({
        'epoch': f'{epoch + 1:.0f}/{num_classify_epoch:.0f}',
        'r_acc': f'{np.mean(tot_r_acc):.4f}',
        'f_acc': f'{np.mean(tot_f_acc):.4f}',
    })
    log.write(
        f'classifier r_acc: {np.mean(tot_r_acc):.4f}, f_acc: {np.mean(tot_f_acc):.4f}\n'
    )
    torch.save(classifier, 'last_classifier.pt')

log.close()
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
classifier = torch.load('last_classifier.pt')
classifier.eval()

# prediction file
out_file = f'prediction_{model_type}.csv'

anomality = list()
with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        img = data.float().to(device)
        output = model(img)
        output = classifier(output[0])
        anomality.append(output.softmax(dim=1).narrow(1, 1, 1))

anomality = torch.cat(anomality, axis=0)
anomality = torch.sqrt(anomality).reshape(len(test), 1).cpu().numpy()

df = pd.DataFrame(anomality, columns=['score'])
df.to_csv(out_file, index_label = 'ID')
