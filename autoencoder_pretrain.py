import h5py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from matplotlib import pyplot as plt
from tqdm import tqdm
import gc

if torch.cuda.is_available():
    cuda_is_available = True
    print("Cuda is available.")
else:
    cuda_is_available = False
    print("Cuda is NOT available.")

learning_rate = 1e-3
num_epochs = 300

# with h5py.File('../input/ae-training-data/ae_data.h5', 'r') as hf:
#     data = hf['training_data_1095'][::5]
#     # data = hf['training_data_1095'][:50]

with h5py.File('../input/aetrainingdataaddingdjia/ae_data_addingDJIA.h5', 'r') as hf:
    data = hf['training_data_1095_addingDJIA'][:]

print(data.shape)
trainingdata = torch.Tensor(data)
trainingdata_loader = DataLoader(trainingdata, batch_size=50, shuffle=True)

del data
gc.collect()

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            #nn.Linear(365 * 3 * 36, 6570),
            nn.Linear(365 * 3 * 41, 6570),
            nn.ReLU(True),
            nn.Linear(6570, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 256),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 6570),
            nn.ReLU(True),
            #nn.Linear(6570, 365 * 3 * 36),
            nn.Linear(6570, 365 * 3 * 41),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if cuda_is_available:
    model = autoencoder().cuda()
else:
    model = autoencoder()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5
)

for epoch in tqdm(range(num_epochs)):
    for data in trainingdata_loader:
        data = data.view(data.size(0), -1)
        if cuda_is_available:
            data = Variable(data).cuda()
        else:
            data = Variable(data)
        # fwd
        output = model(data)
        loss = criterion(output, data)
        # bp
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("epoch[{}/{}], loss:{:.4f}".format(epoch + 1, num_epochs, loss.data))

    if epoch % 10 == 0 or epoch == num_epochs-1:
        savename_origin = "./epoch_" + str(epoch) + "_origin.png"
        savename_origin_all = "./epoch_" + str(epoch) + "_origin_all.png"
        savename_output = "./epoch_" + str(epoch) + "_output.png"
        savename_output_all = "./epoch_" + str(epoch) + "_output_all.png"

        if cuda_is_available:
            a = data.cpu().data
        else:
            a = data.data
        #a = a.reshape((-1, 1095, 36))
        a = a.reshape((-1, 1095, 41))
        plt.clf()
        plt.plot(range(len(a[0])), a[0, :, :10])
        plt.savefig(savename_origin)
        plt.clf()
        plt.plot(range(len(a[0])), a[0, :, :])
        plt.savefig(savename_origin_all)

        if cuda_is_available:
            b = output.cpu().data
        else:
            b = output.data
        #b = b.reshape((-1, 1095, 36))
        b = b.reshape((-1, 1095, 41))

        plt.clf()
        plt.plot(range(len(b[0])), b[0, :, :10])
        plt.savefig(savename_output)
        plt.clf()
        plt.plot(range(len(b[0])), b[0, :, :])
        plt.savefig(savename_output_all)

        torch.save(model.state_dict(), "./ae_model_1095to256_addingDJIA.pth")