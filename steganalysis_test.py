import os, time
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
from torch import optim
from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import wave as audio
import torch.nn.functional as F
import model

def read_wav_data(filename):
    wav = audio.open(filename, "rb")
    num_frame = wav.getnframes()
    str_data = wav.readframes(num_frame)
    wav.close()
    wave_data = np.fromstring(str_data, dtype=np.short)
    return wave_data

def wav_read(p_path):
    data = []
    rootdir = p_path
    list = os.listdir(rootdir)
    for i in range(0, 3000):
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path):
            wav = read_wav_data(path)
            data.append(wav)
    data = np.array(data)
    return data

### STEGO READ ###
n_path = 'Stego file path'
data2 = wav_read(n_path)

y = np.loadtxt('../../DATA/label_3k.txt')
X_train = data2.reshape(-1, 1, 16384)
X_train = torch.from_numpy(X_train.astype(np.float32))

y_train = y.reshape(-1, 1)

y_train = torch.LongTensor(y_train)
y1_onehot = torch.FloatTensor(3000, 2)
y1_onehot.zero_()
y1_onehot.scatter_(1, y_train, 1)

sample_generator = Data.TensorDataset(X_train, y_train)
random_data_loader = Data.DataLoader(
    dataset=sample_generator,
    batch_size=64,
    shuffle=False,
    drop_last=True,
    pin_memory=False)
print('DataLoader created')

steganalysis = torch.nn.DataParallel(model.Steganalysis().to(device))

steganalysis.load_state_dict(torch.load('chen.pkl'))
steganalysis.eval()
train_acc = 0.0
total = 0
for i, (x, y) in enumerate(random_data_loader):
    x = x.to(device)
    y = y.to(device)

    output = steganalysis(x)
    pre_y = torch.LongTensor(64)
    pre_y = pre_y.to(device)
    for k in range(64):
        if output[k] > 0.5:
            pre_y[k] = 1
        else:
            pre_y[k] = 0
    train_acc = (pre_y == y.squeeze(1)).sum().item() + train_acc
    total += y.size(0)
    print("p_md:%f,total:%d" % (1- train_acc / total,total))
