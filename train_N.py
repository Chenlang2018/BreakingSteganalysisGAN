import random
import os, time
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
from torch import optim
from scipy.io import wavfile
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
import wave as audio
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.io.wavfile as wave
import wave as audio
import model

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### READ DATA###
def read_wav_data(filename):
    wav = audio.open(filename,"rb")
    num_frame = wav.getnframes()
    str_data = wav.readframes(num_frame)
    wav.close()
    wave_data = np.fromstring(str_data,dtype=np.short)
    return wave_data

def wav_read(p_path,n_path):
    data = []
    rootdir = p_path
    list = os.listdir(rootdir)
    for i in range(0,len(list)):
        path = os.path.join(rootdir,list[i])
        if os.path.isfile(path):
            wav = read_wav_data(path)
            data.append(wav)

    rootdir = n_path
    list = os.listdir(rootdir)
    for i in range(0,len(list)):
        path = os.path.join(rootdir,list[i])
        if os.path.isfile(path):
            wav = read_wav_data(path)
            data.append(wav)
    data=np.array(data)
    return data


p_path = 'original cover path'
n_path = 'normal stego path'
data = wav_read(p_path,n_path)
print(data.shape)

ones = np.ones(12000)
zeros = np.zeros(12000)
y = np.concatenate((ones,zeros),0)

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=0)

X_train = X_train.reshape(-1, 1, 16384)
X_train = torch.from_numpy(X_train.astype(np.float32))

X_test = X_test.reshape(-1, 1, 16384)
X_test = torch.from_numpy(X_test.astype(np.float32))

y_train = y_train.reshape(-1,1)
y_train = torch.FloatTensor(y_train)

y_test = y_test.reshape(-1,1)
y_test = torch.LongTensor(y_test)

sample_generator = Data.TensorDataset(X_train, y_train)
random_data_loader = Data.DataLoader(
    dataset=sample_generator,
    batch_size=64,
    shuffle=True,
    drop_last=True,
    pin_memory=True)
print('DataLoader created')

test_generator = Data.TensorDataset(X_test, y_test)
test_loader = Data.DataLoader(
    dataset=test_generator,
    batch_size=64,
    shuffle=True,
    drop_last=True,
    pin_memory=True)
print('TestLoader created')

s_learning_rate = 0.0001
steganalysis = torch.nn.DataParallel(Steganalysis().to(device))
print(steganalysis)
s_optimizer = optim.Adam(steganalysis.parameters(),lr=s_learning_rate)
loss = torch.nn.BCELoss().to(device)
train_epoch = 20
test_acc = []
for epoch in range(train_epoch):
    running_loss = 0.0
    for i, (x, y) in enumerate(random_data_loader):
        x = x.to(device)
        y = y.to(device)

        output = steganalysis(x)
        loss_out = loss(output,y)
        s_optimizer.zero_grad()
        loss_out.backward()
        s_optimizer.step()

        running_loss += loss_out.item()
        j = float(i+1)
        print("epoch:%d,batch:%d,loss:%f" % (epoch + 1,i+1, running_loss / j))

        if i == 374:
            steganalysis.eval()
            train_acc = 0.0
            total = 0
            for j,(test_x,test_y) in enumerate(test_loader):
                test_x = test_x.to(device)
                test_y = test_y.to(device)
                output = steganalysis(test_x)
                pre_y = torch.LongTensor(64)
                pre_y = pre_y.to(device)
                for k in range(64):
                    if output[k] > 0.5:
                        pre_y[k] = 1
                    else:
                        pre_y[k] = 0
                train_acc = (pre_y == test_y.squeeze(1)).sum().item() + train_acc
                total += y.size(0)
            steganalysis.train()
            acc = train_acc / total
            print("epoch:%d,acc:%f" % (epoch + 1, acc))
            test_acc.append(acc)

epochs = range(1,len(test_acc)+1)
plt.plot(epochs, test_acc, 'r', linewidth=1.0, label='Training acc')
plt.title('Training and validation acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
torch.save(steganalysis.state_dict(), 'chen.pkl')