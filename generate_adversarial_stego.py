import os
import torch
import numpy as np
import torch.utils.data as Data
from torch import optim
from scipy.io import wavfile
import model
from sklearn.model_selection import train_test_split
use_devices = [0, 1, 2, 3]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### ORIGINAL COVER READ ###
data = []
rootdir = 'Original cover path'
list = os.listdir(rootdir)
for i in range(0, len(list)):
    path = os.path.join(rootdir, list[i])
    if os.path.isfile(path):
        fn, wav_data = wavfile.read(path)
        wave = (2. / 65535.) * (wav_data.astype(np.float32) - 32767) + 1.
        data.append(wave)
data = np.array(data)

yy = np.ones(15000)
y = np.loadtxt('../../DATA/label_15k.txt')

X_test = data
X_test = X_test.reshape(-1, 1, 16384)
X_test = torch.from_numpy(X_test)

### CREAT G ###
generator = torch.nn.DataParallel(model.Generator().to(device))
generator.cuda()

yy = torch.ones(15000)
sample_rate = 16384
batch_size = 1

### CREAT DATALOADER ###
sample_generator = Data.TensorDataset(X_test, yy)
random_data_loader = Data.DataLoader(
    dataset=sample_generator,
    batch_size=1,
    shuffle=False,
    drop_last=True,
    pin_memory=True)
print('DataLoader created')

### LSB MATCHING ###
def embed(generated_outputs):
    cover_audio = generated_outputs
    cover_audio = cover_audio.data.cpu().numpy()
    cover_audio = cover_audio.reshape(-1, 16384, 1)

    for h in range(batch_size):
        cover = cover_audio[h].reshape(16384, )
        cover = (cover - 1) / (2 / 65535) + 32767
        cover = cover.astype(np.int16)
        L = 16384
        stego = cover
        msg = np.random.randint(0, 2, L)
        msg = np.array(msg)
        k = np.random.randint(0, 2, L)
        k = np.array(k)
        for j in range(L):
            x = abs(cover[j])
            x = bin(x)
            x = x[2:]
            y = msg[j]
            if str(y) == x[-1]:
                stego[j] = cover[j]
            else:
                if k[j] == 0:
                    stego[j] = cover[j] - 1
                else:
                    stego[j] = cover[j] + 1
        stego = stego.reshape(16384)
    stego_audio = np.array(stego)
    return stego_audio

generator.load_state_dict(torch.load('best.pkl'))
for i, (ori, _) in enumerate(random_data_loader):
    ori = ori.to(device)
    generated_outputs = generator(ori)
    stego_audio = embed(generated_outputs)
    stego_audio = stego_audio.astype(np.int16)
    path = 'Adversarial Stego Save File Path' + str(i + 1) + 's.wav'
    wavfile.write(path, sample_rate, stego_audio)