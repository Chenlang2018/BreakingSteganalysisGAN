import os, time
import torch
import numpy as np
import torch.utils.data as Data
from torch import optim
from scipy.io import wavfile
import model
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from torch.autograd import Function
import matplotlib.pyplot as plt

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### LSB MATCHING ###
def embed(generated_outputs):
    cover_audio = generated_outputs
    cover_audio = cover_audio.data.cpu().numpy()
    cover_audio = cover_audio.reshape(-1, 16384, 1)
    stego_audio = []
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
        stego = stego.reshape(16384, )
        stego_audio.append(stego)
    stego_audio = np.array(stego_audio)
    #print(stego_audio)
    return stego_audio

### READ ORIGION COVER COVER ###
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
data = data.reshape(-1, 1, 16384)
data = torch.from_numpy(data)

### SELECT ONE ORIGINAL COVER AS TEST DATA ###
testdata = []
fn, wav_data = wavfile.read('Original cover path/1.wav')
wave = (2. / 65535.) * (wav_data.astype(np.float32) - 32767) + 1.
testdata.append(wave)
testdata = np.array(testdata)
testdata = testdata.reshape(-1, 1, 16384)
testdata = torch.from_numpy(testdata)

### TRAINING PARAMETERS ###
batch_size = 128
d_learning_rate = 0.0001
g_learning_rate = 0.0001
s_learning_rate = 0.0001
sample_rate = 16384
use_devices = [0, 1, 2, 3]

### CREAT D and G and N ###
discriminator = torch.nn.DataParallel(model.Discriminator().to(device), device_ids=use_devices)
discriminator.cuda()
# print(discriminator)
print('Discriminator created')

steganalysis = torch.nn.DataParallel(model.Steganalysis().to(device), device_ids=use_devices)
steganalysis.cuda()
# print(steganalysis)
print('steganalysis created')

generator = torch.nn.DataParallel(model.Generator().to(device), device_ids=use_devices)
generator.cuda()
# print(generator)
print('Generator created')

### CREAT DATALOADER ###
yy = torch.ones(15000)

sample_generator = Data.TensorDataset(data, yy)
random_data_loader = Data.DataLoader(
    dataset=sample_generator,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)
print('DataLoader created')

### DEFINITION OF LOSS ###
# loss = nn.CrossEntropyLoss()
bce_loss = torch.nn.BCELoss().to(device)
l1_loss = torch.nn.L1Loss().to(device)
l2_loss = torch.nn.MSELoss().to(device)

### Label ###
ones = np.ones(batch_size)
ones = ones.reshape(batch_size, 1)
ones = torch.FloatTensor(ones)

zeros = np.zeros(batch_size)
zeros = zeros.reshape(batch_size, 1)
zeros = torch.FloatTensor(zeros)

### Train! ###
print('Starting Training...')
start_time = time.time()
train_epoch = 60
D_num = 1
S_num = 1
G_num = 1
S_epoch = 30

steganalysis.load_state_dict(torch.load('chen.pkl')) #load steganalysis parameter
g_optimizer = optim.Adam(generator.parameters(), lr=g_learning_rate, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=d_learning_rate, betas=(0.5, 0.999))
g_scheduler = ReduceLROnPlateau(g_optimizer, 'min',factor=0.1,threshold=0.000001)
best = 10
b = 0.0
L1LOSS = []
BCELOSS = []
DLOSS = []
GLOSS = []

for epoch in range(train_epoch):
    generator = generator.train()
    for i, (ori, _) in enumerate(random_data_loader):
        ones = ones.to(device)
        zeros = zeros.to(device)
        ori = ori.to(device)

        ###train D
        for depoch in range(D_num):
            output = discriminator(ori)
            d_loss_real = torch.mean((output - 1.0) ** 2)

            generated_outputs = generator(ori)
            output = discriminator(generated_outputs.detach())
            d_loss_fake = torch.mean(output ** 2)

            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            discriminator.zero_grad()
            d_loss.backward()
            d_optimizer.step()

        ###train G
        for d_gepoch in range(G_num):
            generated_outputs = generator(ori)

            ###L1_loss
            l1_dist = torch.abs(torch.add(generated_outputs, torch.neg(ori)))
            g_cond_loss = torch.mean(l1_dist)

            ###L_G loss
            output1 = discriminator(generated_outputs)
            g_loss_d = 0.5 * torch.mean((output1 - 1.0) ** 2)

            ###L_N loss(BCE loss)
            if epoch > S_epoch:
                generated_outputs = generator(ori)
                stego_audio = embed(generated_outputs)
                stego_audio = stego_audio.astype(np.float32)
                stego_audio = stego_audio.reshape(-1, 1, 16384)
                stego_audio = torch.from_numpy(stego_audio).to(device)
                output2 = steganalysis(stego_audio)
                bceloss = bce_loss(output2, ones)
                if bceloss < best: #save the best
                    best = bceloss
                    gpath2 = './test/best.pkl'
                    torch.save(generator.state_dict(), gpath2)
                    print('Gnetbestsave')
                s_loss = bce_loss(output2,zeros)
            ###

            g_loss = 0.5 * g_loss_d + 100 * g_cond_loss
            generator.zero_grad()
            g_loss.backward()
            g_optimizer.step()

        print('########################################')
        if epoch > S_epoch:
            ###Acc testing
            train_acc = 0.0
            pre_y = torch.LongTensor(batch_size)
            pre_y = pre_y.to(device)
            yy = np.zeros(batch_size)
            yy = yy.reshape(-1, 1)
            yy = torch.LongTensor(yy)
            yy = yy.to(device)
            for k in range(batch_size):
                if output2[k] > 0.5:
                    pre_y[k] = 1
                else:
                    pre_y[k] = 0
            train_acc = (pre_y == yy.squeeze(1)).sum().item()
            print("ste_acc:%f" % (train_acc / batch_size))
            print('g_cond_loss:%f,g_loss_d:%f,bceloss:%f' % (g_cond_loss.item(), g_loss_d.item(), bceloss.item()))
            print('[%d/%d，%d/%d] - D_loss: %f,S_loss:%f, G_loss: %f' %
                  ((epoch + 1), train_epoch, i + 1, 15000 / batch_size, d_loss.item(), s_loss.item(), g_loss.item()))

        else:
            print('g_cond_loss:%f,g_loss_d:%f' % (g_cond_loss.item(), g_loss_d.item()))
            print('[%d/%d，%d/%d] - D_loss: %f, G_loss:%f' %
                  ((epoch + 1), train_epoch, i + 1, 15000 / batch_size, d_loss.item(), g_loss.item()))

        ### TEST EVERY EPOCH ###
        if i == 110:
            GLOSS.append(0.5 * g_loss_d.item())
            BCELOSS.append(bceloss.item())
            DLOSS.append(d_loss.item())
            L1LOSS.append(100 * g_cond_loss.item())
            generator = generator.eval()
            fake_speech = generator(testdata)
            fake_speech_data = fake_speech.data.cpu().numpy()
            fake_speech_data = fake_speech_data.reshape(-1, 16384, 1)
            fake_speech_data = fake_speech_data.reshape(16384)
            fake_speech_data = (fake_speech_data - 1) / (2 / 65535) + 32767
            fake_speech_data = fake_speech_data.astype(np.int16)
            path = './test/' + str(epoch + 1) + '.wav'
            wavfile.write(path, sample_rate, fake_speech_data)
            print('audio' + str(epoch + 1) + 'save')
            gpath = './test/Gnet' + str(epoch + 1) + '.pkl'
            torch.save(generator.state_dict(), gpath)
            print('Gnet' + str(epoch + 1) + 'save')
            generator = generator.train()
            dpath = './test/Dnet' + str(epoch + 1) + '.pkl'
            torch.save(discriminator.state_dict(), dpath)
            print('Dnet' + str(epoch + 1) + 'save')

        ### MODIFY LEARNING RATE ###
        if epoch > S_epoch:
            g_scheduler.step(bceloss)

print('time：' + str((time.time() - start_time) / 3600) + 'hours')
print('Finished Training!')
epochs = range(1, len(GLOSS) + 1)
plt.plot(epochs, GLOSS, 'r', linewidth=1.0, label='L_G')
plt.plot(epochs, BCELOSS, 'g', linewidth=1.0, label='L_bce')
plt.plot(epochs, DLOSS, 'y', linewidth=1.0, label='L_D')
plt.plot(epochs, L1LOSS, 'b', linewidth=1.0, label='L_L1')

GLOSS = np.array(GLOSS)
BCELOSS = np.array(BCELOSS)
DLOSS = np.array(DLOSS)
L1LOSS = np.array(L1LOSS)
h = np.vstack((GLOSS,DLOSS))
h = np.vstack((h,BCELOSS))
h = np.vstack((h,L1LOSS))
h=h.T
h = h.tolist()
index_name3= ['GLOSS', 'DLOSS', 'BCELOSS','L1LOSS']
pd.DataFrame(columns=index_name3, index=None, data=h).to_csv('loss.csv')

plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
