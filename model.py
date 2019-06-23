import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Discriminator(nn.Module):
    def __init__(self, dropout_drop=0.5):
        super(Discriminator, self).__init__()
        negative_slope = 0.03
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=31, stride=2, padding=15)
        self.vbn1 = nn.BatchNorm1d(32)
        self.lrelu1 = nn.LeakyReLU(negative_slope)
        self.conv2 = nn.Conv1d(32, 64, 31, 2, 15)
        self.vbn2 = nn.BatchNorm1d(64)
        self.lrelu2 = nn.LeakyReLU(negative_slope)
        self.conv3 = nn.Conv1d(64, 64, 31, 2, 15)
        self.dropout1 = nn.Dropout(dropout_drop)
        self.vbn3 = nn.BatchNorm1d(64)
        self.lrelu3 = nn.LeakyReLU(negative_slope)
        self.conv4 = nn.Conv1d(64, 128, 31, 2, 15)
        self.vbn4 = nn.BatchNorm1d(128)
        self.lrelu4 = nn.LeakyReLU(negative_slope)
        self.conv5 = nn.Conv1d(128, 128, 31, 2, 15)
        self.vbn5 = nn.BatchNorm1d(128)
        self.lrelu5 = nn.LeakyReLU(negative_slope)
        self.conv6 = nn.Conv1d(128, 256, 31, 2, 15)
        self.dropout2 = nn.Dropout(dropout_drop)
        self.vbn6 = nn.BatchNorm1d(256)
        self.lrelu6 = nn.LeakyReLU(negative_slope)
        self.conv7 = nn.Conv1d(256, 256, 31, 2, 15)
        self.vbn7 = nn.BatchNorm1d(256)
        self.lrelu7 = nn.LeakyReLU(negative_slope)
        self.conv8 = nn.Conv1d(256, 512, 31, 2, 15)
        self.vbn8 = nn.BatchNorm1d(512)
        self.lrelu8 = nn.LeakyReLU(negative_slope)
        self.conv9 = nn.Conv1d(512, 512, 31, 2, 15)
        self.dropout3 = nn.Dropout(dropout_drop)
        self.vbn9 = nn.BatchNorm1d(512)
        self.lrelu9 = nn.LeakyReLU(negative_slope)
        self.conv10 = nn.Conv1d(512, 1024, 31, 2, 15)
        self.vbn10 = nn.BatchNorm1d(1024)
        self.lrelu10 = nn.LeakyReLU(negative_slope)
        self.conv11 = nn.Conv1d(1024, 2048, 31, 2, 15)
        self.vbn11 = nn.BatchNorm1d(2048)
        self.lrelu11 = nn.LeakyReLU(negative_slope)
        self.conv_final = nn.Conv1d(2048, 1, kernel_size=1, stride=1)
        self.lrelu_final = nn.LeakyReLU(negative_slope)
        self.fully_connected = nn.Linear(in_features=8, out_features=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        x = self.conv1(x)
        x = self.vbn1(x)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.vbn2(x)
        x = self.lrelu2(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        x = self.vbn3(x)
        x = self.lrelu3(x)
        x = self.conv4(x)
        x = self.vbn4(x)
        x = self.lrelu4(x)
        x = self.conv5(x)
        x = self.vbn5(x)
        x = self.lrelu5(x)
        x = self.conv6(x)
        x = self.dropout2(x)
        x = self.vbn6(x)
        x = self.lrelu6(x)
        x = self.conv7(x)
        x = self.vbn7(x)
        x = self.lrelu7(x)
        x = self.conv8(x)
        x = self.vbn8(x)
        x = self.lrelu8(x)
        x = self.conv9(x)
        x = self.dropout3(x)
        x = self.vbn9(x)
        x = self.lrelu9(x)
        x = self.conv10(x)
        x = self.vbn10(x)
        x = self.lrelu10(x)
        x = self.conv11(x)
        x = self.vbn11(x)
        x = self.lrelu11(x)
        x = self.conv_final(x)
        x = self.lrelu_final(x)
        x = torch.squeeze(x)
        x = self.fully_connected(x)
        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.enc1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=32, stride=2,
                              padding=15)  #8192
        self.enc1_nl = nn.PReLU()
        self.enc2 = nn.Conv1d(16, 32, 32, 2, 15)  #4096
        self.enc2_nl = nn.PReLU()
        self.enc3 = nn.Conv1d(32, 32, 32, 2, 15)  #2048
        self.enc3_nl = nn.PReLU()
        self.enc4 = nn.Conv1d(32, 64, 32, 2, 15)  #1024
        self.enc4_nl = nn.PReLU()
        self.enc5 = nn.Conv1d(64, 64, 32, 2, 15)  #512
        self.enc5_nl = nn.PReLU()
        self.enc6 = nn.Conv1d(64, 128, 32, 2, 15)  #256
        self.enc6_nl = nn.PReLU()
        self.enc7 = nn.Conv1d(128, 128, 32, 2, 15)  #128
        self.enc7_nl = nn.PReLU()
        self.enc8 = nn.Conv1d(128, 256, 32, 2, 15)  #64
        self.enc8_nl = nn.PReLU()

        self.dec7 = nn.ConvTranspose1d(256, 128, 32, 2, 15)  #128
        self.dec7_nl = nn.PReLU()
        self.dec6 = nn.ConvTranspose1d(256, 128, 32, 2, 15)  #256
        self.dec6_nl = nn.PReLU()
        self.dec5 = nn.ConvTranspose1d(256, 64, 32, 2, 15)  #512
        self.dec5_nl = nn.PReLU()
        self.dec4 = nn.ConvTranspose1d(128, 64, 32, 2, 15)  #1024
        self.dec4_nl = nn.PReLU()
        self.dec3 = nn.ConvTranspose1d(128, 32, 32, 2, 15)  #2048
        self.dec3_nl = nn.PReLU()
        self.dec2 = nn.ConvTranspose1d(64, 32, 32, 2, 15)  #4096
        self.dec2_nl = nn.PReLU()
        self.dec1 = nn.ConvTranspose1d(64, 16, 32, 2, 15)  #8192
        self.dec1_nl = nn.PReLU()
        self.dec_final = nn.ConvTranspose1d(32, 1, 32, 2, 15)  #16384
        self.dec_tanh = nn.Tanh()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.enc1_nl(e1))
        e3 = self.enc3(self.enc2_nl(e2))
        e4 = self.enc4(self.enc3_nl(e3))
        e5 = self.enc5(self.enc4_nl(e4))
        e6 = self.enc6(self.enc5_nl(e5))
        e7 = self.enc7(self.enc6_nl(e6))
        e8 = self.enc8(self.enc7_nl(e7))

        c = self.enc8_nl(e8)

        d7 = self.dec7(c)
        d7_c = self.dec7_nl(torch.cat((d7, e7), dim=1))
        d6 = self.dec6(d7_c)
        d6_c = self.dec6_nl(torch.cat((d6, e6), dim=1))
        d5 = self.dec5(d6_c)
        d5_c = self.dec5_nl(torch.cat((d5, e5), dim=1))
        d4 = self.dec4(d5_c)
        d4_c = self.dec4_nl(torch.cat((d4, e4), dim=1))
        d3 = self.dec3(d4_c)
        d3_c = self.dec3_nl(torch.cat((d3, e3), dim=1))
        d2 = self.dec2(d3_c)
        d2_c = self.dec2_nl(torch.cat((d2, e2), dim=1))
        d1 = self.dec1(d2_c)
        d1_c = self.dec1_nl(torch.cat((d1, e1), dim=1))
        out = self.dec_tanh(self.dec_final(d1_c))
        return out

class Steganalysis(nn.Module):  #Chen network

    def __init__(self):
        super(Steganalysis, self).__init__()
        K = np.array([[-1, 2, -1]], dtype=float)
        K = K.reshape(1, 1, 3).astype(np.float32)

        K = torch.from_numpy(K)
        self.K = nn.Parameter(data=K, requires_grad=False)
        self.conv2 = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=5,stride=1,padding=2)
        self.conv3 = nn.Conv1d(1,8,1,1,0)
        self.conv4 = nn.Conv1d(8,8,3,2,2)

        self.conv5 = nn.Conv1d(8, 8, 5, 1, 2)
        self.conv6 = nn.Conv1d(8, 16, 1, 1, 0)
        self.conv7 = nn.Conv1d(16, 16, 3, 2, 2)

        self.conv8 = nn.Conv1d(16, 16, 5, 1, 2)
        self.T1 = nn.ReLU()
        self.conv9 = nn.Conv1d(16, 32, 1, 1, 0)
        self.T2 = nn.ReLU()
        self.Max1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.conv10 = nn.Conv1d(32, 32, 5, 1, 2)
        self.T3 = nn.ReLU()
        self.conv11 = nn.Conv1d(32, 64, 1, 1, 0)
        self.T4 = nn.ReLU()
        self.Max2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.conv12 = nn.Conv1d(64, 64, 5, 1, 2)
        self.T5 = nn.ReLU()
        self.conv13 = nn.Conv1d(64, 128, 1, 1, 0)
        self.T6 = nn.ReLU()
        self.Max3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.conv14 = nn.Conv1d(128, 128, 5, 1, 2)
        self.T7 = nn.ReLU()
        self.conv15 = nn.Conv1d(128, 256, 1, 1, 0)
        self.T8 = nn.ReLU()
        self.Max4 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.conv16 = nn.Conv1d(256, 256, 5, 1, 2)
        self.T9 = nn.ReLU()
        self.conv17 = nn.Conv1d(256, 512, 1, 1, 0)
        self.T10 = nn.ReLU()
        self.avg = nn.AvgPool1d(kernel_size=256,stride=256)

        self.fully_connected = nn.Linear(in_features=512, out_features=1)
        self.finall = nn.Sigmoid()
        self.init_weights()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.zeros_(m.bias)

    def forward(self,x):
        x = F.conv1d(x, self.K, padding=1)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        x = self.conv8(x)
        x = self.T1(x)
        x = self.conv9(x)
        x = self.T2(x)
        x = self.Max1(x)

        x = self.conv10(x)
        x = self.T3(x)
        x = self.conv11(x)
        x = self.T4(x)
        x = self.Max2(x)

        x = self.conv12(x)
        x = self.T5(x)
        x = self.conv13(x)
        x = self.T6(x)
        x = self.Max3(x)

        x = self.conv14(x)
        x = self.T7(x)
        x = self.conv15(x)
        x = self.T8(x)
        x = self.Max4(x)

        x = self.conv16(x)
        x = self.T9(x)
        x = self.conv17(x)
        x = self.T10(x)
        x = self.avg(x)

        x = torch.squeeze(x)
        x = self.fully_connected(x)
        x = self.finall(x)
        return x