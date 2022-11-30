import torch
import torch.nn.functional as F
import torch.nn as nn

device = torch.device('cuda:1')
torch.manual_seed(12)

class UNet1(torch.nn.Module):
    def __init__(self, C):
        super(UNet1, self).__init__()

        self.conv1 = nn.Conv2d(C, 128, stride=1, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128*2, C, stride=1, kernel_size=3, padding=1)
        self.conv21 = nn.Conv2d(128,128,stride=1, kernel_size=3,padding=1)
        self.conv24 = nn.Conv2d(128*2,128,stride=1, kernel_size=3,padding=1)
        self.conv31 = nn.Conv2d(128,128,stride=1, kernel_size=3,padding=1)
        self.conv32 = nn.Conv2d(128, 128, stride=1, kernel_size=3, padding=1)
        self.relu = nn.PReLU()
    def forward(self, x):

        out11 = self.relu(self.conv1(x))
        out21 = self.relu(self.conv21(out11))
        out31 = self.relu(self.conv31(out21))
        out32 = self.relu(self.conv32(out31))
        out24 = self.relu(self.conv24(torch.cat((out21, out32), 1)))
        out = self.conv6(torch.cat((out11, out24), 1))
        return out

class UNet2(torch.nn.Module):
    def __init__(self, C):
        super(UNet2, self).__init__()

        self.conv1 = nn.Conv2d(C, 128, stride=1, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128*2, C, stride=1, kernel_size=3, padding=1)
        self.conv21 = nn.Conv2d(128,128,stride=2, kernel_size=3,padding=1)
        self.conv24 = nn.ConvTranspose2d(128*2, 128, stride=2, kernel_size=4, padding=1)
        self.conv31 = nn.Conv2d(128,128,stride=2, kernel_size=3,padding=1)
        self.conv32 = nn.ConvTranspose2d(128,128,stride=2, kernel_size=4,padding=1)
        self.relu = nn.PReLU()
    def forward(self, x):

        out11 = self.relu(self.conv1(x))
        out21 = self.relu(self.conv21(out11))
        out31 = self.relu(self.conv31(out21))
        out32 = self.relu(self.conv32(out31))
        out24 = self.relu(self.conv24(torch.cat((out21, out32), 1)))
        out = self.conv6(torch.cat((out11, out24), 1))
        return out

class ProxNet(torch.nn.Module):
    def __init__(self, Cin):
        super(ProxNet, self).__init__()
        self.high_f = UNet1(Cin)
        self.low_f = UNet2(Cin)
        self.relu = nn.PReLU()
        self.downsample = nn.UpsamplingBilinear2d(scale_factor=0.25)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=4)
    def forward(self, x):
        out = self.downsample(x)
        low_x = self.upsample(out)
        high_x = x - low_x
        high_x = self.high_f(high_x)
        low_x = self.low_f(low_x)
        out  = high_x + low_x
        return out

class Up_sample(torch.nn.Module):
    def __init__(self,Cin):
        super(Up_sample, self).__init__()
        self.denet1 = nn.ConvTranspose2d(Cin, Cin, stride=2, kernel_size=4, padding=1)
        self.denet2 = nn.ConvTranspose2d(Cin, Cin, stride=2, kernel_size=4, padding=1)
        self.relu = nn.PReLU()
    def forward(self, x):
        out = self.relu(self.denet1(x))
        out = self.denet2(out)
        return out


class Down_sample(torch.nn.Module):
    def __init__(self,Cin):
        super(Down_sample, self).__init__()
        self.ennet1 = nn.Conv2d(Cin, Cin, stride=2, kernel_size=3, padding=1)
        self.ennet2 = nn.Conv2d(Cin, Cin, stride=2, kernel_size=3, padding=1)
        self.relu = nn.PReLU()
    def forward(self, x):
        out = self.relu(self.ennet1(x))
        out = self.ennet2(out)
        return out


class NBNet(torch.nn.Module):
    def __init__(self, Cin,Cout, N, ratio):
        super(NBNet, self).__init__()
        self.u = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).to(device)
        self.lamba = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).to(device)
        self.lamba1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).to(device)
        self.R = nn.Conv2d(Cin, Cout, 1)
        self.RT = nn.Conv2d(Cout, Cin, 1)
        self.super = nn.Conv2d(Cout, Cin, 1)
        self.RTRE = nn.Conv2d(Cin, Cin, 1)
        self.HTHE = nn.Conv2d(Cin, Cin, 1)
        self.HT = Up_sample(Cin)
        self.H = Down_sample(Cin)
        self.M = ProxNet(Cin)
        self.L = ProxNet(Cin)
        self.N = N
        self.ratio = ratio
        self.lamba1.fill_(0.5)

    def forward(self, X, Y):
        X_hat = F.interpolate(X, scale_factor=self.ratio, mode='bicubic', align_corners=False)

        B = self.RT(Y - self.R(X_hat))
        X_tilde = self.RTRE(B)
        for i in range(self.N - 1):
            M = self.M(X_tilde)
            B = self.RT(Y - self.R(X_hat)) + self.u * M
            X_tilde = (self.RTRE(B))
        out11 = X_tilde + X_hat
        out12 = self.R(out11)

        Y_hat = self.super(Y)
        D = self.HT(X-self.H(Y_hat))
        Y_title = self.HTHE(D)
        for i in range (self.N-1):
            L = self.L(Y_title)
            D = self.HT(X-self.H(Y_hat)) + self.lamba * L
            Y_title = self.HTHE(D)
        out21 = Y_title + Y_hat
        out22 = self.H(out21)
        out = out11*self.lamba1 + out21*(1-self.lamba1)
        return out, out12, out22

# a = torch.randn(1,102,40,40).to(device)
# b = torch.randn(1,4,160,160).to(device)
# net = NBNet(102,4,4,4).to(device)
# c,d ,e= net(a,b)
# print(c.shape,d.shape,e.shape)

# a = torch.randn(1,102,102)
# b = torch.randn(1,102,160)
# c = torch.bmm(a,b)
# print(c.shape)

# a = torch.randn(1,102,160,160)
# net = ProxNet1(102)
# b = net(a)
# print(b.shape)










