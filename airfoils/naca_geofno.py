"""
@author: Zongyi Li and Daniel Zhengyu Huang
"""

import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
from utilities3 import *
from Adam import Adam

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 8  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(4, self.width)  # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


################################################################
# configs
################################################################
PATH = ".."
INPUT_X = PATH+'/data/naca/NACA_Cylinder_X.npy'
INPUT_Y = PATH+'/data/naca/NACA_Cylinder_Y.npy'
OUTPUT_Sigma = PATH+'/data/naca/NACA_Cylinder_Q.npy'

ntrain = 1000
ntest = 200

batch_size = 20
learning_rate = 0.001

epochs = 501
step_size = 100
gamma = 0.5

modes = 12
width = 32

r1 = 1
r2 = 1
s1 = int(((221 - 1) / r1) + 1)
s2 = int(((51 - 1) / r2) + 1)

################################################################
# load data and data normalization
################################################################
inputX = np.load(INPUT_X)
inputX = torch.tensor(inputX, dtype=torch.float)
inputY = np.load(INPUT_Y)
inputY = torch.tensor(inputY, dtype=torch.float)
input = torch.stack([inputX, inputY], dim=-1)

output = np.load(OUTPUT_Sigma)[:, 4]
output = torch.tensor(output, dtype=torch.float)
print(input.shape, output.shape)

x_train = input[:ntrain, ::r1, ::r2][:, :s1, :s2]
y_train = output[:ntrain, ::r1, ::r2][:, :s1, :s2]
x_test = input[ntrain:ntrain+ntest, ::r1, ::r2][:, :s1, :s2]
y_test = output[ntrain:ntrain+ntest, ::r1, ::r2][:, :s1, :s2]
x_train = x_train.reshape(ntrain, s1, s2, 2)
x_test = x_test.reshape(ntest, s1, s2, 2)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
                                          shuffle=False)
test_loader2 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1,
                                          shuffle=False)


################################################################
# training and evaluation
################################################################
model = FNO2d(modes*2, modes, width).cuda()
print(count_params(model))

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = LpLoss(size_average=False)

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x)

        loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        loss.backward()

        optimizer.step()
        train_l2 += loss.item()

    scheduler.step()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x)
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

    train_l2 /= ntrain
    test_l2 /= ntest

    t2 = default_timer()
    print(ep, t2 - t1, train_l2, test_l2)

    # plot
    if ep%step_size==0:
        # torch.save(model, '../model/naca_plain_model_'+str(ep))

        ind = -1
        X = x[ind, :, :, 0].squeeze().detach().cpu().numpy()
        Y = x[ind, :, :, 1].squeeze().detach().cpu().numpy()
        truth = y[ind].squeeze().detach().cpu().numpy()
        pred = out[ind].squeeze().detach().cpu().numpy()
        nx = 40//r1
        ny = 20//r2
        X_small = X[nx:-nx, :ny]
        Y_small = Y[nx:-nx, :ny]
        truth_small = truth[nx:-nx, :ny]
        pred_small = pred[nx:-nx, :ny]

        fig, ax = plt.subplots(nrows=3, ncols=2,  figsize=(16, 16))
        ax[0,0].pcolormesh(X, Y, truth, shading='gouraud')
        ax[1,0].pcolormesh(X, Y, pred, shading='gouraud')
        ax[2,0].pcolormesh(X, Y, pred-truth, shading='gouraud')
        ax[0,1].pcolormesh(X_small, Y_small, truth_small, shading='gouraud')
        ax[1,1].pcolormesh(X_small, Y_small, pred_small, shading='gouraud')
        ax[2,1].pcolormesh(X_small, Y_small, np.abs(pred_small-truth_small), shading='gouraud')
        fig.show()


