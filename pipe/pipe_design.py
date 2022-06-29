"""
@author: Zongyi Li and Daniel Zhengyu Huang
"""

import torch.nn.functional as F
import matplotlib.pyplot as plt
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
        self.fc0 = nn.Linear(3, self.width)  # input channel is 3: (a(x, y), x, y)

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
        gridx = torch.tensor(np.linspace(0, 10, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(-0.5, 0.5, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


################################################################
# configs
################################################################
INPUT_X = '../data/pipe/Pipe_X.npy'
INPUT_Y = '../data/pipe/Pipe_Y.npy'
OUTPUT_Sigma = '../data/pipe/Pipe_Q.npy'

ntrain = 1000
ntest = 200

batch_size = 20
learning_rate = 0.01

epochs = 501
step_size = 100
gamma = 0.5

modes = 12
width = 32

r1 = 1
r2 = 2
s1 = int(((129 - 1) / r1) + 1)
s2 = int(((129 - 1) / r2) + 1)

################################################################
# load data and data normalization
################################################################

inputX = np.load(INPUT_X)
inputX = torch.tensor(inputX, dtype=torch.float)
inputY = np.load(INPUT_Y)
inputY = torch.tensor(inputY, dtype=torch.float)
input = torch.tensor(inputY, dtype=torch.float)

output = np.load(OUTPUT_Sigma)[:, 0]
output = torch.tensor(output, dtype=torch.float)

x_train = input[:ntrain, ::r1, ::r2][:, :s1, :s2]
y_train = output[:ntrain, ::r1, ::r2][:, :s1, :s2]
x_test = input[-ntest:, ::r1, ::r2][:, :s1, :s2]
y_test = output[-ntest:, ::r1, ::r2][:, :s1, :s2]
x_train = x_train.reshape(ntrain, s1, s2, 1)
x_test = x_test.reshape(ntest, s1, s2, 1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
                                          shuffle=False)
test_loader2 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1,
                                          shuffle=False)


################################################################
# design
################################################################
y = np.linspace(-0.5, 0.5, s2)
y = torch.tensor(y, dtype=torch.float, device="cuda")
y = y.reshape(1, 1, s2, 1).repeat(1, s1, 1, 1)
theta0 = torch.tensor(0, dtype=torch.float, device="cuda").reshape(1,)
# input theta contains 9 U[-1,1] variables

cnx = (s1-1)//2
# This part is independent on theta
N_e = 4
Nx = 2 * cnx + 1  # points
# Construct uniform grid
Lx, Ly = 10.0, 1.0
x = np.linspace(0, Lx, Nx)
# Construct interpolation basis
x_l, x_h = x[0], x[-1]
nx = len(x)
delta_e = (x_h - x_l) / N_e
int_val = np.zeros(nx, dtype=np.int64)
int_val[:] = np.floor((x - x_l) / delta_e)
int_val[-1] = N_e - 1
xi = (x - x_l - delta_e * (int_val)) / delta_e
N1 = (1 - 3 * xi ** 2 + 2 * xi ** 3)  # 0  -> 1
N2 = (xi - 2 * xi ** 2 + xi ** 3)  # 0' -> 1
N3 = (3 * xi ** 2 - 2 * xi ** 3)  # 1  -> 1
N4 = (-xi ** 2 + xi ** 3)  # 1' -> 1

N1 = torch.tensor(N1, dtype=torch.float, device="cuda")
N2 = torch.tensor(N2, dtype=torch.float, device="cuda")
N3 = torch.tensor(N3, dtype=torch.float, device="cuda")
N4 = torch.tensor(N4, dtype=torch.float, device="cuda")

# This part is dependent on theta
def Theta2Mesh(theta):
    theta10 = torch.cat([theta0, torch.tanh(theta)], dim=0)
    deformation = N1 * theta10[2 * int_val] + N2 * theta10[2 * int_val + 1] + N3 * theta10[2 * int_val + 2] + N4 * theta10[2 * int_val + 3]
    return y + deformation.reshape(1, s1, 1, 1).repeat(1, 1, s2, 1)

theta = np.zeros(9,)
theta = torch.tensor(theta, dtype=torch.float, requires_grad=True, device="cuda")

################################################################
# optimize
################################################################
model = torch.load('../model/pipe-u-ep200')
print(count_params(model))
print(model)

epochs = 5001
step_size = 1000
learning_rate = 0.1
gamma = 0.5

optimizer = Adam([theta], lr=learning_rate, weight_decay=0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
myloss = LpLoss(size_average=False)

for ep in range(epochs):
    optimizer.zero_grad()

    x = Theta2Mesh(theta)
    print(x.shape)
    out = model(x)

    loss = torch.mean(out**2)
    loss.backward()

    optimizer.step()
    scheduler.step()

    print(ep, loss.item())

    if ep%step_size==0:
        X = inputX[0, ::r1, ::r2].squeeze().detach().cpu().numpy()
        Y = x[0, :, :, 0].squeeze().detach().cpu().numpy()
        pred = out[0].squeeze().detach().cpu().numpy()

        fig, ax = plt.subplots(nrows=1, figsize=(16, 4))
        ax.pcolormesh(X, Y, pred, shading='gouraud')
        fig.show()

        # torch.save(model, "../model/pipe-u-ep" + str(ep))
