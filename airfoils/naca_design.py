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

output = np.load(OUTPUT_Sigma)[:,3]
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
# design
################################################################
from scipy import optimize

# symmetrical 4-digit NASA airfoil
# the airfoil is in [0,1]
def NACA_shape(x, digit=12):
    return 5 * (digit / 100.0) * (
                0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)

# generate mesh between a and b
# dx0, dx0*r, dx0*r^2 ... dx0*r^{N-2}
# b - a = dx0*(r^{N-1} - 1)/(r - 1)
def GeoSpace(a, b, N, r=-1.0, dx0=-1.0):
    xx = np.linspace(a, b, N)
    if r > 1 or dx0 > 0:
        if r > 1:
            dx0 = (b - a) / ((r ** (N - 1) - 1) / (r - 1))
            dx = dx0
            for i in range(1, N - 1):
                xx[i] = xx[i - 1] + dx
                dx *= r
        else:
            # first use r=1.05 to generate half of the grids
            # then compute r and generate another half of the grids
            f = lambda r: (r - 1) * (b - a) - dx0 * (r ** (N - 1) - 1)
            r = optimize.bisect(f, 1 + 1e-4, 1.5)

            if r > 1.02:
                r = min(r, 1.02)
                dx = dx0
                Nf = 3 * N // 4

                for i in range(1, Nf):
                    xx[i] = xx[i - 1] + dx
                    dx *= r

                a = xx[Nf - 1]
                dx0 = dx

                f = lambda r: (r - 1) * (b - a) - dx0 * (r ** (N - Nf) - 1)
                r = optimize.bisect(f, 1 + 1e-4, 2.0)

                for i in range(Nf, N - 1):
                    xx[i] = xx[i - 1] + dx
                    dx *= r
            else:
                dx = dx0
                for i in range(1, N - 1):
                    xx[i] = xx[i - 1] + dx
                    dx *= r
    return xx

# Nx point on top skin
def NACA_shape_mesh(Nx, method="stretching", ratio=1.0):
    if method == "stretching":
        xx = np.zeros(Nx)
        xx[1:] = GeoSpace(0, 1, Nx - 1, r=ratio ** (1 / (Nx - 3)))
        xx[1] = xx[2] / 4.0
    else:
        print("method : ", method, " is not recognized")

    xx = xx[::-1]
    yy = np.hstack((NACA_shape(xx), -NACA_shape(xx[-2::-1])))
    xx = np.hstack((xx, xx[-2::-1]))
    return xx, yy


# The undeformed box is
# 0.5 - Lx/2  (8)        0.5 - Lx/6  (7)         0.5 + Lx/6  (6)          0.5 + Lx/2  (5)         (y = Ly/2)
#
# 0.5 - Lx/2  (1)        0.5 - Lx/6  (2)         0.5 + Lx/6  (3)          0.5 + Lx/2  (4)         (y = -Ly/2)
#
# basis function at node (i)   is Bᵢ   = Φᵢ(x) Ψ₁(y)    (1 ≤ i ≤ 4)
# basis function at node (i+4) is Bᵢ₊₄ = Φᵢ(x) Ψ₂(y)    (1 ≤ i ≤ 4)
#
# The map is
# (x, y) -> (x, y) + dᵢ Bᵢ(x,  y)
#
def NACA_sdesign(theta, x, y, Lx=1.5, Ly=0.2):
    x1, x2, x3, x4 = 0.5 - Lx / 2, 0.5 - Lx / 6, 0.5 + Lx / 6, 0.5 + Lx / 2
    y1, y2 = - Ly / 2, Ly / 2

    phi1 = (x - x2) * (x - x3) * (x - x4) / ((x1 - x2) * (x1 - x3) * (x1 - x4))
    phi2 = (x - x1) * (x - x3) * (x - x4) / ((x2 - x1) * (x2 - x3) * (x2 - x4))
    phi3 = (x - x1) * (x - x2) * (x - x4) / ((x3 - x1) * (x3 - x2) * (x3 - x4))
    phi4 = (x - x1) * (x - x2) * (x - x3) / ((x4 - x1) * (x4 - x2) * (x4 - x3))

    psi1 = (y - y2) / (y1 - y2)
    psi2 = (y - y1) / (y2 - y1)

    B = torch.stack([phi2 * psi1, phi3 * psi1, phi4 * psi1, phi4 * psi2, phi3 * psi2, phi2 * psi2, phi1 * psi2], dim=0)
    return x, y + torch.matmul(theta, B)

def Cgrid2Cylinder(cnx1, cnx2, cny, Cgrid, Cylinder):
    # Cgrid
    nx1, nx2, ny = cnx1 + 1, cnx2 + 1, cny + 1

    for j in range(cny + 1):
        if j == 0:
            Cylinder[0:cnx1 + cnx2, j] = Cgrid[0:cnx1 + cnx2]
            Cylinder[cnx1 + cnx2:2 * nx1 + cnx2 - 1, j] = Cylinder[cnx1::-1, j]
        else:
            Cylinder[:, j] = Cgrid[(j - 1) * (2 * cnx1 + cnx2 + 1) + cnx1 + cnx2: (j - 1) * (
                        2 * cnx1 + cnx2 + 1) + cnx1 + cnx2 + 1 + 2 * cnx1 + cnx2]

def Cylinder2Cgrid(cnx1, cnx2, cny, Cylinder, Cgrid):
    # Cylinder,
    nx1, nx2, ny = cnx1 + 1, cnx2 + 1, cny + 1

    for j in range(cny + 1):
        if j == 0:
            Cgrid[0:cnx1 + cnx2] = Cylinder[0:cnx1 + cnx2, j]
        else:
            Cgrid[(j - 1) * (2 * cnx1 + cnx2 + 1) + cnx1 + cnx2: (j - 1) * (
                        2 * cnx1 + cnx2 + 1) + cnx1 + cnx2 + 1 + 2 * cnx1 + cnx2] = Cylinder[:, j]

# c: number of cells
# cnx1 C mesh behind trailing edge
# cnx2 C mesh around airfoil
# cny radial direction
#
# The airfoil is in [0,1]
# R: radius of C mesh
# L: the right end of the mesh
# the bounding box of the mesh is [Rc-R, L], [-R, R]
#
# dy0, vertical mesh size
cnx1=50
dy0=2.0 / 120.0
cnx2=120
cny=50
R=40
Rc=1.0
L=40
cnx = 2 * cnx1 + cnx2
nx1, nx2, ny = cnx1 + 1, cnx2 + 1, cny + 1  # points
nnodes = (2 * nx1 + cnx2 - 1) * cny + (nx1 + cnx2 - 1)

xx_airfoil, yy_airfoil = NACA_shape_mesh(cnx2 // 2 + 1, method="stretching")
xx_inner = GeoSpace(0, 1, nx1, dx0=np.sqrt((xx_airfoil[0] - xx_airfoil[1]) ** 2 + (yy_airfoil[0] - yy_airfoil[1]) ** 2) / (L - 1))
xx_outer = GeoSpace(Rc, L, nx1)
wy = GeoSpace(0, 1, ny, dx0=dy0 / R)

xx_airfoil = torch.tensor(xx_airfoil, device='cuda', dtype=torch.float)
yy_airfoil = torch.tensor(yy_airfoil, device='cuda', dtype=torch.float)
xx_inner = torch.tensor(xx_inner, device='cuda', dtype=torch.float)
xx_outer = torch.tensor(xx_outer, device='cuda', dtype=torch.float)
wy = torch.tensor(wy, device='cuda', dtype=torch.float)

def Theta2Mesh(theta, xx_airfoil=xx_airfoil, yy_airfoil=yy_airfoil, xx_inner=xx_inner, xx_outer=xx_outer):
    # assert (len(theta) == 8 and theta[0] == 0.0)
    assert (len(theta) == 7)

    xx_airfoil, yy_airfoil = NACA_sdesign(theta, xx_airfoil, yy_airfoil)

    xy_inner = torch.zeros((2 * nx1 + cnx2 - 1, 2), device='cuda', dtype=torch.float)
    xy_outer = torch.zeros((2 * nx1 + cnx2 - 1, 2), device='cuda', dtype=torch.float)
    # top flat
    xy_inner[:nx1, 0] = torch.flip(xx_airfoil[0] * (1 - xx_inner) + L * xx_inner, dims=[0])
    xy_inner[:nx1, 1] = torch.flip(yy_airfoil[0] * (1 - xx_inner), dims=[0])
    xy_outer[:nx1, 0] = torch.flip(xx_outer, dims=[0])
    xy_outer[:nx1, 1] = R

    # airfoil
    xy_inner[nx1 - 1:nx1 + cnx2, 0] = xx_airfoil
    xy_inner[nx1 - 1:nx1 + cnx2, 1] = yy_airfoil

    θθ = torch.linspace(np.pi / 2, 3 * np.pi / 2, nx2)
    xy_outer[nx1 - 1:nx1 + cnx2, 0] = R * torch.cos(θθ) + Rc
    xy_outer[nx1 - 1:nx1 + cnx2, 1] = R * torch.sin(θθ)
    # bottom flat
    xy_inner[nx1 + cnx2 - 1:2 * nx1 + cnx2 - 1, 0] = torch.flip(xy_inner[:nx1, 0], dims=[0])
    xy_inner[nx1 + cnx2 - 1:2 * nx1 + cnx2 - 1, 1] = torch.flip(xy_inner[:nx1, 1], dims=[0])
    xy_outer[nx1 + cnx2 - 1:2 * nx1 + cnx2 - 1, 0] = xx_outer
    xy_outer[nx1 + cnx2 - 1:2 * nx1 + cnx2 - 1, 1] = -R

    # Construct Cylinder grid
    xx_Cylinder = torch.outer(xy_inner[:, 0], 1 - wy) + torch.outer(xy_outer[:, 0], wy)
    yy_Cylinder = torch.outer(xy_inner[:, 1], 1 - wy) + torch.outer(xy_outer[:, 1], wy)
    out = torch.stack([xx_Cylinder, yy_Cylinder], dim=-1).unsqueeze(0)
    return out, xx_Cylinder, yy_Cylinder

def compute_F(XC, YC, p, cnx1=50, cnx2=120, cny=50):
    p = p.squeeze()
    xx, yy, p = XC[cnx1:-cnx1, 0], YC[cnx1:-cnx1, 0], p[cnx1:-cnx1, 0]

    drag = torch.matmul(yy[0:cnx2]-yy[1:cnx2+1], (p[0:cnx2] + p[1:cnx2+1])/2.0)
    lift = torch.matmul(xx[1:cnx2+1]-xx[0:cnx2], (p[0:cnx2] + p[1:cnx2+1])/2.0)
    return drag, lift

################################################################
# inverse optimization
################################################################
model = torch.load(PATH + '/model/naca_p_w32_500')
print(count_params(model))

learning_rate = 0.00001
epochs = 5001
step_size = 1000
gamma = 0.5
theta = torch.zeros(7, dtype=torch.float, requires_grad=True, device="cuda")
optimizer = Adam([theta], lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = LpLoss(size_average=False)

for ep in range(epochs):
    model.train()
    t1 = default_timer()

    x, XC, YC = Theta2Mesh(theta)
    out = model(x)
    drag, lift = compute_F(XC, YC, out)
    loss = ((drag/lift) ** 2)
    reg = torch.norm(theta)
    loss_sum = loss + 1*reg
    loss_sum.backward()

    optimizer.step()
    scheduler.step()

    t2 = default_timer()
    print(ep, t2 - t1, drag.item(), lift.item(), reg.item(), loss.item())

    if ep%step_size==0:
        print(theta.detach().cpu().numpy())
        ind = -1
        X = x[ind, :, :, 0].squeeze().detach().cpu().numpy()
        Y = x[ind, :, :, 1].squeeze().detach().cpu().numpy()
        pred = out[ind].squeeze().detach().cpu().numpy()
        nx = 40//r1
        ny = 20//r2
        X_small = X[nx:-nx, :ny]
        Y_small = Y[nx:-nx, :ny]
        pred_small = pred[nx:-nx, :ny]
        fig, ax = plt.subplots(ncols=2,  figsize=(16, 8))
        ax[0].pcolormesh(X, Y, pred, shading='gouraud')
        ax[1].pcolormesh(X_small, Y_small, pred_small, shading='gouraud')
        fig.show()

