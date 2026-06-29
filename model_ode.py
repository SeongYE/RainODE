import torch
import torch.nn as nn
from torchdiffeq import odeint
from modules import ConvSC, VANSubBlock

def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if reverse:
        return list(reversed(samplings[:N]))
    else:
        return samplings[:N]
    
class Encoder(nn.Module):
    def __init__(self, C_in, C_hid, N_S, spatio_kernel):
        super().__init__()
        samplings = sampling_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, spatio_kernel, downsampling=samplings[0]),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s)
              for s in samplings[1:]]
        )

    def forward(self, x):
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class Decoder(nn.Module):
    def __init__(self, C_hid, C_out, N_S, spatio_kernel):
        super().__init__()
        samplings = sampling_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s)
              for s in samplings]
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid):
        for block in self.dec:
            hid = block(hid)
        return self.readout(hid)



class VAN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = VANSubBlock(
            in_channels,
            mlp_ratio=8.,
            drop=0.0,
            drop_path=0.0,
            act_layer=nn.GELU
        )
        self.reduction = (
            nn.Identity() if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x):
        return self.reduction(self.block(x))


class Translator(nn.Module):
    def __init__(self, channel_in, channel_hid, N2):
        super().__init__()
        layers = [VAN(channel_in, channel_hid)]
        for _ in range(N2 - 2):
            layers.append(VAN(channel_hid, channel_hid))
        layers.append(VAN(channel_hid, channel_in))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)
        x = self.net(x)
        return x.reshape(B, T, C, H, W)


class ODEDynamics(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.net = VAN(2 * C, C)

    def forward(self, t, z, cond):
        return self.net(torch.cat([z, cond], dim=1))


class ODEWrapper(nn.Module):
    def __init__(self, func, cond):
        super().__init__()
        self.func = func
        self.cond = cond

    def forward(self, t, z):
        return self.func(t, z, self.cond)


class SimvpFlowODE(nn.Module):
    def __init__(self, shape_in, hid_S=16, hid_T=256, N_S=4, N_T=8):
        super().__init__()
        T, C, H, W = shape_in

        self.enc = Encoder(C, hid_S, N_S, 3)
        self.translator = Translator(T * hid_S, hid_T, N_T)
        self.dec = Decoder(hid_S, C, N_S, 3)

        self.ode_func = ODEDynamics(hid_S)

    def forward(self, x_raw):
        B, T, C, H, W = x_raw.shape

        x = x_raw.reshape(B * T, C, H, W)
        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)

        hid_teacher = self.translator(z)           # [B,T,C_,H,W]

        h0 = hid_teacher[:, 0]                     # t0
        hT = hid_teacher[:, -1]                    # condition

        t_interval = 12
        t_span = torch.linspace(0, 1, t_interval, device=x.device)
        ode = ODEWrapper(self.ode_func, hT)

        hid_ode = odeint(
            ode, h0, t_span, method="rk4"
        )                                          # [T,B,C,H,W]

        hid_ode = hid_ode.permute(1, 0, 2, 3, 4)   # [B,T,C,H,W]
        Y_teacher = self.dec(
            hid_teacher.reshape(B * T, C_, H_, W_)
        ).reshape(B, T, C, H, W)

        Y_ode = self.dec(
            hid_ode.reshape(B * t_interval, C_, H_, W_)
        ).reshape(B, t_interval, C, H, W)


        return Y_teacher, Y_ode

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimvpFlowODE(shape_in=[6, 1, 128, 128]).to(device)

    x = torch.randn(2, 6, 1, 128, 128).to(device)

    with torch.no_grad():
        Y_teacher, Y_ode = model(x)

    print("Input shape     :", x.shape)
    print("Y_teacher shape :", Y_teacher.shape)
    print("Y_ode shape     :", Y_ode.shape)


