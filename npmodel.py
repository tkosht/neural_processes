import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F


class NPMlp(nn.Module):
    def __init__(self, layers=[2, 4, 8, 16, 8, 4, 2], activate=F.relu):
        super().__init__()

        self.layers = layers
        self.activate = activate
        self.fcs = []

        prev_sz = layers[0]
        for hsz in self.layers[1:]:
            fc = nn.Linear(prev_sz, hsz)
            self.fcs.append(fc)
            prev_sz = hsz

        self.device = torch.device("cpu")

    def forward(self, x):
        B, N, D = x.shape
        h = x.reshape(-1, D)
        for fc in self.fcs[:-1]:
            h = fc(h)
            h = self.activate(h)
        h = self.fcs[-1](h)
        return h.reshape(B, N, -1)

    def to(self, device):
        for idx, fc in enumerate(self.fcs):
            self.fcs[idx] = fc.to(device)
        self.device = device
        return super().to(device)


class NPEmbedder(nn.Module):
    def __init__(self, x_size:int, y_size:int, embed_layers=[64, 128], activate=F.relu):
        super().__init__()

        self.x_size = x_size
        self.y_size = y_size
        self.embed_layers = embed_layers
        self.activate = activate

        layers = [x_size + y_size] + embed_layers
        self.emb = NPMlp(layers, activate)
        self.device = torch.device("cpu")

    def h(self, x, y):
        xy = torch.cat([x, y], dim=-1)
        r_vector = self.emb(xy)
        return r_vector

    def encode(self, x, y):
        r_vector = self.h(x, y)
        return r_vector

    def forward(self, x, y):
        return self.encode(x, y)

    def to(self, device):
        self.emb = self.emb.to(device)
        self.device = device
        return super().to(device)


class NPEncoder(nn.Module):
    def __init__(self, hidden_size: int, z_size: int):
        super().__init__()

        assert hidden_size > 0
        assert z_size > 0
        self.hidden_size = hidden_size
        self.latent_size = z_size

        self.a = torch.mean
        self.mu = nn.Linear(self.hidden_size, self.latent_size)
        self.logsgm = nn.Linear(self.hidden_size, self.latent_size)
        self.softplus = torch.nn.Softplus()
        self.device = torch.device("cpu")

    def u(self, r):
        return self.mu(r), self.logsgm(r)

    def encode(self, r_vector):     # (B, N, r_dim)
        r = self.a(r_vector, dim=1)
        mu, logsgm = self.u(r.unsqueeze(1))
        sigma = 0.1 + 0.9 * self.softplus(logsgm)
        return mu, sigma

    def forward(self, r_vector):
        return self.encode(r_vector)

    def to(self, device):
        self.mu= self.mu.to(device)
        self.logsgm= self.logsgm.to(device)
        self.device = device
        return super().to(device)


class NPDecoder(nn.Module):
    def __init__(self, xT_size, z_size, expand_layers=[32, 64], activate=F.relu):
        super().__init__()

        self.xT_size = xT_size
        self.latent_size = z_size
        self.out_size = expand_layers[-1]

        layers = [xT_size + z_size] + expand_layers
        self.dec_yhat = NPMlp(layers, activate)
        self.dec_logsgm = NPMlp(layers, activate)
        self.softplus = torch.nn.Softplus()
        self.device = torch.device("cpu")

    def g(self, x, z):
        xz = torch.cat([x, z], dim=-1)
        yhat = self.dec_yhat(xz)
        logsgm = self.dec_logsgm(xz)
        sgm = 0.1 + 0.9 * self.softplus(logsgm)
        return yhat, sgm

    def decode(self, x, z):
        return self.g(x, z)

    def forward(self, x, z):
        return self.decode(x, z)

    def to(self, device):
        self.dec_yhat = self.dec_yhat.to(device)
        self.dec_logsgm = self.dec_logsgm.to(device)
        self.device = device
        return super().to(device)


class NPModel(nn.Module):
    def __init__(self, xC_size, yC_size, xT_size, yT_size, z_size,
                 embed_layers=[32, 64, 28], expand_layers=[32, 64]):
        super().__init__()

        self.embedder_C = NPEmbedder(xC_size, yC_size, embed_layers)
        self.embedder_T = NPEmbedder(xT_size, yT_size, embed_layers)
        self.encoder = NPEncoder(embed_layers[-1], z_size)
        self.decoder = NPDecoder(xT_size, z_size, expand_layers)
        self.device = torch.device("cpu")

    def encode_context(self, xC, yC):
        rC = self.embedder_C(xC, yC)
        muC, sgmC = self.encoder(rC.to(self.device))
        qC = self.distribution(muC, sgmC)
        zC = qC.sample()
        return zC, qC

    def encode_full(self, xC, yC, xT, yT):
        rC = self.embedder_C(xC, yC)
        rT = self.embedder_T(xT, yT)
        rCT = torch.cat([rC, rT], dim=1)
        muCT, sgmCT = self.encoder(rCT.to(self.device))
        qCT = self.distribution(muCT, sgmCT)
        zCT = qCT.sample()
        return zCT, qCT

    def distribution(self, mu, sgm):
        return torch.distributions.Normal(mu, sgm)

    def decode_context(self, xT, zC):
        T = xT.shape[1]
        yhatT, sgm = self.decoder(xT, zC.repeat(1, T, 1))
        return yhatT, sgm

    def log_likelihood(self, mu, std, D):
        norm = torch.distributions.Normal(mu, std)
        agg_dim = tuple(range(1, len(D.shape)))     # possibly agg_dim == (1, 2)
        return norm.log_prob(D).sum(dim=agg_dim).mean()

    def kl_loss(self, q, p):
        kl = torch.distributions.kl.kl_divergence(q, p)
        kl_div = kl.squeeze(1).sum(-1).mean()
        return kl_div

    def predict(self, xC, yC, xT):
        zC, distC = self.encode_context(xC, yC)
        yhatT, sgm = self.decode_context(xT, zC)
        return yhatT, sgm

    def forward(self, xC, yC, xT, yT):
        zC, qC = self.encode_context(xC, yC)
        zCT, qCT = self.encode_full(xC, yC, xT, yT)
        yhatT, sgm = self.decode_context(xT, zC)
        log_p = self.log_likelihood(yhatT, std=sgm, D=yT)
        kl_div = self.kl_loss(q=qCT, p=qC)
        loss = - log_p + kl_div
        return yhatT, sgm, loss

    def to(self, device):
        self.embedder_C = self.embedder_C.to(device)
        self.embedder_T = self.embedder_T.to(device)
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        self.device = device
        return super().to(device)

