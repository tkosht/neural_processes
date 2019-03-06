import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F


class MLP(nn.Module):
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

    def forward(self, x):
        h = x
        for fc in self.fcs:
            h = fc(h)
            h = self.activate(h)
        return h

class NPEmbedder(nn.Module):
    def __init__(self, x_size:int, y_size:int, embed_layers=[64, 128], activate=F.relu):
        super().__init__()

        self.x_size = x_size
        self.y_size = y_size
        self.embed_layers = embed_layers
        self.activate = activate

        layers = [x_size + y_size] + embed_layers
        self.emb = MLP(layers, activate)

    def h(self, x, y):
        xy = torch.cat([x, y], dim=-1)
        r_vector = self.emb(xy)
        return r_vector

    def encode(self, x, y):
        r_vector = self.h(x, y)
        return r_vector

    def forward(self, x, y):
        return self.encode(x, y)


class NPEncoder(nn.Module):
    def __init__(self, z_size=20):
        super().__init__()

        self.latent_size = z_size
        self.a = torch.mean

        self.mu = nn.Linear(1, self.latent_size)
        self.logsgm = nn.Linear(1, self.latent_size)
        self.softplus = torch.nn.Softplus()

    def u(self, r):
        return self.mu(r), self.logsgm(r)

    def encode(self, r_vector):     # (B, r_dim)
        r = self.a(r_vector, axis=1)
        mu, logsgm = self.u(r)
        sigma = 0.1 + 0.9 * self.softplus(logsgm)
        return mu, sigma

    def forward(self, r_vector):
        return self.encode(r_vector)


class NPDecoder(nn.Module):
    def __init__(self, xT_size, z_size, expand_layers=[32, 64], activate=F.relu):
        super().__init__()

        self.xT_size = xT_size
        self.latent_size = z_size
        self.out_size = expand_layers[-1]

        layers = [xT_size + z_size] + expand_layers
        self.dec_yhat = MLP(layers, activate)
        self.dec_logsgm = MLP(layers, activate)
        self.softplus = torch.nn.Softplus()

    def g(self, x, z):
        xz = torch.cat([x, z], axis=1)
        yhat = self.dec_yhat(xz)
        logsgm = self.dec_logsgm(xz)
        sgm = 0.1 + 0.9 * self.softplus(logsgm)
        return yhat, sgm

    def decode(self, x, z):
        return self.g(x, z)

    def forward(self, x, z):
        return self.decode(x, z)


class NPModel(nn.Module):
    def __init__(self, xC_size, yC_size, xT_size, yT_size, z_size,
                 embed_layers=[32, 64, 28], expand_layers=[32, 64]):
        super().__init__()

        self.embedder_C = NPEmbedder(xC_size, yC_size, embed_layers)
        self.embedder_T = NPEmbedder(xT_size, yT_size, embed_layers)
        self.encoder = NPEncoder(z_size)
        self.decoder = NPDecoder(xT_size, z_size, expand_layers)

    def encode_context(self, xC, yC):
        rC = self.embedder_C(xC, yC)
        muC, sgmC = self.encoder(rC)
        qC = self.distribution(muC, sgmC)
        zC = qC.sample()
        return zC, qC

    def encode_full(self, xC, yC, xT, yT):
        rC = self.embedder_C(xC, yC)
        rT = self.embedder_T(xT, yT)
        rCT = torch.cat([rC, rT], axis=-1)
        muCT, sgmCT = self.encoder(rCT)
        qCT = self.distribution(muCT, sgmCT)
        zCT = qCT.sample()
        return zCT, qCT

    def distribution(self, mu, sgm):
        return torch.distributions.Normal(mu, sgm)

    def decode_context(self, xT, zC):
        yhatT, sgm = self.decoder(xT, zC)
        return yhatT, sgm

    def log_likelihood(self, mu, std, D):
        norm = torch.distributions.Normal(mu, std)
        return norm.log_prob(D).sum(dim=-1).mean()

    def kl_loss(self, q, p):
        return torch.distributions.kl.kl_divergence(q, p)

    def predict(self, xC, yC, xT):
        zC, distC = self.encode_context(xC, yC)
        y_hatT = self.decode_context(xT, zC)
        return y_hatT

    def forward(self, xC, yC, xT, yT):
        zC, qC = self.encode_context(xC, yC)
        zCT, qCT = self.encode_full(xC, yC, xT, yT)
        y_hatT, sgm = self.decode_context(xT, zC)
        log_p = self.log_likelihood(y_hatT, std=sgm, D=yT)
        kl = self.kl_loss(q=qCT, p=qC)
        loss = - log_p + kl
        return y_hatT, loss

