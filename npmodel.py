import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F


def weight_initialize(w):
    return torch.nn.init.xavier_uniform_(w)
    # return torch.nn.init.kaiming_uniform_(w)
    # return torch.nn.init.xavier_normal_(w)


class NPMlp(nn.Module):
    def __init__(self, layers=[2, 4, 8, 16, 8, 4, 2], activate=F.relu):
        super().__init__()

        self.layers = layers
        self.activate = activate
        self.fcs = []

        prev_sz = layers[0]
        for idx, hsz in enumerate(self.layers[1:]):
            fc = nn.Linear(prev_sz, hsz)
            weight_initialize(fc.weight)
            self.fcs.append(fc)
            setattr(self, f"fc_{idx:03d}", fc)
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
    def __init__(self, hidden_size: int, z_size: int,
                 encoder_layers=[2, 4, 8], activate=F.relu):
        super().__init__()

        assert hidden_size > 0
        assert z_size > 0
        self.hidden_size = hidden_size
        self.latent_size = z_size

        self.a = torch.mean
        layers = [hidden_size] + encoder_layers
        self.enc = NPMlp(layers, activate)
        self.mu = nn.Linear(layers[-1], self.latent_size)
        self.logsgm = nn.Linear(layers[-1], self.latent_size)
        self.softplus = torch.nn.Softplus()
        self.device = torch.device("cpu")

        weight_initialize(self.mu.weight)
        weight_initialize(self.logsgm.weight)

    @staticmethod
    def distribution(mu, sgm):
        return torch.distributions.Normal(mu, sgm)

    def u(self, r):
        h = self.enc(r)
        return self.mu(h), self.logsgm(h)

    def encode(self, r_vector):     # (B, N, r_dim)
        r = self.a(r_vector, dim=1) # r:(B, 1, r_dim)
        mu, logsgm = self.u(r.unsqueeze(1))
        sgm = 0.1 + 0.9 * self.softplus(logsgm)
        q = self.distribution(mu, sgm)
        return q

    def forward(self, r_vector):
        return self.encode(r_vector)

    def to(self, device):
        self.enc = self.enc.to(device)
        self.mu = self.mu.to(device)
        self.logsgm = self.logsgm.to(device)
        self.device = device
        return super().to(device)


class NPDecoder(nn.Module):
    def __init__(self, xT_size, z_size, decoder_layers=[32, 64], activate=F.relu):
        super().__init__()

        self.xT_size = xT_size
        self.latent_size = z_size
        self.out_size = decoder_layers[-1]

        layers = [xT_size + z_size] + decoder_layers
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
    def __init__(self, xC_size, yC_size, xT_size, yT_size, z_size, use_deterministic_path=False,
                 embed_layers=[32, 64, 28], latent_encoder_layers=[28, 32],
                 deterministic_layers=[18, 33], decoder_layers=[32, 64]):
        super().__init__()

        self.use_deterministic_path = use_deterministic_path
        self.embedder_C = NPEmbedder(xC_size, yC_size, embed_layers)
        self.embedder_T = NPEmbedder(xT_size, yT_size, embed_layers)
        self.latent_encoder = NPEncoder(embed_layers[-1], z_size, latent_encoder_layers)
        if use_deterministic_path:
            # self.deterministic_encoder = NPEmbedder(xC_size, yC_size, deterministic_layers)
            # z_size += deterministic_layers[-1]
            self.deterministic_encoder = self.embedder_C
            z_size += embed_layers[-1]
        self.decoder = NPDecoder(xT_size, z_size, decoder_layers)
        self.device = torch.device("cpu")

    def encode_context(self, xC, yC):
        rC = self.embedder_C(xC, yC)
        qC = self.latent_encoder(rC.to(self.device))
        return qC

    def get_encoded_sample(self, xC, yC, qC, do_randomize=True):
        if do_randomize:
            zC = qC.rsample()   # must get reparameterized sample
        else:
            zC = qC.mean
        if self.use_deterministic_path:
            zD = self.deterministic_encoder(xC, yC)
            zD = zD.mean(dim=1).unsqueeze(1)
            zC = torch.cat([zC, zD], dim=-1)
        return zC

    def encode_full(self, xC, yC, xT, yT):
        rC = self.embedder_C(xC, yC)
        rT = self.embedder_T(xT, yT)
        rCT = torch.cat([rC, rT], dim=1)
        qCT = self.latent_encoder(rCT.to(self.device))
        return qCT

    def decode_context(self, xT, zC):
        T = xT.shape[1]
        yhatT, sgm = self.decoder(xT, zC.repeat(1, T, 1))
        return yhatT, sgm

    @staticmethod
    def log_likelihood(mu, std, D):
        norm = torch.distributions.Normal(mu, std)
        agg_dim = tuple(range(1, len(D.shape)))     # possibly agg_dim == (1, 2)
        return norm.log_prob(D).sum(dim=agg_dim).mean()

    @staticmethod
    def kl_loss(q, p):
        kl = torch.distributions.kl.kl_divergence(q, p)
        kl_div = kl.squeeze(1).sum(-1).mean()
        return kl_div

    def forward(self, xC, yC, xT, yT):
        qC = self.encode_context(xC, yC)        # (B, 1, zC_dim)
        zC = self.get_encoded_sample(xC, yC, qC)
        qCT = self.encode_full(xC, yC, xT, yT)  # (B, 1, zCT_dim)
        # qCT = self.encode_context(xT, yT)     # deepmind's implements
        yhatT, sgm = self.decode_context(xT, zC)
        log_p = self.log_likelihood(yhatT, std=sgm, D=yT)
        kl_div = self.kl_loss(q=qCT, p=qC)
        loss = - log_p + kl_div
        return yhatT, sgm, loss

    def predict(self, xC, yC, xT):
        qC = self.encode_context(xC, yC)
        zC = self.get_encoded_sample(xC, yC, qC, do_randomize=False)
        yhatT, sgm = self.decode_context(xT, zC)
        return yhatT, sgm

    _func_plotter = None

    def plot_prediction(self, bidx, xC, yC, xT, yT):
        # to check kl collapse or not
        import utils
        p = self._func_plotter
        if p is None:
            p = utils.VisdomLinePlotter(env_name='main')
            self._func_plotter = p
        with torch.no_grad():
            qC = self.encode_context(xC, yC)
            for _ in range(10):
                zC = self.get_encoded_sample(xC, yC, qC)
                yhatT, sgm = self.decode_context(xT, zC)
                x, indices = xT[bidx, :, 0].sort(dim=-1)
                p.plot("xT", f"y[{bidx}]", "yhatT", "trainset: x - yhat",
                       xT[bidx, indices, 0].cpu().numpy(),
                       yhatT[bidx, indices, 0].cpu().numpy(), reset=True)
                p.scatter(xT[bidx, indices, 0].cpu().numpy(),
                          yT[bidx, indices, 0].cpu().numpy(), f"y[{bidx}]", "yT")
                p.scatter(xC[bidx, :, 0].cpu().numpy(),
                          yC[bidx, :, 0].cpu().numpy(), f"y[{bidx}]", "yC", color=(128, 128, 128))
                import time
                time.sleep(0.2)
        return yhatT, sgm

    def to(self, device):
        self.embedder_C = self.embedder_C.to(device)
        self.embedder_T = self.embedder_T.to(device)
        self.latent_encoder = self.latent_encoder.to(device)
        self.decoder = self.decoder.to(device)
        self.device = device
        return super().to(device)

