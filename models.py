"""Step 6: baseline models.

1. QoIMLP              parameters + load history -> scalar QoIs. The sanity floor: if the graph
                       model cannot beat this on the QoIs, something is wrong.
2. MeshGraphNetOperator  MeshGraphNet-style encode-process-decode network that predicts the whole
                       displacement / von Mises time history in ONE shot (no rollout), conditioned
                       on the global parameters and the load time series.

Decoder options for the graph model (an ablation worth reporting):
    "modal"   u_i(t) = sum_k phi_k(i) * q_k(t).  Node latents give K spatial shapes phi (per channel),
              a global network gives K temporal coefficients q(t). This mirrors modal superposition
              for linear dynamics, and is cheap: the output is a low-rank (space x time) field.
    "direct"  each node maps [node latent, global embedding] straight to its T x C history.

The graph model also has a QoI head on the global embedding (log-QoIs), so scalar quantities can be
predicted directly as well as read off the predicted field.
"""
import torch
import torch.nn as nn


def mlp(sizes, layer_norm=False, final_act=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or final_act:
            layers.append(nn.SiLU())
    if layer_norm:
        layers.append(nn.LayerNorm(sizes[-1]))
    return nn.Sequential(*layers)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------
# baseline 1: QoI regressor
# ----------------------------------------------------------------------------
class QoIMLP(nn.Module):
    def __init__(self, u_dim=14, load_len=101, n_qoi=9, hidden=256, depth=3):
        super().__init__()
        sizes = [u_dim + load_len] + [hidden] * depth + [n_qoi]
        self.net = mlp(sizes)

    def forward(self, batch):
        return {"qoi": self.net(torch.cat([batch.u, batch.load], dim=-1))}


# ----------------------------------------------------------------------------
# baseline 2: MeshGraphNet-style one-shot operator
# ----------------------------------------------------------------------------
class MPLayer(nn.Module):
    """Edge update then node update, both residual (MeshGraphNets processor block)."""

    def __init__(self, d):
        super().__init__()
        self.edge_mlp = mlp([3 * d, d, d], layer_norm=True)
        self.node_mlp = mlp([2 * d, d, d], layer_norm=True)

    def forward(self, h, e, edge_index):
        src, dst = edge_index
        e = e + self.edge_mlp(torch.cat([e, h[src], h[dst]], dim=-1))
        agg = h.new_zeros(h.shape).index_add(0, dst, e)          # sum of incoming edge messages
        h = h + self.node_mlp(torch.cat([h, agg], dim=-1))
        return h, e


class MeshGraphNetOperator(nn.Module):
    def __init__(self, node_in=5, edge_in=3, u_dim=14, load_len=101, n_frames=101, n_chan=3,
                 n_qoi=9, hidden=128, n_layers=8, n_modes=16, decoder="modal"):
        super().__init__()
        assert decoder in ("modal", "direct")
        self.decoder, self.T, self.C, self.K = decoder, n_frames, n_chan, n_modes
        d = hidden

        self.node_enc = mlp([node_in, d, d], layer_norm=True)
        self.edge_enc = mlp([edge_in, d, d], layer_norm=True)
        self.layers = nn.ModuleList([MPLayer(d) for _ in range(n_layers)])

        # global branch: parameters + load history + pooled mesh embedding
        self.load_enc = mlp([load_len, d, d])
        self.glob = mlp([u_dim + 2 * d, 2 * d, d], layer_norm=True, final_act=False)

        if decoder == "modal":
            self.node_dec = mlp([d, d, n_modes * n_chan])                       # spatial shapes phi
            self.time_dec = mlp([d, 2 * d, n_modes * n_chan * n_frames])        # temporal coefficients q
        else:
            self.node_dec = mlp([2 * d, d, d, n_frames * n_chan])
        self.qoi_head = mlp([d, d, n_qoi])

    @staticmethod
    def _graph_mean(h, batch, n_graphs):
        s = h.new_zeros(n_graphs, h.shape[1]).index_add(0, batch, h)
        cnt = torch.bincount(batch, minlength=n_graphs).clamp(min=1).unsqueeze(1).to(h.dtype)
        return s / cnt

    def forward(self, b):
        n_graphs = b.num_graphs
        h = self.node_enc(b.x)
        e = self.edge_enc(b.edge_attr)
        for layer in self.layers:
            h, e = layer(h, e, b.edge_index)

        pooled = self._graph_mean(h, b.batch, n_graphs)
        g = self.glob(torch.cat([b.u, self.load_enc(b.load), pooled], dim=-1))     # (B, d)

        if self.decoder == "modal":
            phi = self.node_dec(h).view(-1, self.K, self.C)                           # (N, K, C)
            q = self.time_dec(g).view(n_graphs, self.K, self.C, self.T)               # (B, K, C, T)
            ptr = b.ptr
            field = torch.cat([torch.einsum("nkc,kct->ntc", phi[ptr[i]:ptr[i + 1]], q[i])
                               for i in range(n_graphs)], dim=0)                      # (N, T, C)
        else:
            field = self.node_dec(torch.cat([h, g[b.batch]], dim=-1)).view(-1, self.T, self.C)
        return {"field": field, "qoi": self.qoi_head(g)}


def build_model(name, **kw):
    if name == "mlp":
        return QoIMLP(**kw)
    if name == "mgn":
        return MeshGraphNetOperator(**kw)
    raise ValueError(name)
