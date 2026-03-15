import torch
import torch.nn as nn
import torch.nn.functional as F


class ScoreNetwork(nn.Module):
    def __init__(self, enc_in, d_model, kernel=3):
        super(ScoreNetwork, self).__init__()

        assert kernel % 2 == 1, "Kernel size must be an odd number"
        pad = kernel // 2

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=enc_in, out_channels=int(d_model / 4), kernel_size=kernel, stride=2, padding=pad),
            nn.ReLU(),
            nn.Conv1d(in_channels=int(d_model / 4), out_channels=int(d_model / 2), kernel_size=kernel, stride=2, padding=pad),
            nn.ReLU(),
            nn.Conv1d(in_channels=int(d_model / 2), out_channels=d_model, kernel_size=kernel, stride=2, padding=pad),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=d_model, out_channels=int(d_model / 2), kernel_size=kernel, stride=2, padding=pad, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=int(d_model / 2), out_channels=int(d_model / 4), kernel_size=kernel, stride=2, padding=pad, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=int(d_model / 4), out_channels=enc_in, kernel_size=kernel, stride=2, padding=pad, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, N, L)  -> Conv1d expects (B, C, L) where C=N
        encoded = self.encoder(x)
        skip = x
        decoded = self.decoder(encoded)
        output = decoded + skip[:, :, :decoded.size(2)]  # adjust length if needed
        return output


class MLP_attention(nn.Module):
    """
    Returns:
      - output: (B, N, d_model)
      - attn:   (B, N, d_model)  or (B, N, L) depending on stage
    In this architecture, after first Linear(seq_len->d_model), L becomes d_model.
    So attention map is (B, N, d_model). This is still a valid heatmap for "token/time-like" dimension in latent space.
    If you want real time-step heatmap, export at the input stage BEFORE Linear; see robustness script note.
    """
    def __init__(self, d_model, enc_in, kernel):
        super(MLP_attention, self).__init__()
        self.value = nn.Linear(d_model, d_model)
        self.score = ScoreNetwork(enc_in, d_model, kernel)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, return_attn: bool = False):
        # x: (B, N, L)  where L is current feature length (after Linear it becomes d_model)
        B, N, L = x.shape

        attn = torch.softmax(self.score(x), dim=1)  # (B,N,L) channel-wise attention per "position"
        x1 = attn * self.value(x) + x
        x1 = self.norm(x1.reshape(-1, L)).reshape(B, N, L)

        out = self.norm((self.mlp(x1) + x1).reshape(-1, L)).reshape(B, N, L)

        if return_attn:
            return out, attn
        return out


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.e_layers = configs.e_layers

        # whether output attention maps
        self.output_attention = int(getattr(configs, "output_attention", 0))

        self.in_proj = nn.Sequential(
            nn.Linear(self.seq_len, self.d_model),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList([
            nn.Sequential(
                # we will call MLP_attention manually to get attn
                # then ReLU
            )
            for _ in range(self.e_layers)
        ])
        self.attn_layers = nn.ModuleList([
            MLP_attention(self.d_model, self.enc_in, configs.kernel)
            for _ in range(self.e_layers)
        ])
        self.post_relu = nn.ReLU()

        self.out_proj = nn.Linear(self.d_model, self.pred_len)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        x: (B, seq_len, enc_in)
        Return:
          - y: (B, pred_len, enc_in)
          - attns (optional): list of attention maps, each (B, enc_in, d_model)
        """
        # instance norm
        seq_mean = torch.mean(x, dim=1, keepdim=True)
        seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        x = (x - seq_mean) / torch.sqrt(seq_var)

        # channel-first
        z = x.permute(0, 2, 1)  # (B, N, seq_len)

        # project time dimension seq_len -> d_model
        z = self.in_proj(z)  # (B, N, d_model)

        attn_list = []

        for i in range(self.e_layers):
            if self.output_attention:
                z, attn = self.attn_layers[i](z, return_attn=True)  # attn: (B,N,d_model)
                attn_list.append(attn)
            else:
                z = self.attn_layers[i](z, return_attn=False)
            z = self.post_relu(z)

        # d_model -> pred_len
        y = self.out_proj(z)  # (B, N, pred_len)
        y = y.permute(0, 2, 1)  # (B, pred_len, N)

        # instance denorm
        y = y * torch.sqrt(seq_var) + seq_mean

        if self.output_attention:
            return y, attn_list
        return y
