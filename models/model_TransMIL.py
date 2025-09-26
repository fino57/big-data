import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,
            pinv_iterations=6,
            residual=True,
            dropout=0.1
        )

    def forward(self, x):
        return x + self.attn(self.norm(x))


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, n_classes):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(768, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)

    def forward(self, data_s, data_l=None, label=None):
        """
        Args:
            data_s: tensor [B, N, 768] hoặc [N, 768] nếu thiếu batch dim
            data_l: (không dùng, để giữ API đồng bộ với core_utils)
            label: ground truth label (optional)

        Returns:
            Y_prob: softmax probability [B, n_classes]
            Y_hat: predicted class index [B]
            loss: cross_entropy nếu có label, else None
        """
        # đảm bảo có batch dim
        if data_s.dim() == 2:
            data_s = data_s.unsqueeze(0)  # -> (1, N, 768)

        h = self._fc1(data_s.float())  # [B, N, 512]

        # pad để reshape thành ma trận vuông
        B, N, D = h.shape
        H = int(np.ceil(np.sqrt(N)))
        W = H
        add_len = H * W - N
        if add_len > 0:
            pad = torch.zeros(B, add_len, D, device=h.device)
            h = torch.cat([h, pad], dim=1)  # [B, H*W, 512]

        # thêm cls token
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)  # [B, 1+H*W, 512]

        # transformer layers
        h = self.layer1(h)
        h = self.pos_layer(h, H, W)
        h = self.layer2(h)

        # lấy cls token
        h = self.norm(h)[:, 0]  # [B, 512]

        # predict
        logits = self._fc2(h)  # [B, n_classes]
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.argmax(Y_prob, dim=1)

        loss = None
        if label is not None:
            loss = F.cross_entropy(logits, label)

        return Y_prob, Y_hat, loss


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.randn((2, 6000, 768), device=device)
    labels = torch.tensor([0, 1], device=device)
    model = TransMIL(n_classes=2).to(device)
    model.eval()
    with torch.no_grad():
        Y_prob, Y_hat, loss = model(data, label=labels)
    print("Y_prob:", Y_prob.shape)
    print("Y_hat:", Y_hat)
    print("Loss:", loss.item() if loss is not None else None)
