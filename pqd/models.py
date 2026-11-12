"""Small 1D classifiers.  The paper is about open-set behaviour, not
architectures, so both models are deliberately plain."""
import torch
import torch.nn as nn

INPUT_SCALE = 2.0   # inputs are divided by this constant (fixed, not data dependent)


class _Pool(nn.Module):
    """Concatenate global average and global max pooling -> (B, 2C)."""
    def forward(self, x):
        return torch.cat([x.mean(dim=2), x.amax(dim=2)], dim=1)


def _conv_block(cin, cout, k, pool):
    layers = [nn.Conv1d(cin, cout, k, padding=k // 2, bias=False),
              nn.BatchNorm1d(cout), nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool1d(2))
    return nn.Sequential(*layers)


class CNN1D(nn.Module):
    def __init__(self, n_classes, width=32):
        super().__init__()
        w = width
        self.body = nn.Sequential(
            _conv_block(1, w, 7, True),
            _conv_block(w, 2 * w, 5, True),
            _conv_block(2 * w, 4 * w, 3, True),
            _conv_block(4 * w, 4 * w, 3, False),
        )
        self.pool = _Pool()
        self.feat_dim = 8 * w
        self.fc = nn.Linear(self.feat_dim, n_classes)

    def forward(self, x):
        f = self.pool(self.body(x))
        return self.fc(f), f


class _BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False),
                                       nn.BatchNorm1d(cout))

    def forward(self, x):
        y = torch.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        s = x if self.short is None else self.short(x)
        return torch.relu(y + s)


class ResNet1D(nn.Module):
    def __init__(self, n_classes, width=32):
        super().__init__()
        w = width
        self.stem = nn.Sequential(nn.Conv1d(1, w, 7, padding=3, bias=False),
                                  nn.BatchNorm1d(w), nn.ReLU(inplace=True))
        self.layers = nn.Sequential(_BasicBlock(w, w, 1),
                                    _BasicBlock(w, 2 * w, 2),
                                    _BasicBlock(2 * w, 4 * w, 2))
        self.pool = _Pool()
        self.feat_dim = 8 * w
        self.fc = nn.Linear(self.feat_dim, n_classes)

    def forward(self, x):
        f = self.pool(self.layers(self.stem(x)))
        return self.fc(f), f


def build_model(arch, n_classes, width=32):
    if arch == "cnn":
        return CNN1D(n_classes, width)
    if arch == "resnet":
        return ResNet1D(n_classes, width)
    raise ValueError(f"unknown arch {arch!r} (use 'cnn' or 'resnet')")
