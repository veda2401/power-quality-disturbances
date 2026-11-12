"""Closed-set training on the known classes only, with model selection on a
known-class validation set (unknown classes are never touched)."""
import copy
import random
import numpy as np
import torch
import torch.nn.functional as F
from .models import build_model, INPUT_SCALE


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_tensor(X):
    return torch.from_numpy(np.asarray(X, dtype=np.float32) / INPUT_SCALE).unsqueeze(1)


@torch.no_grad()
def extract(model, X, device, bs=1024):
    """Return (logits, penultimate features) as numpy arrays."""
    model.eval()
    logits, feats = [], []
    T = to_tensor(X)
    for i in range(0, len(T), bs):
        lo, fe = model(T[i:i + bs].to(device))
        logits.append(lo.cpu().numpy())
        feats.append(fe.cpu().numpy())
    return np.concatenate(logits), np.concatenate(feats)


def train_model(arch, X_tr, y_tr, X_va, y_va, n_classes, epochs=25, bs=128,
                lr=1e-3, wd=1e-4, width=32, device="cpu", seed=0, verbose=False):
    set_seed(seed)
    model = build_model(arch, n_classes, width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xt, yt = to_tensor(X_tr), torch.from_numpy(y_tr)
    Xv, yv = to_tensor(X_va), torch.from_numpy(y_va)
    g = torch.Generator().manual_seed(seed)

    best_loss, best_state = float("inf"), None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:      # BatchNorm needs > 1 sample
                continue
            logits, _ = model(Xt[idx].to(device))
            loss = F.cross_entropy(logits, yt[idx].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            lo = torch.cat([model(Xv[i:i + 1024].to(device))[0].cpu()
                            for i in range(0, len(Xv), 1024)])
            vloss = F.cross_entropy(lo, yv).item()
            vacc = (lo.argmax(1) == yv).float().mean().item()
        if vloss < best_loss:
            best_loss, best_state = vloss, copy.deepcopy(model.state_dict())
        if verbose:
            print(f"    epoch {ep + 1:02d}/{epochs}  val_loss {vloss:.4f}  val_acc {vacc:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    return model
