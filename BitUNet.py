import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ── Weight quantization: RoundClip from BitNet b1.58 paper ──

def weight_quant(w: torch.Tensor) -> torch.Tensor:
    """
    Ternary weight quantization.
    Formula (paper Eq. 3-5):
        W_ternary = RoundClip( W / (mean(|W|) + eps), -1, 1 )
    Maps every weight to exactly {-beta, 0, +beta}
    where beta = mean(|W|).
    """
    eps   = 1e-5
    beta  = w.abs().mean().clamp(min=eps)
    scale = 1.0 / beta
    w_t   = torch.clamp(torch.round(w * scale), -1.0, 1.0)
    return w_t / scale   # restore magnitude for stable training


def activation_quant(x: torch.Tensor) -> torch.Tensor:
    """
    Per-sample absmax 8-bit activation quantization.
    Formula (paper Eq. 6-7):
        x_q = RoundClip( (127 / max(|x|)) * x, -128, 127 )
    """
    eps    = 1e-5
    shape  = x.shape
    x_flat = x.view(shape[0], -1)
    gamma  = x_flat.abs().max(dim=1, keepdim=True).values.clamp(min=eps)
    scale  = 127.0 / gamma
    scale  = scale.view(shape[0], *([1] * (len(shape) - 1)))
    return torch.clamp(torch.round(x * scale), -128.0, 127.0) / scale


# ── BitConv2d: drop-in replacement for nn.Conv2d ────────────

class BitConv2d(nn.Conv2d):
    """
    Conv2d with BitNet b1.58 ternary weights + 8-bit activations.
    Uses STE (.detach() trick) so gradients flow to latent weights.
    """
    def __init__(self, *args, quant_activations=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant_activations = quant_activations
        #self.register_buffer("weight_scale", torch.ones(1, dtype=torch.float32))

    def forward(self, x):
        w = self.weight
        # At inference time weight_scale is loaded from checkpoint.
        # At training time it stays ones(1) and gets ignored by STE path.
        w_q = w + (weight_quant(w) - w).detach()
        if self.quant_activations:
            x_q = x + (activation_quant(x) - x).detach()
        else:
            x_q = x
        return F.conv2d(x_q, w_q, self.bias,
                        self.stride, self.padding,
                        self.dilation, self.groups)


# ── Quick unit tests ─────────────────────────────────────────

def run_bitnet_tests():
    print("Running BitNet unit tests...")
    torch.manual_seed(42)

    # Test 1: weight_quant → only ternary values
    w  = torch.randn(64, 32, 3, 3)
    wq = weight_quant(w)
    beta = w.abs().mean().item()
    scaled = (wq * (1.0 / beta)).round()
    assert set(scaled.unique().tolist()).issubset({-1.0, 0.0, 1.0}), "NOT TERNARY!"
    print(f"  [PASS] weight_quant produces only {{-1, 0, +1}}")

    # Test 2: activation_quant stays in int8 range
    x  = torch.randn(4, 64, 32, 32)
    xq = activation_quant(x)
    gamma = x.view(4, -1).abs().max(dim=1).values
    x_rescaled = xq * (127.0 / gamma).view(4, 1, 1, 1)
    assert x_rescaled.min().item() >= -128 - 1e-3
    assert x_rescaled.max().item() <=  127 + 1e-3
    print(f"  [PASS] activation_quant range: [{x_rescaled.min():.1f}, {x_rescaled.max():.1f}]")

    # Test 3: gradients flow through BitConv2d
    layer = BitConv2d(32, 64, 3, padding=1)
    x     = torch.randn(2, 32, 16, 16, requires_grad=True)
    loss  = layer(x).sum()
    loss.backward()
    assert x.grad is not None and layer.weight.grad is not None
    assert layer.weight.grad.abs().sum() > 0
    print(f"  [PASS] Gradients flow, weight.grad.norm = {layer.weight.grad.norm():.4f}")

    print("All BitNet tests passed!\n")

run_bitnet_tests()

class DoubleConv(nn.Module):
    """Two Conv→BN→ReLU blocks. Core UNet unit."""
    def __init__(self, in_ch, out_ch, use_bit=True):
        super().__init__()
        Conv = BitConv2d if use_bit else nn.Conv2d
        self.block = nn.Sequential(
            Conv(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            Conv(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """MaxPool → DoubleConv. Halves spatial resolution."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """Upsample → concat skip → DoubleConv. Doubles spatial resolution."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x  = self.up(x)
        dH = skip.size(2) - x.size(2)
        dW = skip.size(3) - x.size(3)
        x  = F.pad(x, [dW//2, dW-dW//2, dH//2, dH-dH//2])
        return self.conv(torch.cat([skip, x], dim=1))  # skip concat in float32


class BitUNet(nn.Module):
    """
    UNet with BitNet b1.58 ternary-weight convolutions.

    NOT quantized:  first conv, final head, all BatchNorm
    ARE quantized:  all encoder/decoder Conv2d layers

    Args:
        in_channels:   RGB = 3
        num_classes:   road/obstacle/background = 3
        base_channels: Slightly wider than std (72 vs 64) to
                       compensate for reduced weight capacity
    """
    def __init__(self, in_channels=3, num_classes=3, base_channels=72):
        super().__init__()
        c = base_channels

        # Encoder
        # First conv: float32 (input interface — NOT quantized)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )
        self.down1 = Down(c,     c*2)   # BitConv2d
        self.down2 = Down(c*2,   c*4)   # BitConv2d
        self.down3 = Down(c*4,   c*8)   # BitConv2d

        # Bridge
        self.bridge = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(c*8, c*16, use_bit=True)
        )

        # Decoder
        self.up3 = Up(c*16 + c*8, c*8)
        self.up2 = Up(c*8  + c*4, c*4)
        self.up1 = Up(c*4  + c*2, c*2)
        self.up0 = Up(c*2  + c,   c)

        # Final head: float32 (output interface — NOT quantized)
        self.head = nn.Conv2d(c, num_classes, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, BitConv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        s1 = self.enc1(x)        # [B,  C,    H,   W]
        s2 = self.down1(s1)      # [B, 2C,  H/2, W/2]
        s3 = self.down2(s2)      # [B, 4C,  H/4, W/4]
        s4 = self.down3(s3)      # [B, 8C,  H/8, W/8]
        b  = self.bridge(s4)     # [B,16C, H/16,W/16]
        d3 = self.up3(b,  s4)
        d2 = self.up2(d3, s3)
        d1 = self.up1(d2, s2)
        d0 = self.up0(d1, s1)
        return self.head(d0)     # [B, num_classes, H, W]

    def count_parameters(self):
        bit_p   = sum(m.weight.numel() for m in self.modules() if isinstance(m, BitConv2d))
        float_p = sum(m.weight.numel() for m in self.modules() if type(m) is nn.Conv2d)
        total   = bit_p + float_p
        f32_mb  = total * 4 / 1024**2
        bit_mb  = (bit_p * 1.58/8 + float_p * 4) / 1024**2
        print(f"  Total params:      {total:,}")
        print(f"  Ternary params:    {bit_p:,}  ({100*bit_p//total}%)")
        print(f"  Float32 size:      {f32_mb:.2f} MB")
        print(f"  Ternary size:      {bit_mb:.2f} MB")
        print(f"  Compression ratio: {f32_mb/bit_mb:.1f}x")


class BaselineUNet(nn.Module):
    """Standard float32 UNet. Same architecture, no quantization. Used as teacher."""
    def __init__(self, in_channels=3, num_classes=3, base_channels=64):
        super().__init__()
        c = base_channels
        def dc(ic, oc):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
                nn.Conv2d(oc, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
            )
        self.enc1   = dc(in_channels, c)
        self.down1  = nn.Sequential(nn.MaxPool2d(2), dc(c,   c*2))
        self.down2  = nn.Sequential(nn.MaxPool2d(2), dc(c*2, c*4))
        self.down3  = nn.Sequential(nn.MaxPool2d(2), dc(c*4, c*8))
        self.bridge = nn.Sequential(nn.MaxPool2d(2), dc(c*8, c*16))
        self.up3    = dc(c*16+c*8, c*8)
        self.up2    = dc(c*8 +c*4, c*4)
        self.up1    = dc(c*4 +c*2, c*2)
        self.up0    = dc(c*2 +c,   c)
        self.head   = nn.Conv2d(c, num_classes, 1)

    def _up(self, x, skip, conv):
        x  = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        dH = skip.size(2)-x.size(2); dW = skip.size(3)-x.size(3)
        x  = F.pad(x, [dW//2, dW-dW//2, dH//2, dH-dH//2])
        return conv(torch.cat([skip, x], dim=1))

    def forward(self, x):
        s1=self.enc1(x); s2=self.down1(s1); s3=self.down2(s2); s4=self.down3(s3)
        b =self.bridge(s4)
        d3=self._up(b,s4,self.up3); d2=self._up(d3,s3,self.up2)
        d1=self._up(d2,s2,self.up1); d0=self._up(d1,s1,self.up0)
        return self.head(d0)


# ── Architecture test ────────────────────────────────────────

print("Architecture test:")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

model = BitUNet(in_channels=3, num_classes=3, base_channels=72).to(device)
x_test = torch.randn(2, 3, 256, 256).to(device)
out    = model(x_test)
assert out.shape == (2, 3, 256, 256)
print(f"  Input:  {x_test.shape}  →  Output: {out.shape}")
print("  Parameter stats:")
model.count_parameters()

loss = out.mean(); loss.backward()
print("  Backward pass: OK\n")


import random
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
import torchvision.transforms as T


def joint_augment(image, mask, img_size=(256,256), is_train=True):
    image = TF.resize(image, img_size, TF.InterpolationMode.BILINEAR)
    mask  = TF.resize(mask,  img_size, TF.InterpolationMode.NEAREST)

    # Force mask back to L mode after resize (torchvision can silently convert it)
    if mask.mode != "L":
        mask = mask.convert("L")

    if is_train:
        if random.random() > 0.5:
            image = TF.hflip(image); mask = TF.hflip(mask)

        crop_h = int(img_size[0] * 0.9)
        crop_w = int(img_size[1] * 0.9)
        i,j,h,w = T.RandomCrop.get_params(image, (crop_h, crop_w))
        image = TF.resized_crop(image, i,j,h,w, img_size, TF.InterpolationMode.BILINEAR)
        mask  = TF.resized_crop(mask,  i,j,h,w, img_size, TF.InterpolationMode.NEAREST)

        # Force again after crop (same issue)
        if mask.mode != "L":
            mask = mask.convert("L")

        image = TF.adjust_brightness(image, 1 + random.uniform(-0.2, 0.2))
        image = TF.adjust_contrast(image,   1 + random.uniform(-0.2, 0.2))

    image = TF.to_tensor(image)
    image = TF.normalize(image, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

    # Convert mask safely — always 2D
    mask_np = np.array(mask)
    if mask_np.ndim == 3:
        mask_np = mask_np[:, :, 0]
    mask = torch.from_numpy(mask_np).long()

    return image, mask


class SyntheticRoadDataset(Dataset):
    """
    Synthetic road + obstacle images. No downloads needed.
    Use this to verify the full pipeline before using KITTI/Cityscapes.

    Classes: 0=background, 1=road, 2=obstacle
    """
    def __init__(self, n_samples=500, img_size=(256,256), split="train"):
        self.n  = n_samples if split=="train" else n_samples//5
        self.sz = img_size
        self.is_train = (split=="train")
        print(f"SyntheticRoadDataset [{split}]: {self.n} samples")

    def __len__(self): return self.n

    def __getitem__(self, idx):
        H, W = self.sz
        rng  = np.random.RandomState(idx)
        img  = (rng.rand(H,W,3)*0.4+0.3).astype(np.float32)
        mask = np.zeros((H,W), dtype=np.uint8)

        # Road: trapezoid in lower half
        cx = W//2
        top_w = int(W*rng.uniform(0.2,0.4)); bot_w = int(W*rng.uniform(0.6,0.9))
        top_y = int(H*rng.uniform(0.3,0.5))
        for y in range(top_y, H):
            t = (y-top_y)/max(H-top_y,1)
            hw = int((top_w + t*(bot_w-top_w))/2)
            xl = max(0,cx-hw); xr = min(W,cx+hw)
            img[y,xl:xr]  = [0.45+rng.rand()*0.1]*3
            mask[y,xl:xr] = 1  # road

        # Obstacles: random rectangles on road
        for _ in range(rng.randint(0,4)):
            oy=rng.randint(top_y,H-20); ox=rng.randint(cx-bot_w//2,cx+bot_w//2)
            oh=rng.randint(15,40);       ow=rng.randint(10,30)
            img [oy:oy+oh, ox:ox+ow] = rng.rand(3)*0.5+0.3
            mask[oy:oy+oh, ox:ox+ow] = 2  # obstacle

        img_pil  = Image.fromarray((img*255).astype(np.uint8))   # mode inferred from shape
        mask_pil = Image.fromarray(mask)
        return joint_augment(img_pil, mask_pil, self.sz, self.is_train)


class KITTIRoadDataset(Dataset):
    """
    KITTI Road dataset.
    Download: http://www.cvlibs.net/datasets/kitti/eval_road.php
    Mount to Drive or upload to /content/kitti_road/
    """
    GT_PREFIXES = ["um_lane_","um_road_","umm_road_","uu_road_"]

    def __init__(self, root, split="train", img_size=(256,256), val_frac=0.15, seed=42):
        from pathlib import Path
        self.sz       = img_size
        self.is_train = (split=="train")
        img_dir = Path(root)/"training"/"image_2"
        gt_dir  = Path(root)/"training"/"gt_image_2"

        pairs = []
        for ip in sorted(img_dir.glob("*.png")):
            for pfx in self.GT_PREFIXES:
                gp = gt_dir/f"{pfx}{ip.stem.split('_')[-1]}.png"
                if gp.exists(): pairs.append((ip,gp)); break

        rng = random.Random(seed); rng.shuffle(pairs)
        n_val = max(1, int(len(pairs)*val_frac))
        self.pairs = pairs[:n_val] if split=="val" else pairs[n_val:]
        print(f"KITTIRoadDataset [{split}]: {len(self.pairs)} samples")

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        ip, gp = self.pairs[idx]
        img  = Image.open(ip).convert("RGB")
        gt   = np.array(Image.open(gp).convert("RGB"))
        mask = np.zeros(gt.shape[:2], dtype=np.uint8)
        mask[gt[:,:,0] > 128] = 1   # road → class 1
        return joint_augment(img, Image.fromarray(mask,"L"), self.sz, self.is_train)


def get_dataloaders(dataset="synthetic", root="", img_size=(256,256),
                    batch_size=8, num_workers=0):
    if dataset == "synthetic":
        train_ds = SyntheticRoadDataset(1000, img_size, "train")
        val_ds   = SyntheticRoadDataset(1000, img_size, "val")
    elif dataset == "kitti":
        train_ds = KITTIRoadDataset(root, "train", img_size)
        val_ds   = KITTIRoadDataset(root, "val",   img_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_loader = DataLoader(train_ds, batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)
    return train_loader, val_loader

class DiceLoss(nn.Module):
    def __init__(self, num_classes=3, smooth=1.0, ignore_index=255):
        super().__init__()
        self.C = num_classes; self.smooth = smooth; self.ign = ignore_index

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        B,C,H,W = probs.shape
        toh = torch.zeros_like(probs)
        valid = (targets != self.ign)
        ts = targets.clone(); ts[~valid] = 0
        toh.scatter_(1, ts.unsqueeze(1), 1.0)
        dims = (0,2,3)
        inter = (probs*toh).sum(dims)
        union = (probs+toh).sum(dims)
        dice  = (2*inter+self.smooth)/(union+self.smooth)
        return 1.0 - dice.mean()


class SegmentationLoss(nn.Module):
    def __init__(self, num_classes=3, alpha=0.5, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.ce    = nn.CrossEntropyLoss(weight=class_weights)
        self.dice  = DiceLoss(num_classes)

    def forward(self, logits, targets):
        return self.alpha*self.ce(logits,targets) + (1-self.alpha)*self.dice(logits,targets)


class SegmentationMetrics:
    """Tracks mIoU and per-class IoU via confusion matrix."""
    def __init__(self, num_classes=3):
        self.C = num_classes
        self.reset()

    def reset(self):
        self.conf = np.zeros((self.C, self.C), dtype=np.int64)

    def update(self, logits, targets):
        preds = logits.argmax(1).cpu().numpy().ravel()
        tgts  = targets.cpu().numpy().ravel()
        valid = tgts < self.C
        preds, tgts = preds[valid], tgts[valid]
        self.conf += np.bincount(self.C*tgts+preds,
                                 minlength=self.C**2).reshape(self.C,self.C)

    def compute(self):
        tp  = np.diag(self.conf)
        iou = tp / np.maximum(self.conf.sum(1)+self.conf.sum(0)-tp, 1e-6)
        return {"miou": float(iou.mean()), "iou_per_class": iou.tolist(),
                "pixel_acc": float(tp.sum()/np.maximum(self.conf.sum(),1))}


class BitNetScheduler:
    """
    Two-stage LR + weight decay schedule from the BitNet b1.58 paper.

    Stage 1 (first half): High LR + cosine weight decay (peak 0.1)
    Stage 2 (second half): Decayed LR + weight decay = 0

    Paper quote: "after removing weight decay, the training loss decreased
    significantly faster" — Wang et al. 2025, Sec 2.2.2
    """
    def __init__(self, optimizer, total_steps,
                 lr1=1.5e-3, lr2=1.0e-3, lr_min=1e-5,
                 wd1=0.1, warmup_steps=375):
        self.opt    = optimizer
        self.T      = total_steps
        self.S      = total_steps // 2   # stage boundary (midpoint)
        self.lr1    = lr1;   self.lr2  = lr2;  self.lr_min = lr_min
        self.wd1    = wd1
        self.warmup = warmup_steps
        self.step   = 0

    def update(self):
        s = self.step
        # Learning rate
        if s < self.warmup:
            lr = self.lr1 * (s+1) / self.warmup
        elif s < self.S:
            prog = (s-self.warmup) / max(self.S-self.warmup, 1)
            lr   = self.lr_min + (self.lr1-self.lr_min) * 0.5*(1+np.cos(np.pi*prog))
        else:
            prog = (s-self.S) / max(self.T-self.S, 1)
            lr   = self.lr_min + (self.lr2-self.lr_min) * 0.5*(1+np.cos(np.pi*prog))

        # Weight decay: cosine in stage 1, zero in stage 2
        wd = self.wd1 * 0.5*(1+np.cos(np.pi*s/max(self.S,1))) if s < self.S else 0.0

        for g in self.opt.param_groups:
            g["lr"] = lr; g["weight_decay"] = wd

        self.step += 1
        return lr, wd


class AverageMeter:
    def __init__(self): self.reset()
    def reset(self):    self.sum=0.; self.n=0
    def update(self, v, n=1): self.sum+=v*n; self.n+=n
    @property
    def avg(self): return self.sum/max(self.n,1)


# ─────────────────────────────────────────────────────────────
# CELL 6 — Path-Centric Algorithm (path_centric.py contents)
# ─────────────────────────────────────────────────────────────

def build_path_weight_mask(projected_waypoints, H, W, sigma=30.0, min_weight=0.05):
    """
    Gaussian weight mask centred on the projected path.
    W(x,y) = exp( -D(x,y)^2 / (2*sigma^2) )
    where D = distance to nearest projected waypoint.
    """
    if len(projected_waypoints) == 0:
        return np.ones((H,W), dtype=np.float32)

    yy, xx = np.mgrid[0:H, 0:W]
    pts    = projected_waypoints.astype(np.float32)
    dx = xx[:,:,None] - pts[None,None,:,0]
    dy = yy[:,:,None] - pts[None,None,:,1]
    min_dist_sq = (dx**2 + dy**2).min(axis=2)
    weight = np.exp(-min_dist_sq / (2*sigma**2))
    return np.clip(weight, min_weight, 1.0).astype(np.float32)


def project_waypoints(waypoints_3d, K, extrinsic=None):
    """
    Project 3D path waypoints (robot frame) into 2D image coordinates.

    Args:
        waypoints_3d: [N,3] in camera/robot frame (X_right, Y_down, Z_forward)
        K:            [3,3] camera intrinsic matrix
        extrinsic:    [4,4] world→camera transform (optional)
    Returns:
        [M,2] (u,v) pixel coordinates for in-front waypoints
    """
    pts = waypoints_3d.astype(np.float32)
    if extrinsic is not None:
        pts = (extrinsic[:3,:3] @ pts.T).T + extrinsic[:3,3]
    valid = pts[:,2] > 0.1
    pts   = pts[valid]
    if len(pts) == 0:
        return np.empty((0,2), dtype=np.float32)
    u = K[0,0]*pts[:,0]/pts[:,2] + K[0,2]
    v = K[1,1]*pts[:,1]/pts[:,2] + K[1,2]
    return np.stack([u,v], axis=1)


@torch.no_grad()
def path_centric_inference(model, image_tensor, K, waypoints_3d=None,
                            sigma=30.0, obs_threshold=0.4, device=None):
    """
    Full path-centric segmentation pipeline.

    Steps:
      1. Run BitNet UNet → full scene segmentation
      2. Project path waypoints into image plane
      3. Build Gaussian corridor weight mask
      4. Multiply segmentation probs by weight mask
      5. Check for obstacle in corridor

    Args:
        model:          Trained BitUNet (eval mode)
        image_tensor:   [1,3,H,W] normalised input
        K:              [3,3] camera intrinsics
        waypoints_3d:   [N,3] planned path in camera frame
        sigma:          Corridor width in pixels
        obs_threshold:  Weighted obstacle conf threshold for alert

    Returns:
        dict with seg_pred, weight_mask, obstacle_alert, weighted_probs
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    image_tensor = image_tensor.to(device)
    H, W = image_tensor.shape[2], image_tensor.shape[3]

    # Step 1: Segmentation
    logits = model(image_tensor)
    probs  = F.softmax(logits, dim=1)
    pred   = logits.argmax(dim=1)

    # Step 2 & 3: Project waypoints and build mask
    proj_wp = np.empty((0,2), dtype=np.float32)
    if waypoints_3d is not None and len(waypoints_3d) > 0:
        proj_wp = project_waypoints(waypoints_3d, K)

    weight_mask = build_path_weight_mask(proj_wp, H, W, sigma=sigma)

    # Step 4: Apply mask
    wm = torch.from_numpy(weight_mask).to(device).unsqueeze(0).unsqueeze(0)
    weighted_probs = probs * wm

    # Step 5: Obstacle alert (class 2)
    obs_map       = weighted_probs[0,2].cpu().numpy() if probs.shape[1] > 2 \
                    else np.zeros((H,W))
    obstacle_alert = bool(obs_map.max() > obs_threshold)

    return {
        "seg_pred":          pred.cpu(),
        "seg_probs":         probs.cpu(),
        "weight_mask":       weight_mask,
        "weighted_probs":    weighted_probs.cpu(),
        "obstacle_alert":    obstacle_alert,
        "obstacle_conf_map": obs_map,
        "projected_waypoints": proj_wp,
    }


print("Path-centric module loaded.\n")