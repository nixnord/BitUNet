import numpy.nn as nn

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