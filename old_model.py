import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    '''(Conv2d => BN => ReLU) x 2'''
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)
    

class DeconvBlock(nn.Module):
    '''(Conv2d => BN => ReLU) x 2'''
    def __init__(self, in_ch, out_ch):
        super(DeconvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.ConvTranspose2d(in_ch, 2*out_ch, kernel_size=4, stride=1, padding=1, output_padding=0, bias=True),
            nn.BatchNorm2d(2*out_ch),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2*out_ch, out_ch, kernel_size=4, stride=1, padding=1, output_padding=0, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)


class DownBlock(nn.Module):
    '''Pooling => (Conv2d => BN => ReLU)*2'''
    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.AvgPool2d(pool=2),
            ConvBlock(in_ch, out_ch)
        )

    def forward(self, x):
        return self.convblock(x)


class UpBlock(nn.Module):
	'''(Conv2d => BN => ReLU)*2 + Upscale'''
	def __init__(self, in_ch, out_ch):
		super(UpBlock, self).__init__()
		self.convblock = nn.Sequential(
			DeconvBlock(in_ch, in_ch),
			nn.Upsample(scale_factor=2, mode='nearest')
		)

	def forward(self, x):
		return self.convblock(x)




class NewUNet(nn.Module):
    def __init__(self):
        super(NewUNet, self).__init__()

        self.base = 64

        self.chs_lyr0 = 1 * self.base
        self.chs_lyr1 = 2 * self.base
        self.chs_lyr2 = 4 * self.base
        self.chs_lyr3 = 8 * self.base
        self.chs_lyr4 = 16 * self.base

        self.inc = ConvBlock(in_ch=4, out_ch=self.chs_lyr0)
        self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
        self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
        self.downc2 = DownBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr3)
        self.downc3 = DownBlock(in_ch=self.chs_lyr3, out_ch=self.chs_lyr4)
        self.upc4 = UpBlock(in_ch=self.chs_lyr4, out_ch=self.chs_lyr3)
        self.upc3 = UpBlock(in_ch=2*self.chs_lyr3, out_ch=self.chs_lyr2)
        self.upc2 = UpBlock(in_ch=2*self.chs_lyr2, out_ch=self.chs_lyr1)
        self.upc1 = UpBlock(in_ch=2*self.chs_lyr1, out_ch=self.chs_lyr0)
        self.outc = DeconvBlock(in_ch=2*self.chs_lyr0, out_ch=1)

    def forward(self, x):

        x0 = self.inc(x)
        x1 = self.downc0(x0)
        x2 = self.downc1(x1)
        x3 = self.downc2(x2)
        x4 = self.downc3(x3)
        x4 = self.upc4(x4)
        x3 = self.upc3(x3 + x4)
        x2 = self.upc3(x2 + x3)
        x1 = self.upc1(x1 + x2)
        x = self.outc(x0 + x1)
        return x