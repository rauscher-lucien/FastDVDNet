import torch
import torch.nn as nn

class CvBlock(nn.Module):
    '''(Conv2d => BN => ReLU) x 2'''
    def __init__(self, in_ch, out_ch):
        super(CvBlock, self).__init__()
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

class InputCvBlock(nn.Module):
    '''Adjusted to take grayscale images (single channel) as input'''
    def __init__(self, num_in_frames, out_ch):
        super(InputCvBlock, self).__init__()
        self.interm_ch = 30
        self.convblock = nn.Sequential(
            nn.Conv2d(num_in_frames, num_in_frames*self.interm_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_in_frames*self.interm_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_in_frames*self.interm_ch, out_ch, kernel_size=3, padding=1, bias=False),
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
            nn.MaxPool2d(kernel_size=2, stride=2),
            CvBlock(in_ch, out_ch)
        )

    def forward(self, x):
        return self.convblock(x)

class UpBlock(nn.Module):
	'''(Conv2d => BN => ReLU)*2 + Upscale'''
	def __init__(self, in_ch, out_ch):
		super(UpBlock, self).__init__()
		self.convblock = nn.Sequential(
			CvBlock(in_ch, in_ch),
			nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0)
		)

	def forward(self, x):
		return self.convblock(x)

class OutputCvBlock(nn.Module):
	'''Conv2d => BN => ReLU => Conv2d'''
	def __init__(self, in_ch, out_ch):
		super(OutputCvBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(in_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
		)

	def forward(self, x):
		return self.convblock(x)

class DenBlock(nn.Module):
    """Modified DenBlock to accept 2 input frames for the FastDVDnet model."""
    def __init__(self):
        super(DenBlock, self).__init__()
        self.chs_lyr0 = 32
        self.chs_lyr1 = 64
        self.chs_lyr2 = 128

        # Adjusting InputCvBlock to accept 2 frames concatenated along the channel dimension
        self.inc = CvBlock(in_ch=2, out_ch=self.chs_lyr0)
        self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
        self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
        self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
        self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
        self.outc = CvBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr0)  # Adjusting for grayscale output
        self.finalconv = nn.Conv2d(self.chs_lyr0, 1, kernel_size=3, padding=1, bias=False)

        #self.reset_params()

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, frames):
        # frames is expected to be a tensor of shape [N, 2, H, W], where N is the batch size
        x0 = self.inc(frames)  # frames already concatenated
        x1 = self.downc0(x0)
        x2 = self.downc1(x1)
        x2 = self.upc2(x2)
        x1 = self.upc1(x1 + x2)
        x = self.outc(x0 + x1)
        x = self.finalconv(x)
        return x

class FastDVDnet(nn.Module):
    """Modified FastDVDnet model to process 4 grayscale input frames."""
    def __init__(self):
        super(FastDVDnet, self).__init__()
        # Using the same DenBlock for frames 0 and 2, and 1 and 3, with shared weights.
        self.shared_den_block = DenBlock()
        # Another DenBlock to process the outputs of the first DenBlocks.
        self.final_den_block = DenBlock()

    def forward(self, x):
        x0, x1, x2, x3 = [x[..., :, :, :, i] for i in range(4)]  # Adjust indexing if necessary

        # Assuming x0, x1, x2, x3 are your grayscale frames with shape [N, 1, H, W]
        # Concatenate frames for processing: frames 0 and 2, and frames 1 and 3
        input_02 = torch.cat([x0, x2], dim=1)  # Now shape [N, 2, H, W]
        input_13 = torch.cat([x1, x3], dim=1)  # Now shape [N, 2, H, W]

        # Process through the shared DenBlocks
        output_02 = self.shared_den_block(input_02)
        output_13 = self.shared_den_block(input_13)

        # Process the outputs through the final DenBlock
        final_input = torch.cat([output_02, output_13], dim=1)
        final_output = self.final_den_block(final_input)

        return final_output
