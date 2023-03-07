""" Network architectures.
"""
# pylint: disable=W0221,W0622,C0103,R0913

##
import torch
import torch.nn as nn
import torch.nn.parallel

##
def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)

###

class PGGAN_Encoder(nn.Module):
    """
    input_channel : grayscale = 1, RGB = 3
    latent_dim : latent vector dimesnion
    base_channel : 첫번째 conv layer에서의 필터 수
    """
    def __init__(self, input_channel=1, latent_dim=512, base_channel=64):
        super(PGGAN_Encoder, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channel, base_channel, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(base_channel, base_channel*2, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(base_channel*2, base_channel*4, 4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(base_channel*4, base_channel*8, 4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(base_channel*8, base_channel*16, 4, stride=2, padding=1)
        self.conv6 = nn.Conv2d(base_channel*16, base_channel*32, 4, stride=2, padding=1)
        self.conv7 = nn.Conv2d(base_channel*32, base_channel*32, 4, stride=2, padding=1)
        self.conv8 = nn.Conv2d(base_channel*32, latent_dim, 4, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(base_channel)
        self.bn2 = nn.BatchNorm2d(base_channel*2)
        self.bn3 = nn.BatchNorm2d(base_channel*4)
        self.bn4 = nn.BatchNorm2d(base_channel*8)
        self.bn5 = nn.BatchNorm2d(base_channel*16)
        self.bn6 = nn.BatchNorm2d(base_channel*32)
        self.bn7 = nn.BatchNorm2d(base_channel*32)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.lrelu(self.bn1(self.conv1(x)))
        x = self.lrelu(self.bn2(self.conv2(x)))
        x = self.lrelu(self.bn3(self.conv3(x)))
        x = self.lrelu(self.bn4(self.conv4(x)))
        x = self.lrelu(self.bn5(self.conv5(x)))
        x = self.lrelu(self.bn6(self.conv6(x)))
        x = self.lrelu(self.bn7(self.conv7(x)))
        x = self.conv8(x)
        return x.view(x.size(0), -1)

class PGGANDecoder(nn.Module):
    def __init__(self, latent_dim, base_channels=512, max_channels=32):
        super(PGGANDecoder, self).__init__()

        self.base_channels = base_channels
        self.max_channels = max_channels
        self.latent_dim = latent_dim

        self.fc = nn.Linear(latent_dim, 4 * 4 * base_channels)

        self.blocks = nn.ModuleList()
        in_channels = base_channels
        for i in range(int(torch.log2(torch.tensor(256))) - 2):
            out_channels = max(base_channels // (2 ** (i + 1)), max_channels)
            self.blocks.append(UpsamplingBlock(in_channels, out_channels))
            in_channels = out_channels * 2
        self.final_block = UpsamplingBlock(in_channels, 1)

    def forward(self, x):
        x = self.fc(x).view(-1, self.base_channels, 4, 4)
        for block in self.blocks:
            x = block(x)
        x = self.final_block(x)
        x = torch.sigmoid(x)

        return x

class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsamplingBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels * 2, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels * 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


# class Encoder(nn.Module):
#     """
#     DCGAN ENCODER NETWORK
#     """

#     def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0, add_final_conv=True):
#         super(Encoder, self).__init__()
#         self.ngpu = ngpu
#         assert isize % 16 == 0, "isize has to be a multiple of 16"

#         main = nn.Sequential()
#         # input is nc x isize x isize
#         main.add_module('initial-conv-{0}-{1}'.format(nc, ndf),
#                         nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
#         main.add_module('initial-relu-{0}'.format(ndf),
#                         nn.LeakyReLU(0.2, inplace=True))
#         csize, cndf = isize / 2, ndf

#         # Extra layers
#         for t in range(n_extra_layers):
#             main.add_module('extra-layers-{0}-{1}-conv'.format(t, cndf),
#                             nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
#             main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cndf),
#                             nn.BatchNorm2d(cndf))
#             main.add_module('extra-layers-{0}-{1}-relu'.format(t, cndf),
#                             nn.LeakyReLU(0.2, inplace=True))

#         while csize > 4:
#             in_feat = cndf
#             out_feat = cndf * 2
#             main.add_module('pyramid-{0}-{1}-conv'.format(in_feat, out_feat),
#                             nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
#             main.add_module('pyramid-{0}-batchnorm'.format(out_feat),
#                             nn.BatchNorm2d(out_feat))
#             main.add_module('pyramid-{0}-relu'.format(out_feat),
#                             nn.LeakyReLU(0.2, inplace=True))
#             cndf = cndf * 2
#             csize = csize / 2

#         # state size. K x 4 x 4
#         if add_final_conv:
#             main.add_module('final-{0}-{1}-conv'.format(cndf, 1),
#                             nn.Conv2d(cndf, nz, 4, 1, 0, bias=False))

#         self.main = main

#     def forward(self, input):
#         if self.ngpu > 1:
#             output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
#         else:
#             output = self.main(input)

#         return output

# ##
# class Decoder(nn.Module):
#     """
#     DCGAN DECODER NETWORK
#     """
#     def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
#         super(Decoder, self).__init__()
#         self.ngpu = ngpu
#         assert isize % 16 == 0, "isize has to be a multiple of 16"

#         cngf, tisize = ngf // 2, 4
#         while tisize != isize:
#             cngf = cngf * 2
#             tisize = tisize * 2

#         main = nn.Sequential()
#         # input is Z, going into a convolution
#         main.add_module('initial-{0}-{1}-convt'.format(nz, cngf),
#                         nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
#         main.add_module('initial-{0}-batchnorm'.format(cngf),
#                         nn.BatchNorm2d(cngf))
#         main.add_module('initial-{0}-relu'.format(cngf),
#                         nn.ReLU(True))

#         csize, _ = 4, cngf
#         while csize < isize // 2:
#             main.add_module('pyramid-{0}-{1}-convt'.format(cngf, cngf // 2),
#                             nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
#             main.add_module('pyramid-{0}-batchnorm'.format(cngf // 2),
#                             nn.BatchNorm2d(cngf // 2))
#             main.add_module('pyramid-{0}-relu'.format(cngf // 2),
#                             nn.ReLU(True))
#             cngf = cngf // 2
#             csize = csize * 2

#         # Extra layers
#         for t in range(n_extra_layers):
#             main.add_module('extra-layers-{0}-{1}-conv'.format(t, cngf),
#                             nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
#             main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cngf),
#                             nn.BatchNorm2d(cngf))
#             main.add_module('extra-layers-{0}-{1}-relu'.format(t, cngf),
#                             nn.ReLU(True))

#         main.add_module('final-{0}-{1}-convt'.format(cngf, nc),
#                         nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
#         main.add_module('final-{0}-tanh'.format(nc),
#                         nn.Tanh())
#         self.main = main

#     def forward(self, input):
#         if self.ngpu > 1:
#             output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
#         else:
#             output = self.main(input)
#         return output


##
class NetD(nn.Module):
    """
    DISCRIMINATOR NETWORK
    """

    def __init__(self, opt):
        super(NetD, self).__init__()
        model = Encoder(opt.isize, 1, opt.nc, opt.ngf, opt.ngpu, opt.extralayers)
        layers = list(model.main.children())

        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        features = self.features(x)
        features = features
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)

        return classifier, features

##
class NetG(nn.Module):
    """
    GENERATOR NETWORK
    """

    def __init__(self, opt):
        super(NetG, self).__init__()
        self.encoder1 = Encoder(opt.isize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.extralayers)
        self.decoder = Decoder(opt.isize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.extralayers)
        self.encoder2 = Encoder(opt.isize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.extralayers)

    def forward(self, x):
        latent_i = self.encoder1(x)
        gen_imag = self.decoder(latent_i)
        latent_o = self.encoder2(gen_imag)
        return gen_imag, latent_i, latent_o
