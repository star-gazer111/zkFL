'''
The Following is an implementation of a differentially private GAN using Opacus library.
I have Fine tuned the model to get the best results.
Further I have used the MNIST dataset to train the model.
Also one important thing to note is that I have used GroupNorm instead of BatchNorm.
GroupNorm is a normalization technique that is faster and more stable than BatchNorm.
Also BatchNorm cannot work for differential privacy as it uses the batch statistics.
'''

import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from opacus import PrivacyEngine
from tqdm import tqdm

# Setting a random seed ensures reproducibility, as using the same seed will result in the same sequence of random numbers
torch.manual_seed(random.randint(1, 10000))

# Enabling benchmark mode allows CuDNN to select the most efficient convolution algorithms for the given hardware and input size
cudnn.benchmark = True

try:
    dataset = dset.MNIST(
        root="./dataset/dp",
        download=True,
        # A composition of image transformations applied to each sample. In this case, it resizes the image to 64, converts it to a tensor, and normalizes the pixel values to the range [-1, 1] by providing appropriate means and std.
        transform=transforms.Compose(
            [
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
    )
    # Select only the samples with the label 8
    idx = dataset.targets == 8
    dataset.targets = dataset.targets[idx]
    dataset.data = dataset.data[idx]
except ValueError:
    print("Cannot load dataset")

# Creating a Torch DataLoader with 2 workers to load the data in parallel and a batch size of 64
dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=2,
    batch_size=64,
)

# Checking if CUDA is available
device = torch.device("cuda")

# Defining some parameters
ngpu = 1
nz = 100
ngf = 128
ndf = 128
# number of channels in the images. Here I used MNIST dataset which is grayscale, so nc = 1
nc = 1

# Weight initialization for convolutional and batch normalization layers


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        # normal distribution with mean 0 and standard deviation 0.02
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        # normal distribution with mean 1 and standard deviation 0.02
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Defining the generator


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        # Defining the generator as a sequence of convolutional layers
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            # nn.GroupNorm(num_groups, num_in_channels) , here num_in_channels = ngf * 8 = num_out_channels of the previous layer
            nn.GroupNorm(min(32, ndf * 8), ndf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.GroupNorm(min(32, ndf * 4), ndf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.GroupNorm(min(32, ndf * 2), ndf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.GroupNorm(min(32, ndf), ndf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        output = self.main(input)
        return output


# Creating the generator
netG = Generator(ngpu)
netG = netG.to(device)
netG.apply(weights_init)

# Defining the discriminator


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            # since MNIST are grayscale images, nc = 1
            # Rest all is the oposite of the generator. At the end, we have a single value as output, as we expect to get a binary True/False value
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.GroupNorm(min(32, ndf * 2), ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.GroupNorm(min(32, ndf * 4), ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.GroupNorm(min(32, ndf * 8), ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)


# Creating the discriminator
netD = Discriminator(ngpu)
netD = netD.to(device)
netD.apply(weights_init)

# Defining the loss function and optimizer
criterion = nn.BCELoss()

# Creating a fixed noise vector to visualize the progression of the generator
FIXED_NOISE = torch.randn(64, nz, 1, 1, device=device)
REAL_LABEL = 1.0
FAKE_LABEL = 0.0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

# wrap the optimizer with the PrivacyEngine
privacy_engine = PrivacyEngine(secure_mode=False)

netD, optimizerD, dataloader = privacy_engine.make_private(
    module=netD,
    optimizer=optimizerD,
    data_loader=dataloader,
    noise_multiplier=1,
    max_grad_norm=1.0,
)

# setup optimizer
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))


for epoch in range(25):
    data_bar = tqdm(dataloader)
    for i, data in enumerate(data_bar, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        optimizerD.zero_grad(set_to_none=True)

        real_data = data[0].to(device)
        batch_size = real_data.size(0)

        # train with real
        label_true = torch.full((batch_size,), REAL_LABEL, device=device)
        output = netD(real_data)
        errD_real = criterion(output, label_true)
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label_fake = torch.full((batch_size,), FAKE_LABEL, device=device)
        output = netD(fake.detach())
        errD_fake = criterion(output, label_fake)

        # below, you actually have two backward passes happening under the hood
        # which opacus happens to treat as a recursive network
        # and therefore doesn't add extra noise for the fake samples
        # noise for fake samples would be unnecesary to preserve privacy

        errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step()
        optimizerD.zero_grad(set_to_none=True)

        D_G_z1 = output.mean().item()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        optimizerG.zero_grad()

        label_g = torch.full((batch_size,), REAL_LABEL, device=device)
        output_g = netD(fake)
        errG = criterion(output_g, label_g)
        errG.backward()
        D_G_z2 = output_g.mean().item()
        optimizerG.step()

        epsilon = privacy_engine.accountant.get_epsilon(delta=1e-5)
        data_bar.set_description(
            f"epoch: {epoch}, Loss_D: {errD.item()} "
            f"Loss_G: {errG.item()} D(x): {D_x} "
            f"D(G(z)): {D_G_z1}/{D_G_z2}"
            "(ε = %.2f, δ = %.2f)" % (epsilon, 1e-5)
        )

        if i % 100 == 0:
            vutils.save_image(
                real_data, "%s/real_samples.png" % ".", normalize=True
            )
            fake = netG(FIXED_NOISE)
            vutils.save_image(
                fake.detach(),
                "%s/fake_samples_epoch_%03d.png" % (".", epoch),
                normalize=True,
            )
