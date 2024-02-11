# After understanding the paper on GANs by Alec Radford & Luke Metz, Here is the implementation of the same.

"""
We will be using the CIFAR-10 dataset to train the DCGAN. The dataset consists of 60000 32x32 colour images in
10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. We will be using
the training images to train the DCGAN.

Also taking all the hyper-parameters as it is specified in the paper. So we will be using a batch size of 128,a LeakyReLU with a slope of 0.2, a tanh activation function for the generator output, a Sigmoid activation function for the discriminator output, a learning rate of 0.0002, a beta1 value of 0.5, and a latent vector size of 100.
"""

# Importing the libraries
import torch
import torch.nn as nn
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
import numpy as np
import matplotlib
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm
matplotlib.style.use('ggplot')

# ANSI escape code for colors


class colors:
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'


# learning parameters / configurations according to paper
image_size = 64  # we need to resize image to 64x64
batch_size = 128 # given in the paper
nz = 100  # latent vector size
beta1 = 0.5  # beta1 value for Adam optimizer, basically momentum parameter
lr = 0.0002  # learning rate according to paper
sample_size = 64  # fixed sample size
epochs = 25  # number of epoch to train

# set the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def label_real(size):
    """
    Fucntion to create real labels (ones)
    :param size: batch size
    :return real label vector
    """
    data = torch.ones(size, 1)
    return data.to(device)


def label_fake(size):
    """
    Fucntion to create fake labels (zeros)
    :param size: batch size
    :returns fake label vector
    """
    data = torch.zeros(size, 1)
    return data.to(device)


def create_noise(sample_size, nz):
    """
    Fucntion to create noise
    :param sample_size: fixed sample size or batch size
    :param nz: latent vector size
    :returns random noise vector
    """
    return torch.randn(sample_size, nz, 1, 1).to(device)


def save_generator_image(image, path):
    """
    Function to save torch image batches
    :param image: image tensor batch
    :param path: path name to save image
    """
    save_image(image, path, normalize=True)


def weights_init(m):
    """
    This function initializes the model weights randomly from a 
    Normal distribution. This follows the specification from the DCGAN paper.
    If the class name contains the substring 'Conv', it means the module is a convolutional layer. In this case, 
    the function initializes the weights of the convolutional layer from a normal distribution with mean 0.0 and 
    standard deviation 0.02.
    If the class name contains the substring 'BatchNorm', it means the module is a batch normalization layer. In 
    this case, the function initializes the weights from a normal distribution with mean 1.0 and standard 
    deviation 0.02. Additionally, it sets the bias terms to 0.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# generator

# Output size=(Input size−1)×stride−2×padding+kernel size
class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.nz = nz  # The noise vector that the generator takes as input, nz = 100
        self.main = nn.Sequential(
            # We use the sequential container to build the generator network
            # nz will be the input to the first convolution
            # State size. (nz) x 1 x 1
            nn.ConvTranspose2d(  # First Reverse Convolution layer with stride 1, padding 0
                nz, 512, kernel_size=4,
                stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # State size. (512) x 4 x 4
            # Rest 4 convolutional layers with stride 2, padding 1
            nn.ConvTranspose2d(
                512, 256, kernel_size=4,
                stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # State size. (256) x 8 x 8
            nn.ConvTranspose2d(
                256, 128, kernel_size=4,
                stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # State size. (128) x 16 x 16
            nn.ConvTranspose2d(
                128, 64, kernel_size=4,
                stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # State size. (64) x 32 x 32
            nn.ConvTranspose2d(
                64, 3, kernel_size=4,
                stride=2, padding=1, bias=False),
            # With each subsequent Reverse convolution operation, we keep on reducing the output channels. We start from
            # 512 output channels and have 3 output channels after the last convolution operation. 512 => 256 =>
            # 128 => 64 => 3. This 3 refers to the three channels (RGB) of the colored images
            nn.Tanh()
            # The output of the last convolution layer is a 3x64x64 tensor. This tensor is the colored image generated
            # finally a Tanh activation function to get the pixel values in the range [-1,1]
        )

    def forward(self, input):
        return self.main(input)


# discriminator
class Discriminator(nn.Module):
    # The discriminator is just the reverse of the Generator class. A look at strides and padding will make it
    # clear.
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(  # We use the sequential container to build the discriminator network
            # State size. (3) x 64 x 64
            nn.Conv2d(
                3, 64, kernel_size=4,
                stride=2, padding=1, bias=False),
            # 3 input channels for the colored images
            # We will take the hyper-parameters as it is specified in the paper. So We specify the slope of the LeakyReLU as 0.2
            nn.LeakyReLU(0.2, inplace=True),
            # State size. (64) x 32 x 32
            nn.Conv2d(
                64, 128, kernel_size=4,
                stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # State size. (128) x 16 x 16
            nn.Conv2d(
                128, 256, kernel_size=4,
                stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # State size. (256) x 8 x 8
            nn.Conv2d(
                256, 512, kernel_size=4,
                stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # State size. (512) x 4 x 4
            nn.Conv2d(
                512, 1, kernel_size=4,
                stride=1, padding=0, bias=False),
            # The output of the last convolution layer is a single number. This number is the probability of the
            # input image being real or fake. We use a Sigmoid activation function to get the probability.
            nn.Sigmoid()
            #State size. (1) x 1 x 1
        )

    def forward(self, input):
        # The forward() function forward passes either the real image or fake image batch through the
        # discriminator network. Then the discriminator returns the binary classifications for the batch.
        return self.main(input)

# image transforms
# resize the image, convert the values to tensors, and normalize the values as well. The resizing
# dimensions are 64×64.


transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
])

# prepare the data
train_data = datasets.CIFAR10(
    root='dataset/dcgan/',
    train=True,
    download=True,
    transform=transform
)

# Get the Loader
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# initialize models
generator = Generator(nz).to(device)
discriminator = Discriminator().to(device)

# initialize generator weights
generator.apply(weights_init)

# initialize discriminator weights
discriminator.apply(weights_init)

print(colors.BLUE + '##### GENERATOR #####' + colors.RESET)
print(colors.CYAN + str(generator) + colors.RESET)
print(colors.BLUE + '######################' + colors.RESET)
print(colors.BLUE + '\n##### DISCRIMINATOR #####' + colors.RESET)
print(colors.CYAN + str(discriminator) + colors.RESET)
print(colors.BLUE + '######################' + colors.RESET)

# optimizers
optim_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optim_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# loss function
criterion = nn.BCELoss()

losses_g = []  # to store generator loss after each epoch
losses_d = []  # to store discriminator loss after each epoch

# function to train the discriminator network


def train_discriminator(optimizer, data_real, data_fake):
    b_size = data_real.size(0) # get the batch size
    # get the real label vector
    real_label = label_real(b_size)
    real_label = real_label.squeeze() # remove an extra dimension
    # get the fake label vector
    fake_label = label_fake(b_size)
    fake_label = fake_label.squeeze()
    optimizer.zero_grad()
    # get the outputs by doing real data forward pass
    output_real = discriminator(data_real).view(-1) # reshape to a one dimensional vector, -1 is used to infer the size along that dimension. So, -1 here is equivalent to 1*512*4*4
    loss_real = criterion(output_real, real_label)
    # get the outputs by doing fake data forward pass
    output_fake = discriminator(data_fake)
    output_fake = output_fake.squeeze()
    loss_fake = criterion(output_fake, fake_label)
    # compute gradients of real loss
    loss_real.backward()
    # compute gradients of fake loss
    loss_fake.backward()
    # update discriminator parameters
    optimizer.step()
    return loss_real + loss_fake

# function to train the generator network


def train_generator(optimizer, data_fake):
    b_size = data_fake.size(0)
    # get the real label vector
    real_label = label_real(b_size)
    real_label = real_label.squeeze()
    optimizer.zero_grad()
    # output by doing a forward pass of the fake data through discriminator
    output = discriminator(data_fake)
    output = output.squeeze()
    loss = criterion(output, real_label)
    # compute gradients of loss
    loss.backward()
    # update generator parameters
    optimizer.step()
    return loss


# create the noise vector
noise = create_noise(sample_size, nz)

# switching the generator and discriminator networks to training mode.
generator.train()
discriminator.train()

# a simple for loop to train the generator and discriminator network. We will be training the networks for 25 epochs.
for epoch in range(epochs):
    loss_g = 0.0
    loss_d = 0.0
    for bi, data in tqdm(enumerate(train_loader), total=int(len(train_data)/train_loader.batch_size)):
        image, _ = data
        image = image.to(device)
        b_size = len(image)
        # forward pass through generator to create fake data
        data_fake = generator(create_noise(b_size, nz)).detach()
        data_real = image
        loss_d += train_discriminator(optim_d, data_real, data_fake)
        data_fake = generator(create_noise(b_size, nz))
        loss_g += train_generator(optim_g, data_fake)
    # final forward pass through generator to create fake data after training for current epoch to save the image
    # locally
    generated_img = generator(noise).cpu().detach()
    # save the generated torch tensor models to disk
    save_generator_image(generated_img, f"./outputs/gen_img_{epoch}.png")
    epoch_loss_g = loss_g / bi  # total generator loss for the epoch
    epoch_loss_d = loss_d / bi  # total discriminator loss for the epoch
    losses_g.append(epoch_loss_g)
    losses_d.append(epoch_loss_d)
    print(colors.BLUE + f"Epoch {epoch + 1} of {epochs}" + colors.RESET)
    print(colors.CYAN +
          f"Generator loss: {epoch_loss_g:.8f}, Discriminator loss: {epoch_loss_d:.8f}" + colors.RESET)

# save the model parameters for inference
print(colors.GREEN + 'DONE TRAINING' + colors.RESET)
torch.save(generator.state_dict(), './results/generator.pth')

# plot and save the generator and discriminator loss
plt.figure()
plt.plot([loss.item() for loss in losses_g], label='Generator loss')
plt.plot([loss.item() for loss in losses_d], label='Discriminator Loss')
plt.legend()
plt.savefig('./results/loss.png')
plt.show()
