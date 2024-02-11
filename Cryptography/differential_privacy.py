"""
This is an implementation of Differential privacy using Opacus library.
I took an image classification task on CIFAR10 dataset and used a pre trained resnet model.
Tuning Max Grad Norm is an important step in DP. I started out with a low value of 0.1 and then increased it to 1.2.
Often in DP, to achieve better accuracy, we pre train the model on a public dataset and then train it on the private dataset.
Also, Making the model Differentially private is enough of a regularization that we don't need to use Dropout.
"""

from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
import numpy as np
import torch.optim as optim
import torch.nn as nn
from opacus.validators import ModuleValidator
from torchvision import models
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torchvision
import torch
import warnings
warnings.simplefilter("ignore")

# One common technique is to clip the gradients during training, and max_grad_norm is the threshold used for this purpose.
MAX_GRAD_NORM = 1.2  # The maximum L2 norm of per-sample gradients
EPSILON = 50.0
DELTA = 1e-5  # it should be set to be less than the inverse of the size of the training dataset
EPOCHS = 20
LR = 1e-3 # Learning rate
BATCH_SIZE = 512
MAX_PHYSICAL_BATCH_SIZE = 128

# These values, specific to the CIFAR10 dataset, are assumed to be known.
# If necessary, they can be computed with modest privacy budgets.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)

# transformers are used to perform common image transformations.
transform = transforms.Compose([
    transforms.ToTensor(), # converts a PIL image to tensor
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV), # normalizes the tensor with mean and standard deviation
])

# Load the datasets and specify the loaders. 
DATA_ROOT = './dataset/dp'

train_dataset = CIFAR10(
    root=DATA_ROOT, train=True, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
)

test_dataset = CIFAR10(
    root=DATA_ROOT, train=False, download=True, transform=transform)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# load a pre trained resnet model with 10 output classes. 
model = models.resnet18(num_classes=10)

# the privacy engine of opacus is not compatible with all the pytorch models.

errors = ModuleValidator.validate(model, strict=False)
print(errors[-5:])

model = ModuleValidator.fix(model)
ModuleValidator.validate(model, strict=False)

# check whether GPU is available or not and then move the model to the GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# It combines a softmax activation function and a a negative log-likelihood loss function in one single class.
criterion = nn.CrossEntropyLoss()
# the RMSprop optimizer is a variant of the SGD algorithm
optimizer = optim.RMSprop(model.parameters(), lr=LR)

# Function to get the accuracy of the model
def accuracy(preds, labels):
    return (preds == labels).mean()


# now attatch the privacy engine with the hyperparameters defined above
privacy_engine = PrivacyEngine()

model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    epochs=EPOCHS,
    target_epsilon=EPSILON,
    target_delta=DELTA,
    max_grad_norm=MAX_GRAD_NORM,
)

print(f"Using sigma={optimizer.noise_multiplier} and C={MAX_GRAD_NORM}")

# here we train the model
def train(model, train_loader, optimizer, epoch, device):
    model.train()
    criterion = nn.CrossEntropyLoss()

    losses = []
    top1_acc = []

    with BatchMemoryManager(
        data_loader=train_loader,
        max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
        optimizer=optimizer
    ) as memory_safe_data_loader:

        for i, (images, target) in enumerate(memory_safe_data_loader):
            optimizer.zero_grad()
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()

            # measure accuracy and record loss
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

            loss.backward()
            optimizer.step()

            if (i+1) % 200 == 0:
                epsilon = privacy_engine.get_epsilon(DELTA)
                print(
                    f"\tTrain Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"Acc@1: {np.mean(top1_acc) * 100:.6f} "
                    f"(ε = {epsilon:.2f}, δ = {DELTA})"
                )

# here we test the model
def test(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    top1_acc = []

    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

    top1_avg = np.mean(top1_acc)

    print(
        f"\tTest set:"
        f"Loss: {np.mean(losses):.6f} "
        f"Acc: {top1_avg * 100:.6f} "
    )
    return np.mean(top1_acc)

from tqdm.notebook import tqdm

for epoch in tqdm(range(EPOCHS), desc="Epoch", unit="epoch"):
    train(model, train_loader, optimizer, epoch + 1, device)

top1_acc = test(model, test_loader, device)
print(top1_acc)
