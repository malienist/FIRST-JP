# Lab 1
# CIFAR-10 Attack

# START

!pip install torch torchvision matplotlib numpy foolbox

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

import foolbox as fb
from foolbox.attacks import LinfPGD

# Load CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

# Define a simple model (using pretrained ResNet18 for demonstration)
model = torchvision.models.resnet18(pretrained=True)
model.eval()

# Select a sample image from the testset
dataiter = iter(testloader)
images, labels = next(dataiter)

# Display the original image
def imshow(img, title):
    if img.ndim == 4:
        # Display the first image in the batch
        img = img[0]

    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()

imshow(images[0], title='Original Image')
