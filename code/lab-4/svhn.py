# CHALLENGE 
# START

!pip install torch torchvision foolbox matplotlib numpy

# Import required libraries

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import SVHN
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import foolbox as fb
from foolbox.attacks import LinfPGD
from foolbox.criteria import Misclassification

# Load SVHN dataset
transform = transforms.Compose([transforms.ToTensor()])
testset = SVHN(root='./data', split='test', download=True, transform=transform)
testloader = DataLoader(testset, batch_size=1, shuffle=True)

# Define a simple model (using pretrained ResNet18 for demonstration)
model = torchvision.models.resnet18(pretrained=True)
model.eval()

# Create Foolbox model
fmodel = fb.PyTorchModel(model, bounds=(0, 1))

# Select a sample image from the testset
images, labels = next(iter(testloader))

# Apply an adversarial attack
attack = fb.attacks.FGSM()
criterion = Misclassification(labels) # Pass the true labels to the Misclassification criterion
epsilons = [0.03]
adversarials = attack(fmodel, images, criterion, epsilons=epsilons) # Generate adversarial images

# Convert the adversarials variable to a tensor
adversarial = adversarials[0][0].cpu().numpy().squeeze() # Select the first adversarial image
plt.imshow(adversarial.transpose(1, 2, 0))
plt.title("Adversarial Image")
plt.show()

# Convert the adversarial variable to a PyTorch tensor
adversarial = torch.from_numpy(adversarial)

# Convert the adversarial variable to a 4-dimensional tensor
adversarial = adversarial.unsqueeze(0)

# Check if the attack was successful
print("Original label:", labels.item())
print("Adversarial label:", torch.argmax(model(adversarial), dim=1).item())
