# SECTION 2

import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import foolbox as fb


# Load Fashion-MNIST dataset
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)


# Load the model (for attack)
model.load_state_dict(torch.load('simple_cnn.pth'))
model.eval()

# Wrap the model with Foolbox
fmodel = fb.PyTorchModel(model, bounds=(-1, 1))

# Fetch a sample from the dataset
images, labels = next(iter(testloader))
images, labels = images.to(torch.device('cpu')), labels.to(torch.device('cpu'))

# Run pre-attack predictions
logits = model(images)
pre_attack_predictions = torch.argmax(logits, axis=1)
