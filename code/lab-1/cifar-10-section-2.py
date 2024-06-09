# CIFAR-10 Attack
# Section 2

# Apply an adversarial attack
attack = LinfPGD()
epsilons = [0.03]
adversarials = attack(fmodel, images, labels, epsilons=epsilons)

# Extract the first adversarial image
adversarial_image = adversarials[0]

# Ensure adversarial_image is a tensor
if isinstance(adversarial_image, list):
    adversarial_image = adversarial_image[0]

if isinstance(adversarial_image, torch.Tensor):
    adversarial_image = adversarial_image.cpu()

# Display the adversarial image
imshow(adversarial_image, title='Adversarial Image')

# Display the difference
diff = adversarial_image - (images[0] / 2 + 0.5)
imshow(diff, title='Difference Image')
