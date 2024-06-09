# SECTION 5
# PERTURBATION CONT'D


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

# Apply PGD attack
epsilon = 0.3  # Maximum perturbation
steps = 40  # Number of attack iterations
step_size = 0.01  # Step size
attack = fb.attacks.LinfPGD(steps=steps, rel_stepsize=step_size/epsilon)
raw_advs, clipped_advs, success = attack(fmodel, images, labels, epsilons=epsilon)

# Run post-attack predictions
logits_adv = model(clipped_advs)
post_attack_predictions = torch.argmax(logits_adv, axis=1)

# Visualize pre-attack and post-attack images, and the perturbation
def imshow(img, title):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.title(title)
    plt.axis('off')

# Calculate perturbation
perturbation = clipped_advs - images

# Show original, adversarial, and perturbation images side by side
plt.figure(figsize=(15, 5))

# Original Image
plt.subplot(1, 3, 1)
imshow(torchvision.utils.make_grid(images),
       title=f'Original Image\nPredicted: {class_names[pre_attack_predictions.item()]}')

# Adversarial Image
plt.subplot(1, 3, 2)
imshow(torchvision.utils.make_grid(clipped_advs),
       title=f'Adversarial Image\nPredicted: {class_names[post_attack_predictions.item()]}')

# Perturbation
plt.subplot(1, 3, 3)
imshow(torchvision.utils.make_grid(perturbation),
       title='Perturbation')

plt.tight_layout()
plt.show()
