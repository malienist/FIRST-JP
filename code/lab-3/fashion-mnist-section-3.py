# SECTION 3

# Apply PGD attack
epsilon = 0.3  # Maximum perturbation
steps = 40  # Number of attack iterations
step_size = 0.01  # Step size
attack = fb.attacks.LinfPGD(steps=steps, rel_stepsize=step_size/epsilon)
raw_advs, clipped_advs, success = attack(fmodel, images, labels, epsilons=epsilon)

# Run post-attack predictions
logits_adv = model(clipped_advs)
post_attack_predictions = torch.argmax(logits_adv, axis=1)

# Visualize pre-attack and post-attack images
def imshow(img, title):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.title(title)
    plt.show()

# Show original image
imshow(torchvision.utils.make_grid(images), title=f'Original Image - Predicted: {pre_attack_predictions.item()}')

# Show adversarial image
imshow(torchvision.utils.make_grid(clipped_advs), title=f'Adversarial Image - Predicted: {post_attack_predictions.item()}')
