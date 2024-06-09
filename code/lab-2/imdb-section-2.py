# IMDB Attack
# Section 2

# Select multiple samples from the dataset
num_samples = 10  # Specify the number of samples you want to attack
samples = [(dataset[i][0]['text'], dataset[i][1]) for i in range(num_samples)]

# Create a custom dataset for TextAttack
custom_dataset = Dataset(samples, input_columns=["text"])

# Define the attack method
attack = BAEGarg2019.build(model_wrapper)

# Perform the attack
attacker = Attacker(attack, custom_dataset)

# Attack each sample individually and collect results
results = attacker.attack_dataset()

# Print the adversarial text and predictions
for result in results:
    adversarial_text = result.perturbed_text()
    adversarial_result = pipe([adversarial_text])
    print(f"Adversarial text: {adversarial_text}")
    print(f"Adversarial prediction: {adversarial_result}")
