import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class MiniSom:
    def __init__(self, x, y, input_len, sigma=1.0, learning_rate=0.5):
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.weights = np.random.rand(x, y, input_len)

    def find_bmu(self, x):
        dists = np.linalg.norm(self.weights - x, axis=-1)
        return np.unravel_index(np.argmin(dists), dists.shape)

    def update_weights(self, x, bmu, t, max_iter):
        lr = self.learning_rate * np.exp(-t / max_iter)
        sigma = self.sigma * np.exp(-t / max_iter)
        dx = np.arange(self.weights.shape[0])[:, None] - bmu[0]
        dy = np.arange(self.weights.shape[1])[None, :] - bmu[1]
        dist_sq = dx**2 + dy**2
        influence = np.exp(-dist_sq / (2 * sigma**2))
        self.weights += lr * influence[..., np.newaxis] * (x - self.weights)

    def train(self, data, max_iter):
        for t in range(max_iter):
            for x in data:
                bmu = self.find_bmu(x)
                self.update_weights(x, bmu, t, max_iter)

    def compress_image(self, image_path, new_width, new_height, max_iter=100):
        image = Image.open(image_path).resize((new_width, new_height))
        image_array = np.array(image) / 255.0
        flattened_image = image_array.reshape(-1, 3)
        self.train(flattened_image, max_iter)
        bmu_indices = np.array([self.find_bmu(x) for x in flattened_image])
        compressed_image = self.weights[bmu_indices[:, 0], bmu_indices[:, 1]].reshape(new_width, new_height, 3)
        return compressed_image, image

# Function to display images and information
def display_images_and_info(images_and_sizes, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(20, 12))

    for i, (image, size) in enumerate(images_and_sizes):
        # Display image
        axes[i].imshow(image)
        axes[i].axis('off')

        # Display image resolution and image size with enlarged font
        axes[i].set_title(f'Original Image  -  Size: {size:.2f} KB' if i == 0 else f'Compressed Image  -  Size: {size:.2f} KB', fontsize=16)

    plt.subplots_adjust(wspace=0.1, hspace=0)
    
    # Save the figure if save_path is provided
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Example usage:
som = MiniSom(10, 10, 3)
images_and_sizes = []

# Original Image
original_image = Image.open("input_image.jpg")
original_image_size = original_image.size[0] * original_image.size[1] * len(original_image.getbands()) / 1024.0
original_image = np.array(original_image)
images_and_sizes.append((original_image, original_image_size))

# Compressed Image
compressed_image, _ = som.compress_image("input_image.jpg", 100, 100)
compressed_image_size = compressed_image.nbytes / 1024.0
images_and_sizes.append((compressed_image, compressed_image_size))

# Display images and information and save the figure
display_images_and_info(images_and_sizes, save_path="output_figure.png")
