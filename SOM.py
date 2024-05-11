import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class SOM:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.random((output_size[0], output_size[1], input_size))

    def train(self, data, num_iterations=10000, learning_rate=0.1, sigma=10):
        for iteration in range(num_iterations):
            # Select a random data point
            data_point = data[np.random.randint(len(data))]

            # Find the best matching unit (BMU)
            bmu_index = self.find_bmu(data_point)

            # Update weights of BMU and its neighbors
            self.update_weights(bmu_index, data_point, iteration, num_iterations, learning_rate, sigma)

    def find_bmu(self, data_point):
        # Calculate Euclidean distances between data point and SOM weights
        distances = np.linalg.norm(self.weights - data_point, axis=2)

        # Find the index of the best matching unit (BMU)
        bmu_index = np.unravel_index(np.argmin(distances, axis=None), distances.shape)

        return bmu_index

    def update_weights(self, bmu_index, data_point, iteration, num_iterations, learning_rate, sigma):
        # Calculate neighborhood radius
        radius = sigma * np.exp(-iteration / num_iterations)

        # Update weights of BMU and its neighbors
        for i in range(self.output_size[0]):
            for j in range(self.output_size[1]):
                # Calculate the distance between neuron and BMU
                distance = np.linalg.norm(np.array([i, j]) - np.array(bmu_index))

                # Calculate the influence of the neuron on BMU
                influence = np.exp(-distance ** 2 / (2 * radius ** 2))

                # Update weights
                self.weights[i, j] += influence * learning_rate * (data_point - self.weights[i, j])

    def compress_image(self, input_image_path, output_image_path):
        # Load the image
        image = Image.open(input_image_path)

        # Convert image to numpy array
        image_array = np.array(image)

        # Normalize pixel values
        image_array = image_array / 255.0

        # Get image dimensions
        height, width, channels = image_array.shape

        # Reshape image array
        image_array = image_array.reshape(-1, channels)

        # Train the SOM
        self.train(image_array)

        # Create a compressed image
        compressed_image = np.zeros(image_array.shape)

        # Assign each pixel of the image to the closest neuron's weight
        for i in range(image_array.shape[0]):
            bmu_index = self.find_bmu(image_array[i])
            compressed_image[i] = self.weights[bmu_index]

        # Convert compressed image back to the original scale
        compressed_image *= 255

        # Reshape compressed image array
        compressed_image = compressed_image.reshape(height, width, channels)

        # Save the compressed image
        compressed_image = compressed_image.astype(np.uint8)
        compressed_image = Image.fromarray(compressed_image)
        compressed_image.save(output_image_path)

        print("Image compressed and saved successfully!")


def compare_images(original_image, original_size,compressed_image,compressed_size):
    # Set up the figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display original image
    axes[0].imshow(original_image)
    axes[0].set_title(f'Original Image - Size: {original_size} KB')

    # Display compressed image
    axes[1].imshow(compressed_image)
    axes[1].set_title(f'Compressed Image - Size: {compressed_size} KB')

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.4)

    # Show plot
    plt.show()

def performance_measurement(original_size, compressed_size):
    # Calculate compression ratio
    compression_ratio = original_size / compressed_size

    # Create a table
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.axis('off')  # Turn off axis

    # Define table data
    table_data = [
        ['Compression Ratio', f'{compression_ratio:.2f}']
    ]

    # Create table
    table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 1.5)  # Scale the table for better visibility

    plt.show()


# Example usage
input_image_path = "input_image.jpg"
output_image_path = "compressed_image.jpg"
input_size = 3  # Number of channels
output_size = (20, 20)  # SOM grid size
num_iterations = 10000
learning_rate = 0.1
sigma = 10

# Load original image
original_image = Image.open(input_image_path)
original_size = os.path.getsize(input_image_path) // 1024  # Size in KB

som = SOM(input_size, output_size)
som.compress_image(input_image_path, output_image_path)

# Load compressed image
compressed_image = Image.open(output_image_path)
compressed_size = os.path.getsize(output_image_path) // 1024  # Size in KB

compare_images(original_image, original_size,compressed_image,compressed_size)
performance_measurement(original_size, compressed_size)