import imageio
import os

# Path to the directory containing your images
images_directory = './results'

# List all image files in the directory
image_files = [f for f in os.listdir(images_directory) if f.endswith('.png')]

# Sort the image files to maintain the order
image_files.sort()

# Output GIF file path
gif_output_path = './results/output.gif'

# Create a list to store the images
images = []

# Read each image and append it to the list
for image_file in image_files:
    image_path = os.path.join(images_directory, image_file)
    images.append(imageio.imread(image_path))

# Save the list of images as a GIF
imageio.mimsave(gif_output_path, images, duration=0.1)  # You can adjust the duration between frames

print(f'GIF created: {gif_output_path}')
