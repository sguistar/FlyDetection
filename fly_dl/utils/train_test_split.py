import os
import shutil
import random

# Set the path to your images and annotations
image_dir = r'D:\fly\fly_dl\video_process\images\augmented'
annotation_dir = r'D:\fly\fly_dl\video_process\labels\augmented'

# Set the paths for the train and val directories
train_image_dir = r"D:\fly\fly_dl\video_process\images\train"
val_image_dir = r"D:\fly\fly_dl\video_process\images\val"
train_annotation_dir = r"D:\fly\fly_dl\video_process\labels\train"
val_annotation_dir = r"D:\fly\fly_dl\video_process\labels\val"

# Create directories if they don't exist
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(train_annotation_dir, exist_ok=True)
os.makedirs(val_annotation_dir, exist_ok=True)

# Get list of all files
images = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
annotations = [f for f in os.listdir(annotation_dir) if os.path.isfile(os.path.join(annotation_dir, f))]

# Ensure that the number of images and annotations match
assert len(images) == len(annotations), "Number of images and annotations do not match!"

# Combine images and annotations into a list of tuples
data = list(zip(images, annotations))

# Shuffle the data
random.shuffle(data)

# Define the split ratio
train_ratio = 0.8
val_ratio = 0.2

# Calculate the number of training samples
num_train_samples = int(train_ratio * len(data))

# Split the data into training and validation sets
train_data = data[:num_train_samples]
val_data = data[num_train_samples:]

# Copy files to the respective directories
for image, annotation in train_data:
    shutil.copy(os.path.join(image_dir, image), os.path.join(train_image_dir, image))
    shutil.copy(os.path.join(annotation_dir, annotation), os.path.join(train_annotation_dir, annotation))

for image, annotation in val_data:
    shutil.copy(os.path.join(image_dir, image), os.path.join(val_image_dir, image))
    shutil.copy(os.path.join(annotation_dir, annotation), os.path.join(val_annotation_dir, annotation))

print(f"Training data: {len(train_data)} samples")
print(f"Validation data: {len(val_data)} samples")
