import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Read the CSV file
data = pd.read_csv('train.csv')
# Access the image pixels and emotion labels
pixels = data['pixels']
emotions = data['emotion']

# Preprocess image pixel data
image_pixels = pixels.apply(lambda x: np.array(x.split(), dtype=np.uint8).reshape(48, 48, 1))

# Normalize the pixel values
image_pixels = image_pixels / 255.0
# 0 to 1

# Convert emotion labels to categorical
emotions_categorical = tf.keras.utils.to_categorical(emotions, num_classes=7)
# [1,0,0,0,0,0,0,0] ------> it means happy

# [0,1,0,0,0,0,0,0] ------> it means sad

# Split the dataset into training and validation sets
train_pixels, val_pixels, train_emotions, val_emotions = train_test_split(image_pixels, emotions_categorical, test_size=0.2, random_state=42)

# Define a generator function for the training data
def train_data_generator():
    for pixels, emotions in zip(train_pixels, train_emotions):
        yield pixels, emotions

# Create a TensorFlow dataset from the generator
train_dataset = tf.data.Dataset.from_generator(train_data_generator, output_signature=(
    tf.TensorSpec(shape=(48, 48, 1), dtype=tf.float32),
    tf.TensorSpec(shape=(7,), dtype=tf.float32)
))

# Shuffle and batch the dataset
#  number of samples to work 
batch_size = 32
train_dataset = train_dataset.shuffle(len(train_pixels)).batch(batch_size)

# model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(7, activation='softmax')
])
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#here we  Train the model
epochs = 50
model.fit(train_dataset, epochs=epochs)
# Save the trained model weights
model.save_weights('emotion_model_weights.h5')