## Emotion-based Music Recommendation System

Welcome to the Emotion-based Music Recommendation System! This project aims to recommend songs based on the user's current mood. By leveraging machine learning algorithms, the system analyzes emotional features of songs and matches them with user input.

### Web Application

To use the recommendation system, you can run the provided `app.py` file. This will launch a web application where users can input their current mood and receive song recommendations accordingly.

### Usage:

1. **Install Dependencies:** Ensure you have the necessary dependencies installed by running:
   ```
   pip install -r requirements.txt
   ```

2. **Run the Application:** Execute the `app.py` file using Python:
   ```
   python app.py
   ```

3. **Access the Application:** Once the application is running, open your web browser and go to `http://localhost:5000` to access the system.

### Emotion Detection Model

The system utilizes a Convolutional Neural Network (CNN) for emotion detection, implemented using TensorFlow and Keras. The model architecture is as follows:

```python
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
```

This model is trained to detect emotions from facial images. It accepts images of size 48x48 pixels as input and outputs probabilities for 7 different emotions using a softmax activation function.

### Dataset

The dataset used to train the emotion detection model is hosted on Google Drive. You can download it from the following link:
[Download Dataset](https://drive.google.com/drive/folders/1dqkTTuq6LLhauTqAaNvE1lOu3i1fx2XE?usp=drive_link)

Ensure you download the entire dataset folder and follow the provided instructions in the project documentation for preprocessing and training the model.
