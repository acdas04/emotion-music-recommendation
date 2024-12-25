from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

# cse 2100
# roll: 2003038, 2003005
# create a web app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')  # Set the upload folder path

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the emotion detection model
model = tf.keras.models.Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(7, activation='softmax'))

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Load emotion model weights
model.load_weights('/Users/abirchandradas/Desktop/python/music recommendation based on facial expression/model1.h5')

# Load song dataset
dataset = '/Users/abirchandradas/Desktop/python/music recommendation based on facial expression/Spotify_Youtube.csv'
df = pd.read_csv(dataset)
emotion_categories = {
    'happy': df.loc[df['Valence'] > 0.7],
    'sad': df.loc[df['Valence'] < 0.3],
    'energetic': df.loc[(df['Valence'] >= 0.3) & (df['Valence'] <= 0.7)]
}

# Function to preprocess the input image for emotion detection
def preprocess_image(image):
    image = cv2.resize(image, (48, 48))
    if len(image.shape) > 2 and image.shape[2] > 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image / 255.0
    image = np.expand_dims(image, axis=-1)
    return image

# Function to detect emotion from the image
def detect_emotion(image):
    preprocessed_image = preprocess_image(image)
    input_image = np.expand_dims(preprocessed_image, axis=0)
    predictions = model.predict(input_image)
    emotion_label = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    predicted_label = emotion_label[np.argmax(predictions)]
    
    if predicted_label in ['Happy', 'Surprise']:
        return 'happy'
    elif predicted_label in ['Sad', 'Disgust']:
        return 'sad'
    else:
        return 'energetic'

# Function to recommend songs based on emotion category
def recommend_songs(emotion, num_songs=5):
    if emotion in emotion_categories:
        songs = emotion_categories[emotion]
        return songs.sample(num_songs)
    else:
        return pd.DataFrame()  # Return an empty DataFrame if the emotion category is invalid

# Main route for file upload and emotion detection
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST" and "file" in request.files:
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No file selected.")
        
        # Save the uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Read the image
        image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if image is not None:
            detected_emotion = detect_emotion(image)
            recommended_songs = recommend_songs(detected_emotion)
            if not recommended_songs.empty:
                recommended_songs = recommended_songs.to_dict(orient="records")
            
            # Remove the uploaded file after processing
            os.remove(filepath)
            
            return render_template("index.html", emotion=detected_emotion, recommended_songs=recommended_songs)
        else:
            # Remove the uploaded file in case of error
            os.remove(filepath)
            return render_template("index.html", error="Failed to read the image file.")
    
    return render_template("index.html", error=None)

# Route to refresh songs for a specific emotion
@app.route("/refresh_songs/<emotion>")
def refresh_songs(emotion):
    recommended_songs = recommend_songs(emotion)
    if not recommended_songs.empty:
        recommended_songs = recommended_songs.to_dict(orient="records")
    else :
        recommended_songs=recommend_songs('Happy').to_dict(orient="records")
    return render_template("output.html", recommended_songs=recommended_songs, emotion=emotion)

if __name__ == '__main__':
    app.run(debug=True)