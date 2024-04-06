from flask import Flask, render_template, request
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
#cse 2100
#roll:2003038,2003005
# create a web app
app = Flask(__name__)

# Load the emotion detection model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    # Rectified Linear Unit
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# load emotion model weights.h5 modelfile
model.load_weights('emotion_model_weights.h5')

# song
dataset = 'Spotify_Youtube.csv'
df = pd.read_csv(dataset)
emotion_categories = {
    'happy': df.loc[df['Valence'] > 0.7],
    'sad': df.loc[df['Valence'] < 0.3],
    'energetic': df.loc[(df['Valence'] >= 0.3) & (df['Valence'] <= 0.7)]
}

# Function to preprocess the webcam video stream
# def preprocess_video():
#     cap = cv2.VideoCapture(0)
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Detect emotion from the frame
#         emotion = detect_emotion(frame)

#         # Recommend songs based on detected emotion
#         recommended_songs = recommend_songs(emotion)

#         # Display the emotion and recommended songs on the frame
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         cv2.putText(frame, f"Emotion: {emotion}", (10, 30), font, 1, (0, 255, 0), 2)
#         cv2.putText(frame, "Recommended Songs:", (10, 60), font, 1, (0, 255, 0), 2)

#         for i, (song, artist) in enumerate(recommended_songs.iterrows(), start=1):
#             cv2.putText(frame, f"{i}. {song} - {artist}", (10, 60 + i * 30), font, 0.7, (0, 255, 0), 2)

#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#     cap.release()

# Function to preprocess the input image for emotion detection
def preprocess_image(image):
    # Resize the Image
    image = cv2.resize(image, (48, 48))
    # This code checks if the image has more than one channel
    if len(image.shape) > 2 and image.shape[2] > 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # pixel values are normalized to the range [0, 1]
    image = image / 255.0
    #  is used to add an extra dimension 
    image = np.expand_dims(image, axis=-1)
    return image


#  image
def detect_emotion(image):
    preprocessed_image = preprocess_image(image)
    input_image = np.expand_dims(preprocessed_image, axis=0)
    predictions = model.predict(input_image)
    emotion_label = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    predicted_label = emotion_label[np.argmax(predictions)]
    
    # Map detected emotions to categories
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

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST" and "file" in request.files:
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No file selected.")
        
        # Read the image file
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        if image is not None:
            # Perform emotion detection on the image
            detected_emotion = detect_emotion(image)
            recommended_songs = recommend_songs(detected_emotion)
            if not recommended_songs.empty:
                # Convert recommended songs DataFrame to a list of dictionaries
                recommended_songs = recommended_songs.to_dict(orient="records")
            return render_template("index.html", emotion=detected_emotion, recommended_songs=recommended_songs)
        else:
            return render_template("index.html", error="Failed to read the image file.")
    
    return render_template("index.html", error=None)


if __name__ == '__main__':
    app.run(debug=True)
