"""
For text sentiment classification, each sentence of the complete
transcript is treated as a separate observation to predict the relevant sentiment.  The division in sentences is done to keep the context of the individual words for better understanding and prediction.
Hence, for real-time analysis, the model waits for the completion of a sentence to give the final prediction for the same. Each of the sentences is processed further in the following order:
1. removal of punctuation,
2. lowering of capital letters,
3. tokenization,
4. removal of stop words, and
5. lemmatization.

The resultant string with only keywords is ready for input in the text emotion recognition model."""

import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download("stopwords")
nltk.download("punkt_tab")
nltk.download("wordnet")

table = str.maketrans("", "", string.punctuation)
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(transcript: str):
    sentences = sent_tokenize(transcript)
    processed_sentences = []

    for s in sentences:
        s = s.translate(table).lower()
        tokens = word_tokenize(s)
        tokens = [i for i in tokens if i not in stop_words]
        tokens = [lemmatizer.lemmatize(i) for i in tokens]

        ps = " ".join(tokens)
        processed_sentences.append(ps)

    return " ".join(processed_sentences)


"""The dataset for audio emotion recognition consists of many details about each of the audio files, which contribute to the input for the emotion recognition model. Out of this information, only the gender of the actor is used. Since gender is already provided, other extracted features from this dataset are used to create a Random Forest Classifier model that can effectively predict the gender of the speakers where it is not present.

Multiple studies have shown a positive impact of gender knowledge on audio emotion recognition (Thakare, Chaurasia, Rathod, Joshi, & Gudadhe, 2021); hence, a specific model for gender prediction has been created to improve prediction accuracy for unknown videos. Since the audio files in the training subset are 3 seconds long, the audio from the complete video in testing is also processed in fragments of 3 seconds.

From each of the fragments, multiple features are extracted using the Librosa library in terms of arrays and numerical values and stored in multi-dimensional arrays. The features include:

1. Absolute Short-Time Fourier Transform (STFT)
2. An average of 40 Mel-Frequency Cepstral Coefficients (MFCCs)
3. Chromogram
4. Mel-Scaled Spectrogram
5. Average Root Mean Square (RMS) value
6. Average energy contrast between the highest and lowest energy band
7. Average Tonal Centroid features
  
These features result in an array of 195 values, which are used for gender prediction. The predicted gender is then added as a feature for sentiment prediction."""


import librosa
import numpy as np


def extract_audio_features(audio_file, sampling_rate=22050, duration=3):
    y, _ = librosa.load(audio_file, mono=True)

    samples = sampling_rate * duration
    num_fragments = len(y) // samples

    features = []

    for i in range(num_fragments):
        start = i * samples
        end = start + samples
        fragment = y[start:end]

        if len(fragment) < samples:
            continue

        # Absolute Short-time fourier transofrm - 29
        stft = np.abs(librosa.stft(fragment)).mean(axis=1)
        # average mel-frquenct cepstral coefficients - 40
        mfcc = librosa.feature.mfcc(y=fragment, sr=sampling_rate, n_mfcc=40).mean(axis=1)
        # chrmogram - 12
        chroma = librosa.feature.chroma_stft(y=fragment, sr=sampling_rate).mean(axis=1)
        # melspectrogram - 100
        mel_spectrogram = librosa.feature.melspectrogram(y=fragment, sr=sampling_rate).mean(axis=1)
        # average root mean sqare - 1
        rms = librosa.feature.rms(y=fragment).mean()
        # average energy contrast between bands - 7
        contrast = librosa.feature.spectral_contrast(y=fragment, sr=sampling_rate).mean(axis=1)
        # average tonal centroid features - 6
        tonal_centroid = librosa.feature.tonnetz(y=fragment, sr=sampling_rate).mean(axis=1)
        # 195 values array
        fragment_features = np.hstack([stft[:29], mfcc, chroma, mel_spectrogram[:100], [rms], contrast, tonal_centroid])
        features.append(fragment_features)

    return np.array(features)


"""For image emotion recognition, each of the images in the dataset is
pre-processed. The ImageDataGenerator is used to create more image
data with variations in the angle of the image, stretching, zooming,
etc. keeping the target size as an image of 48*48 size. The data is
ready for input in the 2D CNN model for sentiment prediction. For the
multimodal dataset, the frame is extracted at intervals of one second
and run through the pre-trained Deepface module for Python with
'opencv' as the backend for face detection. If the face is detected, the
frame is converted to a Grayscale and cropped to have only the face
in the frame. The image is sharpened by performing convolution of the
image with a 3 x 3 sharpening filter,  =
[ 0 -1 0
-1 5 -1
0 -1 0]. 
The image is then resized to 48*48 size. 
This resized image is then fed into the model for prediction. 
The convolution operation - [𝐼 * 𝐾](𝑥, 𝑦) = Σ𝑚 Σ𝑛 𝐼(𝑥 - 𝑚, 𝑦 - 𝑛) ⋅ 𝐾(𝑚, 𝑛)
where the output is the result of convolution, 
    𝐼 represents image matrix,
    𝐾 represents the sharpening kernel defined above, 
    (𝑥, 𝑦) represents the location in the output image, 
    and 𝑚 and 𝑛 are the coordinates within the kernel. 
It involves sliding a convolution kernel over the image,
computing the element-wise product at each position, and summing
the results to produce the convolved output, which emphasizes image
features and patterns."""

import cv2
import numpy as np
from deepface import DeepFace
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rescale=1.0 / 255,
)


def extract_frames(video_path, interval=1):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frames = []
    success, frame = cap.read()
    count = 0

    while success:
        if count % (fps * interval) == 0:
            frames.append(frame)
        success, frame = cap.read()
        count += 1

    cap.release()
    return frames


def detect_and_crop_face(frame):
    detected_face = DeepFace.extract_faces(frame, detector_backend="opencv", enforce_detection=False)
    largest_face = None
    for face in detected_face:
        if largest_face is None:
            largest_face = face
        elif face["confidence"] > largest_face["confidence"]:
            largest_face = face

    face_image = largest_face["face"]
    if face_image.dtype != np.uint8:
        face_image = (face_image * 255).astype(np.uint8)

    gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    return gray_face


def sharpen_and_resize_image(image, size=(48, 48)):
    kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ]
    )
    sharpened_image = cv2.filter2D(image, -1, kernel)
    resized_image = cv2.resize(sharpened_image, size, interpolation=cv2.INTER_AREA)
    return resized_image


def preprocess_video(filepath):
    frames = extract_frames(filepath)
    preprocessed_images = []

    for frame in frames:
        gray_face = detect_and_crop_face(frame)

        if gray_face is not None:
            final_image = sharpen_and_resize_image(gray_face)
            if final_image is not None:
                final_image = final_image / 255.0
                final_image = final_image.reshape(48, 48, 1)
                preprocessed_images.append(final_image)
    return np.array(preprocessed_images)
