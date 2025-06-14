import streamlit as st
import tensorflow as tf
import numpy as np
import librosa 
from tensorflow.image import resize

def load_model():
    model = tf.keras.models.load_model("./Trained_model.keras")
    return model

def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    chunk_duration = 4
    overlap_duration = 2

    #convert durations to samples
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate

    #calculate the no. of chunks
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

    #iterate over each chunk
    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)

    return np.array(data)

#MODEL PREDICTION
def model_prediction(X_test):
    model = load_model()
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred,axis=1)
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    #print (unique elements, counts)
    max_count = np.max(counts)
    max_elements = unique_elements[counts == max_count]
    return max_elements[0]

##STREAMLIT UI
st.sidebar.title("Dashboard")

app_mode = st.sidebar.selectbox("Select Page",["Home page","Prediction"])

#Main page
if(app_mode=="Home page"):
    st.markdown(
    """
    <style>
    .stApp {
        background-color: #181646; /*Blue background */
        color: white;
    }
    h2,h3 {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)
    
    st.title(''' ## Welcome to the,\n
    ## MUSIC CLASSIFICATION SYSTEM! 🎵❤️ ''')

    st.write("This app helps users to identify the genre of any song quickly and accurately. It’s useful for music lovers to discover similar tracks, organize playlists better, and enhance their listening experience. For musicians and DJs, it aids in music analysis, curation, and mixing. Streaming platforms and developers also use genre classification to recommend personalized content and improve user engagement through smarter algorithms. ")
    

elif(app_mode=="Prediction"):
    st.header("Model Prediction")
    test_mp3 = st.file_uploader("Upload an audio file",type=["mp3"])

    if test_mp3 is not None:
        filepath = "Test_Music/"+test_mp3.name
        with open(filepath, "wb") as f:
            f.write(test_mp3.read())
    
    #Show button
    if(st.button("Play Audio")):
        st.audio(test_mp3)

    #Predict button
    if(st.button("Predict")):
        with st.spinner("Please wait..."):
            X_test = load_and_preprocess_data(filepath)
            result_index = model_prediction(X_test)
            label = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

            st.markdown("Model Prediction: It's a {} music".format(label[result_index]))



#to create a new virtual env.
#    python -m venv [name of env]

## to activate venv
#      [name of env]\Scripts\activate.bat  (for cmd use .bat)

# to install all libraries create a requirements.txt and then
# pip install -r requiremnets.txt