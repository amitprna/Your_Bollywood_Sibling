import os
import numpy as np
import streamlit as st
import pickle
import requests

from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace

from sklearn.metrics.pairwise import cosine_similarity

from PIL import Image
import cv2
from mtcnn import MTCNN
from streamlit_lottie import st_lottie

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

url = "https://assets7.lottiefiles.com/private_files/lf30_y5tq70sy.json"
res_json = load_lottieurl(url)
st_lottie(res_json)

st.title('Your Sibling From Bollywood')


detector = MTCNN()
model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
feature_list = pickle.load(open('features.pkl','rb'))
filenames = pickle.load(open('filenames.pkl','rb'))

def save_uploaded_image(uploaded_image):
    try:
        with open(os.path.join('sample',uploaded_image.name),'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False

def extract_features(img_path,model,detector):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)

    x, y, width, height = results[0]['box']

    face = img[y:y + height, x:x + width]

    #  extract its features
    image = Image.fromarray(face)
    image = image.resize((224, 224))

    face_array = np.asarray(image)

    face_array = face_array.astype('float32')

    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result

def recommend(feature_list,features):
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos



uploaded_image = st.file_uploader('Upload Your Image')

if uploaded_image is not None:
    # save the image in a directory
    if save_uploaded_image(uploaded_image):
        # load the image
        display_image = Image.open(uploaded_image)

        # extract the features
        features = extract_features(os.path.join('sample',uploaded_image.name),model,detector)
        # recommend
        index_pos = recommend(feature_list,features)
        predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))
        # display
        col1,col2 = st.columns(2)
        with col1:
            st.header('Your Image')
            st.image(display_image)
        with col2:
            lis = filenames[index_pos].split('\\')
            lis1 = "/".join(lis[:2])
            out_img = lis1+'/image.jpg'
            st.header("Your Sibling \n" + predicted_actor)
            st.image(out_img,width=300)
