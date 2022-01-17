from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import cv2

from mtcnn import MTCNN # for face detection
from PIL import Image

feature_list = np.array(pickle.load(open('features.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = VGGFace(model='resnet50',include_top=False,input_shape=(244,244,3),pooling='avg')

# face detection
detector = MTCNN()
sample_img = cv2.imread('sample/05-1-4-784x441.jpg')
results = detector.detect_faces(sample_img)

x,y,width,height = results[0]['box']
face = sample_img[y:y+height,x:x+width]

# extract its features
# PIL  Image library to edit and manipulate images.
image = Image.fromarray(face)
image = image.resize((224,224))

face_array = np.asarray(image)
face_array = face_array.astype('float32')
expanded_img = np.expand_dims(face_array,axis=0)
preprocess_img = preprocess_input(expanded_img)
result = model.predict(preprocess_img).flatten()

# find the cosine similarity  between  current image(result) with all the 8655 features
similarity = []
for i in range(len(feature_list)):
    similarity.append(cosine_similarity(result.reshape(1,-1),feature_list[i].reshape(1,-1))[0][0])

# recommend the image
index_pos = sorted(list(enumerate(similarity)),reverse=True,key=lambda x:x[1])[0][0]

temp_img = cv2.imread(filenames[index_pos])
cv2.imshow('output',temp_img)
cv2.waitKey(0)

