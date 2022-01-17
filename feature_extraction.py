import os
import pickle


# Add all images path to filename.
actors = os.listdir('images')

filenames = []
for actor in actors:
    for file in os.listdir(os.path.join('images',actor)):
        filenames.append(os.path.join('images',actor,file))
pickle.dump(filenames,open('filenames.pkl','wb'))

# len(filenames) = 8664


from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np


# Model Building
model = VGGFace(model='resnet50',include_top=False,input_shape=(244,244,3),pooling='avg')

# Create Function to extract feature of Given Image
def feature_extraction(image_path,model):
    img = image.load_img(image_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    # Keras works with batches of images. So, the first dimension is used for the number of samples we have.
    # In this case 1
    expanded_img = np.expand_dims(img_array,axis=0)
    # normalizing the color channels on which vgg was trained
    preprocessed_img = preprocess_input(expanded_img)

    result = model.predict(preprocessed_img).flatten()

    return result

# Creating list to store all the features of images
features = []
for image_file in filenames:
    features.append(feature_extraction(image_file,model))
print(features)

pickle.dump(features,open('features.pkl','wb'))





