import matplotlib.pyplot as plt
from keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

class_names=['NO DR','DR']
# Load the saved model
model = load_model("binary_dr_wgt_model.hdf5")

# Load the image
img_path = r'C:\Users\Mobile Programming\Desktop\000c1434d8d7.png'
img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x / 255.0  # Normalize the image data

# Make prediction
prediction = model.predict(x)
predicted_class = class_names[int(round(prediction[0][0]))]  # Round the probability to 0 or 1

# Display the image
plt.figure(figsize=(8, 4))
plt.imshow(img)
plt.axis('off')  # Remove axis labels
plt.title(f'Predicted Class: {predicted_class}')
plt.show()

# Print the predicted class
print("Predicted Class:", predicted_class)
'''
import streamlit as st
import tensorflow as tf
import streamlit as st



@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model("DR_twoclass.hdf5")
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Flower Classification
         """
         )

file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png"])
import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):

        size = (299,299)
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.

        img_reshape = img[np.newaxis,...]

        prediction = model.predict(img_reshape)

        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])
    st.write(prediction)
    st.write(score)
    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
'''