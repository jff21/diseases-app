import streamlit as st
import tensorflow  as tf
import numpy as np
from tensorflow.keras.utils import load_img,img_to_array
from tensorflow.keras.preprocessing import image
from PIL import Image,ImageOps

st.title("Image Classification")

upload_filer=st.sidebar.file_uploader("Télécharger un fichier:", type=['jpg','jpeg','png'])

generate_prediction=st.sidebar.button("predict")

model=tf.keras.models.load_model("model.h5")
# clé-valeur = la clé etant COVID19 et la valeur 0 par exemple
covid_classes= {'COVID19':0,'NORMAL':1,'PNEUMONIA':2,'TUBERCULOSIS':3}

if upload_filer:
    st.image(upload_filer,caption="Image téléchargé",use_column_width=True)
    #conversion de l'image en array
    test_image=image.load_img(upload_filer,target_size=(64,64))
    image_array=img_to_array(test_image)
    # on augmente la dimension de l'image
    image_array=np.expand_dims(image_array,axis=0)

    #Faisons passer l'image loader sur le model
    if generate_prediction:
        predictions=model.predict(image_array)
        #on prend la classe max de la prediction du modele
        classes=np.argmax(predictions[0])
        for key,value in covid_classes.items():
            if value==classes:
                st.write("the diagnostic is:",key)
