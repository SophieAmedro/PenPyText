import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

from streamlit_drawable_canvas import st_canvas

from tensorflow import keras
import tensorflow as tf

# lecture d'images
import PIL.ImageOps
from PIL import Image 
import cv2

# Encodage
from tensorflow.keras.preprocessing.text import Tokenizer

title = "Modèle CNN"
sidebar_name = "Modèle CNN"


def run():

    st.markdown("<h1 style='text-align: center; color: black;'>Modèle CNN </h1>",unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: center; color: black;'>Principe de fonctionnement du modèle </h2>",unsafe_allow_html=True)

    st.markdown(
        """
       Pour ce modèle, nous avons abordé notre projet comme une problématique de classification.  
       Chaque mot distinct de notre jeu d'entraînement correspondait à une classe.   
       Nous avions ainsi grâce à notre jeu d'entraînement un dictionnaire de 8004 mots.   
       Un modèle de réseau de neurones convolutifs suffisait à extraire les caractéristiques des images et à en obtenir une classification.
        """
    )
    st.image(Image.open("assets/CNN_PenPyText.drawio.png"))
    st.markdown("<h3 style='text-align: center; color: black;'>Caractéristiques du modèle </h3>",unsafe_allow_html=True)
    st.markdown("""
    <ul> 
    <li>
    <b>Nombre de classes:</b>
    8004
    </li>
    <li>
    <b>Performance:</b>
    Le modèle CNN a une accuracy d'environ 60% dans le top 5 des prédictions et de 22% pour la première prédiction.
    </li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center; color: black;'>Simulation </h2>",unsafe_allow_html=True)
    
    # Charger un image ou créer un composant st_canvas = tableau blanc
    st.markdown("<b>Charger un fichier</b> ",unsafe_allow_html=True )
    uploaded_file = st.file_uploader("", type = ['png', 'jpg'])
   
    st.markdown("<b>OU écrivez un mot en anglais sur le tableau blanc </b> ",unsafe_allow_html=True )
    
    output_canvas = st_canvas(
    fill_color="black",  # couleur fixe grise
    stroke_width= 3, #largeur du pinceau
    stroke_color="black", 
    background_color="white",
    background_image= Image.open(uploaded_file) if uploaded_file else None,
    update_streamlit= True, #realtime_update, # TRUE?
    width = 300,
    height = 75,
    drawing_mode="freedraw",
    key="canvas",
    )
    
    # Conversion images pour preprocess
    def convert2png():
        if output_canvas.image_data is not None:
            img = output_canvas.image_data
            # lecture de l'image avec PIL
            im = Image.fromarray(img.astype(np.uint8), mode = "RGBA")
            #im.resize((100,400))
            im = im.convert('L')
            # Conversion en noir et blanc
            im = im.convert('1')
            st.session_state.img_input = im
            im.save("img_to_transcript.png")
        if uploaded_file is not None:
            img = uploaded_file
            # lecture de l'image avec PIL
            im = Image.open(img)
            img_array = np.array(im)
            st.session_state.img_input = img_array
            im.save("img_to_transcript.png")
    
    

    # Préparation de l'image
    def preprocess_cnn():
        # lecture de l'image
        img = cv2.imread("img_to_transcript.png", cv2.IMREAD_GRAYSCALE)
        # Rogner image
        #contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        #x,y,w,h = cv2.boundingRect(img)
        #st.write(x,y,w,h)
        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        #st.image(img)
        # Nettoyage du bruit Gaussien
        img = cv2.GaussianBlur(img, ksize=(3, 3), sigmaX=0)
        # redimensionner
        img = cv2.resize(img, (128,32), interpolation=cv2.INTER_AREA)
        # erosion
        img = cv2.erode(img,np.ones((1,1), np.uint8),iterations = 1)
        #st.image(img)
        # Nettoyage du bruit en utilisant le niveau de gris (grayscale)
        _ , image_clean = cv2.threshold(img, 200, 255, type = cv2.THRESH_BINARY)
        img_clean =  image_clean
        #st.image(image_clean)
        # Normalisation
        image = (img_clean/255)
        #st.session.img_ready = image
        # Mise en forme en array
        X = np.asarray(image)
        X = X.reshape([-1,32,128,1])
        return X
    
   
    if  'nb_pred' not in st.session_state:
        st.session_state['nb_pred'] = 5
    st.radio('Selectionner le nombre de prédictions:', [1, 5], key = 'nb_pred')
    
    
    # Chargement du modèle
    def load_model():
        model = keras.models.load_model('cnn_model')
        return model
    
    
    if  'affichage' not in st.session_state:
        st.session_state['affichage'] = " "

    # Prédictions
    def pred_top5(predictions):
        # indices/classes des 5 plus grandes probas dans l'ordre croissant des probas
        predicted_class_indices=np.argpartition(predictions,-5,axis=1)[:,-5:] 
        # Décodeur
        words_pred = np.copy(predicted_class_indices).astype('str')
        vocab = pickle.load(open("../vocab_cnn.pkl", 'rb'))
        for i in range(len(words_pred)): # parcourt de chaque ligne
            for k in range(5):# décoder chaque nombre en mot 
                idx = predicted_class_indices[i,k]
                words_pred[i,k] = list(vocab.keys())[list(vocab.values()).index(idx)]
        # Affichage
        if st.session_state.nb_pred == 1:
            st.session_state.affichage = '**Prediction**: {}'.format(words_pred.item(4))
        else:
            st.session_state.affichage = " "
            n = 0
            for i in range(4, -1, -1):
                n += 1
                st.session_state.affichage = st.session_state.affichage + ('**Predictions** {}: {}  \n'.format(n, words_pred.item(i)))
        return st.session_state.affichage
    
    
    if st.button('Cliquer pour transcrire'):
        img_cnn = convert2png()
        img_cnn = preprocess_cnn()
        model_cnn = load_model()
        predictions = model_cnn.predict(img_cnn)
        pred = pred_top5(predictions)
        transcription = st.empty()
        transcription.success(pred)
        img_cnn = None


