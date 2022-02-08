import streamlit as st
import pandas as pd
import numpy as np
import os
import string

from streamlit_drawable_canvas import st_canvas

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from sklearn.preprocessing import LabelEncoder

# lecture d'images
import PIL.ImageOps
from PIL import Image 
from tensorflow import squeeze

from generator_rnn import DatasetGenerator
import rnn_model_streamlit


# evaluation
from jiwer import wer 

title = "Modèle CRNN"
sidebar_name = "Modèle CRNN"


def run():

    st.markdown("<h1 style='text-align: center; color: black;'>Modèle CRNN </h1>",unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: center; color: black;'>Principe de fonctionnement du modèle </h2>",unsafe_allow_html=True)
    
    st.markdown(
        """
       Ce modèle a été choisi pour nous permettre de faire une reconnaissance du mot, caractère par caractère.  
       Le principe général de la reconnaissance de caractères passe par l'extraction de plusieurs caractéristiques uniques. Ainsi la comparaison des caractéristiques du texte manuscrit avec celles des caractères appris permet la reconnaissance.<p>Nous avons utilisé deux types de modèles: un modèle de type CNN et un modèle de type RNN. </p>""", unsafe_allow_html= True


    )
    st.markdown("<h3 style='text-align: center; color: black;'>Caractéristiques du modèle </h3>",unsafe_allow_html=True)
    st.image(Image.open("assets/CRNN_PenPyText.drawio.png"))


    st.markdown("""
    <ul> 
    <li>
    <b>Performance:</b>
    Le modèle CNN a <b>une accuracy </b>d'environ 53% dans le top 5 des prédictions et de 43% pour la première prédiction.
    <p>La <b>CER moyenne</b> sur la première prédiction est de 0.36, ce qui nous indique que plus de 60% des caractères individuels sont correctement prédits.</p>
    </li>
    </ul>
    """, unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: center; color: black;'>Simulation </h2>",unsafe_allow_html=True)
    
    # Charger une image ou créer un composant st_canvas = tableau blanc
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
    height=75,
    drawing_mode="freedraw",
    key="canvas",
    )

   # Conversion images pour preprocess
    def convert2png():
        if output_canvas.image_data is not None:
            img = output_canvas.image_data
            # lecture de l'image avec PIL
            im = Image.fromarray(img.astype(np.uint8), mode = "RGBA")
            im = im.convert('L')
            # Conversion en noir et blanc
            im = im.convert('1')
            im.save("img_to_transcript.png")
        if uploaded_file is not None:
            img = uploaded_file
            # lecture de l'image avec PIL
            im = Image.open(img)
            img_array = np.array(im)
            im.save("img_to_transcript.png")
    
    def preprocess_rnn():
        str1 = ["img_to_transcript.png"]
        str2 = ["success"]
        df = pd.DataFrame(list(zip(str1,str2)), columns = ["path", 'label'])
        return df
    
    if  'nb_pred' not in st.session_state:
        st.session_state['nb_pred'] = 5
    st.radio('Selectionner le nombre de prédictions:', [1, 5], key = 'nb_pred')
    
  

    # Chargement du modèle
    def load_model_rnn():
        model = rnn_model_streamlit.build_model_rnn((128,32),1)
        model.load_weights('rnn_streamlit_weights.h5')
        return model

    # Fonction de décodage des vecteurs en mots
    def transcript_vect_mot(vecteur):
        string_array = np.asarray(list(string.printable[:-17]))
        le = LabelEncoder()  
        le.fit(string_array)
        return le.inverse_transform(vecteur)

    if  'affichage' not in st.session_state:
        st.session_state['affichage'] = " "
    if 'pred' not in st.session_state:
        st.session_state.pred = "."

    # Prédictions
    def pred_top5(predictions):
        # Récupération du top 5 sous forme de vecteur
        pred_code = []
        for i in range(5):
            labels = K.get_value(K.ctc_decode(predictions, 
                                          input_length=np.ones(predictions.shape[0])*predictions.shape[1],
                                          greedy=False, beam_width = 10, top_paths = 5)[0][i])
            pred_code.append(labels)
        # Décoder vecteurs mots
        top_5 = []
        for pred in range(5):
            k = 0
            result = []
            while (pred_code[pred].item(k) != -1):
                result.append(pred_code[pred].item(k))
                k +=1
            mot = transcript_vect_mot(result)
            top_5.append(''.join(mot))
        st.session_state.pred = top_5[0]
        # Affichage
        if st.session_state.nb_pred == 1:
            st.session_state.affichage = '**Prediction**: {}'.format(top_5[0])
        else:
            st.session_state.affichage = " "
            for i in range(5):
                st.session_state.affichage = st.session_state.affichage + ('**Predictions** {}: {}  \n'.format(i+1,top_5[i]))
        return top_5

    if st.button('Cliquer pour transcrire'):
        img_ready = convert2png()
        df_rnn = preprocess_rnn()
        img_gen = DatasetGenerator(dataframe=df_rnn, 
                                  x_col = "path",
                                  y_col = "label", 
                                  targetSize = (128,32),
                                  batchSize = 1,
                                  shuffle = False, 
                                  max_y_length = 10)
        model_rnn =  load_model_rnn()
        predictions = model_rnn.predict(img_gen)
        preds = pred_top5(predictions)
        st.session_state.pred = preds[0]
        transcription = st.empty()
        transcription.success(st.session_state.affichage)
        img_rnn = None
    
    

    st.markdown("<h2 style='text-align: center; color: black;'>Évaluation </h2>",unsafe_allow_html=True)

    st.checkbox("Explication du CER", key= 'demo_cer')
    if st.session_state.demo_cer:
        st.markdown("""
            <p>
            Nous appliquons le calcul du Character Error Rate à la première prédiction de notre modèle pour évaluer 
            notre output d’OCR.<br>   
            On définit 3 types d’erreurs:  </p>  
            <ul> 
            <li><b>S  = Substitution error </b>: un caractère mal identifié  </li>
            <li><b>D = Deletion error </b>: un caractère manquant   </li> 
            <li><b>I = Insertion error </b>: inclusion d’un caractère   </li>
            </ul>
            """, unsafe_allow_html=True)
        
        st.image(Image.open("assets/cer_1.png"), caption = 'Les erreurs prise en compte dans le CER')
        
        col3, col4, col5 = st.columns(3)
        with col4:
            st.image(Image.open("assets/cer_2.png"), width = 320, caption='Calcul dérivé de la distance de Levenshtein')

        st.markdown(
            """
            Le CER montre le pourcentage de caractères mal prédits. Plus la valeur est faible, meilleures sont les 
            performances de notre modèle, un CER de 0 étant un score parfait.  
            Notre CER moyen sur le jeu de données test avec le modèle de transcription de maximum 10 caractères est actuellement à 0.38

            """
        )

    st.markdown("**Écrivez le mot réel pour calculer le CER**",unsafe_allow_html=True)
    st.text_input('', key='reality')
    
    st.button('Calculer CER', key='eval')
    if st.session_state.eval == True:
        st.write("Réalité: ", st.session_state.reality)
        st.write("Prédiction: ", st.session_state.pred)
        cer =  wer(list(st.session_state.reality), list(st.session_state.pred))
        st.write("**Le taux d'erreur par caractère (CER) est de** ", round(cer,2))
    
    st.session_state.affichage = " "

