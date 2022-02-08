import streamlit as st
from PIL import Image

title = "Bilan"
sidebar_name = "Bilan"



def run():

    
    st.title(title)

    st.markdown("---")


    st.markdown(
        """
        Ce projet a été un véritable challenge pour tous les membres de l'équipe. Il a été très instructif, aussi bien en termes de programmation, qu'en termes de préparation des données d'imagerie. 
        Nous avons acquis des bases solides sur le fonctionnement des réseaux de neurones et les différentes couches qui le composent, ainsi que la fonction de coût et les données temporelles.
        Le projet PenPyText portait sur la reconnaissance de caractères manuscrits. Nous sommes plutôt satisfaits de nos résultats, car nous avons réussi à implémenter un CRNN et à l'entraîner. 
        Nous avons programmé notre propre générateur de données pour le CRNN. Enfin, notre accuracy nous semble correcte pour un premier projet d'OCR car elle dépasse 50% dans les 5 premières prédictions, 
        sur le jeu de données test.
        """)
    

    st.subheader('Quelques idées pour aller plus loin ')
    st.markdown(
        """
        <ul> 
        Si on avait eu plus de temps pour poursuivre ce projet,  qu'est-ce qu'on aurait pu faire?
        <li>
        Utiliser le CER comme callback pour surveiller les distances réalité/prédict
        </li>
        <li>
        Augmenter les données avec des mots plus longs
        </li>
        <li>
        Il en découle que l’on pourrait alors augmenter le nombre de caractères maximum à transcrire
        </li>
        <li>
        On a tenté d’ajouter à notre modèle une validation par lexique à notre modèle CRNN (mais couteux en temps – utilisation des mots anglais du dictionnaire NLTK) 
        </li>
        <li>
        La prochaine grande étape aurait été de découper l’image par caractère pour entraîner un modèle sur des images de caractères individuels
        </li>
        <li>
        Enfin utiliser le transfer learning (type Lenet pour caractères individuels, ou entraîné sur des modèles de transcription de captcha)
        </li>
        """
        ,unsafe_allow_html=True)
    
