import streamlit as st
from PIL import Image

title = "What is PenPyText project?"
sidebar_name = "Présentation du projet"



def run():

    st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/2.gif")

    st.title(title)

    st.markdown("---")


    st.markdown(
        """
        Le projet PenPyText pose la problématique complète et ambitieuse de la reconnaissance de textes manuscrits. 
        Au cœur des objectifs de digitalisation des entreprises, la capacité à convertir une image numérisée d'un texte manuscrit en un document textuel exploitable par une application informatique est une brique essentielle. 
        Il devient alors possible de digitaliser un plus grand nombre de documents, et d'automatiser de nouveaux processus.
        """)
    st.subheader('Exemple : ')
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image(Image.open("assets/exemple_img.png"))

    with col4:
        st.text('exactly')

    st.subheader('Jeu de données: ')
    st.markdown(
        """
        Pour réaliser ce projet, nous avons utilisé le jeu de données IAM Handwriting Database 3.0 disponible  à l’adresse suivante: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database.
        
        La base de données contient des formes de texte anglais de natures variées, écrites à la main sans contrainte, qui ont été scannées à une résolution de 300 dpi et enregistrées sous forme d'images PNG avec 256 niveaux de gris.
         """)
    st.markdown(
        """
    
        Explication des deux types de transcription:
        - par mots entiers à l'aide d'un ensemble de vocabulaire
        - par lettres
        
       
        """
    )
