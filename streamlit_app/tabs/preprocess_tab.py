import streamlit as st
from PIL import Image

title = "Prétraitement des données images"
sidebar_name = "Préprocessing"
#st.header('My header')
#st.subheader('My sub')


def run():

    st.markdown("<h1 style='text-align: center; color: black;'>Prétraitement des données images</h1>", 
    unsafe_allow_html=True)

    st.markdown("---")


    st.markdown(
        """
       Une première analyse des images nous a permis de constater que la qualité et la taille de celles-ci n'étaient pas égales. 
Certaines images sont une composition de fragments de mots, les outils scripteurs utilisés sont variés. Afin de pouvoir identifier les caractères et les mots, il est nécessaire d'effectuer un pré-traitement des images.
Pour cette étape, nous avons réalisé les opérations suivantes:
        """)
    st.markdown("""
    <ul> 
    <li><b>Retirer du jeu de données les images endommagées:</b>
    Deux images du dataset sont endommagées et retirées (a01-117-05-02.png et r06-022-03-05.png)
    </li>
    <li><b>Nettoyage du bruit:</b>
    Pour éliminer le bruit nous utilisons un filtre Gaussien qui permet une meilleure préservation des bords.
    </li>
    <li><b>Harmoniser les tailles des images:</b>
    Nous avons choisi de redimensionner les images en 128x32 pixels, car cette dimension permet de limiter la déformation des écritures et d'avoir une meilleure utilisation de la capacité mémoire de nos ordinateurs. 
    </li>
    <li><b>Érosion:</b>
   Cette étape d'érosion permet  d'uniformiser cette largeur de trait (plus mince) et de conserver uniquement le squelette des lettres manuscrites.
    </li>
    <li><b>Images en noir et blanc:</b>
    Cette étape de binairisation permet d'améliorer la qualité de l'image. On utilise le niveau de gris seuil fournit dans nos données.
    """,unsafe_allow_html=True)

    col1, col2, col3, col4, col5= st.columns(5)
    with col1:
        st.image(Image.open("assets/exemple_img.png"))       
    with col2:
        st.image(Image.open("assets/img_gauss.png"))   
    with col3:
        st.image(Image.open("assets/img_resize.png"))
    with col4:
        st.image(Image.open("assets/img_erode.png"))      
    with col5:
        st.image(Image.open("assets/img_clean.png"))
        

    col6, col7, col8, col9, col10= st.columns(5)
    with col6:
        st.markdown("Image d'origine")
    with col7:
        st.markdown("Avec filtre Gaussien")   
    with col8:
        st.markdown("Redimensionnement (128,32)")
    with col9:
        st.markdown("Erosion")
    with col10:
        st.markdown("Nettoyage du bruit")


    st.markdown(
        """
        </li>
        <li><b>Retirer du trainset les images avec erreur de segmentation:</b>
     Nous avons fait le choix de retirer du jeu de données d'entraînement ces données.
        """, unsafe_allow_html=True
    )

    col11, col12, col13 = st.columns(3)
    with col12:
        st.image(Image.open("assets/US.png"), caption='label: US', width = 150)
