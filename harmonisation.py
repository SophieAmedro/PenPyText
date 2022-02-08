"""
Created on Tue Jul 13 2021

@author: samy_ait-ameur, sophie_amedro, stephane_tchatat
"""

# Importations
import numpy as np

# lecture d'images
import cv2



def harmony_clean(image_path, grayscale, new_width, new_height):
    """
    La fonction harmony_clean prend en argument l'adresse d'une image, le niveau de gris et les dimensions à prendre.
    Elle utilise les fonctions clean_bruit et size_harmonisation pour harmoniser les tailles d'une image et nettoyer 
    le bruit.
    Paramètres :
        image: str correspondant à l'adresse de l'image dans le fichier data
        max_width : int correspondant à la largeur de l'image souhaitée
        max_height : int correspondant à la hauteur de l'image souhaitée
        grayscale: int de l'intervalle [0, 255] qui correspond au niveau de gris de l'écriture
    Renvoie:
        array de l'image ayant les mêmes dimensions que celles données en paramètres
    """
    # lecture de l'image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Nettoyage du bruit Gaussien
    img = cv2.GaussianBlur(img, ksize=(3, 3), sigmaX=0)
    # redimensionner
    img = cv2.resize(img, (new_width,new_height), interpolation=cv2.INTER_AREA)
    # erosion
    img = cv2.erode(img,np.ones((1,1), np.uint8),iterations = 1) 
    # Nettoyage du bruit en utilisant le niveau de gris (grayscale)
    _ , image_clean = cv2.threshold(img, float(grayscale), 255, type = cv2.THRESH_BINARY)
    return image_clean