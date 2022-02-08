"""
Created on Tue Jul 13 2021

@author: samy_ait-ameur, sophie_amedro, stephane_tchatat
"""

# Importations

# lecture d'images
import cv2
from PIL import Image
# Visualisation
import matplotlib.pyplot as plt


# Recherche d'images avec des erreurs de lecture ou d'affichage
def error_image(images_list):
    """
    La fonction error_image vérifie qu'il n'y a pas d'erreur de lecture des images d'un dossier
    Paramètre : 
        images_list: liste d'adresse d'image 
    Renvoie:
        affichage des images comportant des erreurs
        errors : liste des path_images avec erreurs de lecture
    """
    errors = []
    for image in images_list:
        try:
            im=Image.open(image)
        except IOError:
            print("Erreur de lecture sur l'image:", image.split('/')[-1])
            errors.append(image)
    return errors

        
# Nettoyage des erreurs de lecture d'image
def clean_data(data):
    """
    La fonction clean_data enlève nettoie le dataframe data des données en enlevant les lignes des 
    images qui ne peuvent pas être utilisées
    Paramètre : 
        data: df comprenant les colonnes data_path, grayscale, transcript, word_id, 'ok_err'
    Renvoie:
        data: data sans les fichiers images avec erreurs
    """
    images_list = data.data_path
    errors = error_image(images_list)
    if (errors != []):
        error_lines = data[data['data_path'].isin(errors)]
        data = data.drop(error_lines.index.values, axis = 0, inplace = False)
    return data

