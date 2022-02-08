"""
Created on Wed Jun 14 2021

@author: samy_ait-ameur, sophie_amedro, stephane_tchatat
"""

# Importations
import pandas as pd
import os
import nettoyage_fichiers

def path(id_image):
    """
    Fonction qui complète l'ID de x par le chemin vers l'image x dans le dossier data
    Paramètre
        x : ID de l'image (string)
    Return
        path_image: chemin d'accès à l'image (string)
    """
    path_image = "data/words/" + id_image.split('-')[0] + "/" + id_image.rsplit('-', 2)[0] + "/" + id_image + ".png"
    return path_image

def txt_to_df(filename = 'words.txt'): 
    """
    Cette fonction lit le fichier d'informations words.txt, conserve les informations utiles pour les modèles.
    Les informations inutiles pour notre projet sont: la nature grammaticale de la transcription et les coordonnées des images.
    Cette fonction crée un dataframe et un csv 
    Paramètre:
        filename: nom du fichier .txt, ici words.txt (string)
    Return:
        data_words: dataframe contenant les ID images 'word_id', si la segmentation de l'image est bonne par rapport à la 
        transcription 'ok_err', le niveau de gris de l'image 'gray_level', la transcription 'transcript', le chemin 
        d'accès de l'image dans le dossier data 'data_path'
    """
    
    # Téléchargement fichier à partir d'une source texte
    words = open("data/ascii/" + filename, "r")
    
    # Création d'un dictionnaire
    d = {}
    i = 0

    # Récupération des informations words.txt dans un csv
    # Parcours de chaque ligne de words.txt
    for line in words:
        # On laisse la partie explications
        if line.startswith("#"):
            continue
        else:
            # Séparation de la ligne à chaque espace pour récupérer chaque élément
            line_element = line.split()
            d[i] = {'word_id':line_element[0], # ID image
                    'ok_err':line_element[1], # segmentation de l'image ok ou non
                    'gray_level': line_element[2], # niveau de gris
                    'transcript': line_element[-1]} # transcription
            i = i + 1
                    
    # Création du dataframe
    data_words = pd.DataFrame.from_dict(d, "index")      

    """Nous avons besoin de l'info de word_id pour récupérer l'information du lines_id. Le line_id est nécessaire pour
       le split dans traintestsplit. C'est pour cela qu'il ne faut pas renommer word_id en data_path, mais créer une nouvelle 
       colonne data_path
    """


    # Appliquer la fonction path à la colonne word_id pour déterminer la colonne data_path
    data_words['data_path'] = data_words['word_id'].apply(lambda x: path(x))
    
    return data_words
    
    