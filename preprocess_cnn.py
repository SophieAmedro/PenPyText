"""
Created on Tue Dec 02 2021

@author: samy_ait-ameur, sophie_amedro, stephane_tchatat
"""

# Importations
import numpy as np
import harmonisation as h

# lecture d'images
import cv2

# Encodage
from tensorflow.keras.preprocessing.text import Tokenizer

# Dataset
import tensorflow as tf

# Fonctions de preprocessing des images et de création des features X 

def preprocess_images(image_path, gray_level, target_size):
    """
    La fonction preprocess_images prend en argument l'adresse d'une image, le niveau de gris et les dimensions à prendre.
    Elle utilise la fonction harmony_clean pour harmoniser et redimensionner une image et nettoyer le bruit.
    Paramètres :
        image_path: str correspondant à l'adresse de l'image dans le fichier data
        grayscale: int de l'intervalle [0, 255] qui correspond au niveau de gris de l'écriture
        target_size : tuple correspondant aux dimensions de l'image souhaitées
    Renvoie:
        image: array de l'image ayant les dimensions target_size
    """
    #  Nettoyage du bruit et harmonisation de la taille des images
    image = h.harmony_clean(image_path, gray_level, target_size[0], target_size[1])
    # Normalisation
    image = (image/255)
    return image


def features(data, target_size):
    """
    La fonction features prend en argument un dataframe et les dimensions souhaitées.
    Elle utilise la fonction preprocess_images pour harmoniser et redimensionner une image et nettoyer le bruit et renvoie
    un numpy array de toutes les images fournie dans le dataframe prêt à être utilisé par un modèle
    Paramètres :
        data: dataframe contenant les informations image_path, gray_level
        target_size : tuple correspondant aux dimensions de l'image souhaitées
    Renvoie:
        image: numpy array des images ayant les nb (d'images, target_size, 1) , 1 car images en niveaux de gris
    """
    X = []
    for i in range(data.shape[0]):
        X.append(preprocess_images(data.data_path[i], data.gray_level[i], target_size))
    X = np.array(X)
    X = X.reshape([-1,target_size[1],target_size[0],1])
    return X

# Fonctions de preprocessing des labels

# Création d'un dictionnaire de codage des mots en nombres
def labels_dict(data):
    """
    La fonction labels_dict crée un dictionnaire dont les clés sont les labels et les valeurs sont un codage en nombre de ces 
    en utilisant Tokenizer.
    Paramètres :
        data: dataframe contenant les labels à coder
    Renvoie:
        labels: dictionnaire mot/code
    """
    y_labels = data.transcript
    tokenizer = Tokenizer(filters = '', lower = False)# pas de conversion en minuscules
    tokenizer.fit_on_texts(y_labels)
    labels = tokenizer.word_index
    return labels

def encoder_labels(data_dict, data):
    """
    La fonction coder_labels encode les labels/transcriptions en utilisant le dictionnaire labels_dict
    Paramètres :
        data_dict: dataframe contenant les mots permettant de créer le dictionnaire d'encodage 
        data: dataframe contenant les labels à encoder
    Renvoie:
        y: numpy array contenant les codes (des int)
    """
    # créer le dictionnaire d'encodage
    labels = labels_dict(data_dict)
    y = []
    for i in range(data.shape[0]):
        if data.transcript[i] in labels.keys():
            y.append(labels.get(data.transcript[i]))
        else:
            y.append(-1)
    y = np.array(y)
    return y

# Créer un dataset tensorflow avec batchs pour modèle cnn
def data_Xy(data, target_size, data_dict, batch_size = 100):
    """
    La fonction data_Xy crée un Dataset 
    Paramètres :
        data: dataframe contenant les données à préparer pour le modèle
        target_size : tuple correspondant aux dimensions de l'image souhaitées
        batch_size: int correspondant à la taille des batchs
        data_dict: dataframe contenant les mots permettant de créer le dictionnaire d'encodage 
    Renvoie:
        dataset: tf.data.Dataset avec les features X et les les targets y
    """
    # features
    X = features(data, target_size)
    # labels
    y = encoder_labels(data_dict, data)
    # Définir un objet dataset à partir du tuple (X, y)
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    # Séparer le jeu de données en batchs
    dataset = dataset.batch(batch_size)
    return dataset