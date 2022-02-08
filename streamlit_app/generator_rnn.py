"""
Created on Fri Oct 15 2021

@author: samy_ait-ameur, sophie_amedro, stephane_tchatat
"""

import numpy as np

import tensorflow as tf
import tensorflow.keras

import string
import cv2
from deskew import determine_skew
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.layers import Input


# Réalisation d'un générateur de données pour le modèle RNN
 
# Cette classe Datasetgénérator doit hériter de la classe Keras.utils.Sequence afin de garantir de parcourir une seule et unique fois nos données au cours d’une époch.

# Structure inspirée de  
# https://deeplylearning.fr/cours-pratiques-deep-learning/realiser-son-propre-generateur-de-donnees/  
# https://stanford.edu/~shervine/l/fr/blog/keras-comment-generer-des-donnees-a-la-volee


class DatasetGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self, dataframe, x_col, y_col, targetSize, directory = "", nb_canaux = 1, batchSize = 32 ,
                 shuffle = True, max_y_length = 21):
        """ fonction d'initiation de notre générateur de données pour Keras
        dataframe: dataframe de nos données
        directory: complément du path vers les images
        x_col: colonne du df contenant données X
        y_col: colonne du df contenant nos données y
        nb_canaux: 1 ou 3 selon la couleur de l'image
        batchSize: taille d'un mini lot de données
        shuffle: booléen pour données aléatoires ou non
        targetSize: pour fair un resize de nos images
        max_y_length: taille maximale des mots du dataset"""
        # initialisations
        self.directory = directory
        self.xData = dataframe[x_col] 
        self.yData = dataframe[y_col]
        self.canaux = nb_canaux
        self.batchSize = batchSize
        self.shuffle = shuffle
        self.targetSize = targetSize
        self.max_length = max_y_length
        self.n = len(dataframe)
        self.on_epoch_end()

        
    def __len__(self):
        """Fonction qui définit le nombre de lots durant une epoch"""
        return int(np.floor(len(self.xData) / self.batchSize))
        
    
    def on_epoch_end(self):
        """Fonction appelée à chaque fin d'epoch pour mise à jour des indices après chaque epoch"""
        # Shuffle = false, les indexs sont dans l'ordre
        self.indexes = np.arange(len(self.xData))
        
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
            
    
    def __getitem__(self, index): 
        """Fonction qui génère des données transformées en lots"""
        # générer des indices correspondant au lot
        currentBatch = self.indexes[index * self.batchSize:(index + 1) * self.batchSize]
        # Etablir la liste des X 
        liste_IDs_temp = [self.xData[i] for i in currentBatch]
        # Etablir la liste des y
        liste_labels_temp = [self.yData[i] for i in currentBatch]
        
        # Initialisation
        width, height = self.targetSize
        X = np.zeros((self.batchSize, height * width), dtype=np.float32)
        y = np.full((self.batchSize, self.max_length), fill_value = 321)
        y_lengths = []
        
        # Création du labelEncoder à partir de la liste de string
        string_array = np.asarray(list(string.printable[:-17]))
        le = LabelEncoder()  
        le.fit(string_array)
        
        # Génération des données X
        for i, ID in enumerate(liste_IDs_temp):
            path = self.directory + ID
            im = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
            # Nettoyage du bruit Gaussien
            img = cv2.GaussianBlur(im, ksize=(3, 3), sigmaX=0)
            # redimensionner
            img = cv2.resize(img, self.targetSize, interpolation=cv2.INTER_AREA)
            # erosion
            img = cv2.erode(img,np.ones((1,1), np.uint8),iterations = 1) 
            # Nettoyage du bruit en utilisant le niveau de gris (grayscale)
            _ , image_clean = cv2.threshold(img, 200.0, 255, type = cv2.THRESH_BINARY)
            # Normalisation
            image = (image_clean/255)
            # Mise en forme en array
            vect_img = np.reshape(image, (height * width))
            X[i,:] = vect_img
        X = X.reshape([-1,  width, height, self.canaux])
        X = np.array(X)
            
        # Génération des données y et y_lengths
        if (self.max_length == 10):
            FILL_VALUE = 14
        else:
            FILL_VALUE = 30
        for k,mot in enumerate(liste_labels_temp):
            j = len(mot)
            y_lengths.append([j])
            y[k,:j] = le.transform(list(mot))
        #y_lengths = np.array(y_lengths)
        y_lengths = tf.convert_to_tensor(np.array(y_lengths), dtype=tf.int32)
        # y = np.array(y)
        y = tf.convert_to_tensor(np.array(y), dtype=tf.int32)
        #input_length = np.full((self.batchSize, 1), fill_value = FILL_VALUE)# fill_value = nb de features 
        input_length = tf.experimental.numpy.full((self.batchSize, 1), fill_value = FILL_VALUE)
        return [X, y, y_lengths]
    

def transcript_vect_mot(vecteur):
    string_array = np.asarray(list(string.printable[:-17]))
    le = LabelEncoder()  
    le.fit(string_array)
    return le.inverse_transform(vecteur)




