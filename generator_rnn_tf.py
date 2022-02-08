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
        X = tf.Variable(tf.zeros((self.batchSize, height * width), dtype=tf.float32))
        y = tf.Variable(tf.experimental.numpy.full((self.batchSize, self.max_length), fill_value = 321))
        y_lengths = tf.Variable(tf.stack([]))
        
        # Création du labelEncoder à partir de la liste de string
        string_array = np.asarray(list(string.printable[:-17]))
        le = LabelEncoder()  
        le.fit(string_array)
        
        # Génération des données X
        for i, ID in enumerate(liste_IDs_temp):
            path = self.directory + ID
            im = tf.io.read_file(path)
            img = tf.image.decode_png(im, channels = 1)
            img = tf.cast(img, dtype = tf.float16)
            img_norm = tf.math.divide(img, tf.constant(255.0, dtype = tf.float16))
            img_resize = tf.image.resize(img_norm, self.targetSize, method='area',
                                         preserve_aspect_ratio=False)
            #img_erod = tf.nn.erosion2d(value = img_resize, filters = tf.keras.backend.random_normal(shape = (3, 3, 1)), 
             #                          strides=[1, 1, 1, 1], padding = 'SAME', 
             #                         data_format = "NHWC", dilations = [1, 1, 1, 1])
            vect_img = tf.reshape(img_resize, (height * width))
            X[i,:].assign(vect_img)
        X = tf.reshape(X,[-1,  width, height, self.canaux])
        print(X)
        #X = np.array(X)
            
        # Génération des données y et y_lengths
        if (self.max_length == 10):
            FILL_VALUE = 14
        else:
            FILL_VALUE = 23
        for k,mot in enumerate(liste_labels_temp):
            j = len(mot)
            tf.stack([y_lengths, j])
            y[k,:j] = y[k,:j].assign(tf.constant(le.transform(list(mot))).numpy())
        #y_lengths = tf.constant(y_lengths)
        #y = tf.constant(y)
        input_length = tf.experimental.numpy.full((self.batchSize, 1), fill_value = 14)# fill_value = nb de features 


        return [X, y, y_lengths]
    

def transcript_vect_mot(vecteur):
    string_array = np.asarray(list(string.printable[:-17]))
    le = LabelEncoder()  
    le.fit(string_array)
    return le.inverse_transform(vecteur)

"""
Erreur non résolue
NotFoundError: No registered 'ResourceStridedSliceAssign' OpKernel for 'GPU' devices compatible with node {{node ResourceStridedSliceAssign}}
(OpKernel was found, but attributes didn't match) Requested Attributes: Index=DT_INT32, T=DT_FLOAT, begin_mask=2, ellipsis_mask=0, end_mask=2, new_axis_mask=0, shrink_axis_mask=1
Registered:  device='DEFAULT'; T in [DT_INT32]
  device='CPU'; T in [DT_UINT64]
  device='CPU'; T in [DT_INT64]
  device='CPU'; T in [DT_UINT32]
  device='CPU'; T in [DT_UINT16]
  device='CPU'; T in [DT_INT16]
  device='CPU'; T in [DT_UINT8]
  device='CPU'; T in [DT_INT8]
  device='CPU'; T in [DT_INT32]
  device='CPU'; T in [DT_HALF]
  device='CPU'; T in [DT_BFLOAT16]
  device='CPU'; T in [DT_FLOAT]
  device='CPU'; T in [DT_DOUBLE]
  device='CPU'; T in [DT_COMPLEX64]
  device='CPU'; T in [DT_COMPLEX128]
  device='CPU'; T in [DT_BOOL]
  device='CPU'; T in [DT_STRING]
  device='CPU'; T in [DT_RESOURCE]
  device='CPU'; T in [DT_VARIANT]
 [Op:ResourceStridedSliceAssign] name: strided_slice/_assign

"""