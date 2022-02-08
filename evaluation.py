"""
Created on Tue Dec 02 2021

@author: samy_ait-ameur, sophie_amedro, stephane_tchatat
"""
from random import sample
import cv2
import matplotlib.pyplot as plt

from jiwer import wer 

def transcript_in_top5(df, i):
    """
    La fonction transcript_in_top5 vérifie si la transcription réelle est présente dans les 5 premières prédictions du modèle.
    Paramètres:
                df: dataframe contenant les prédictions et la transcription réelle
                i : indice 
    Return: 
                booléan
    """
    return (df.transcript[i] == df.predict_1[i]) | \
            (df.transcript[i] == df.predict_2[i]) | \
            (df.transcript[i] == df.predict_3[i]) | \
            (df.transcript[i] == df.predict_4[i]) | \
            (df.transcript[i] == df.predict_5[i]) 

def transcript_in_pred1(df, i):
    """
    La fonction transcript_in_top5 vérifie si la transcription réelle est identique à la première prédiction du modèle.
    Paramètres:
                df: dataframe contenant les prédictions et la transcription réelle
                i : indice 
    Return: 
                booléan
    """
    return (df.transcript[i] == df.predict_1[i])

def viz_pred5(df):
    """
    La fonction viz_pred5 permet de visualiser 3 images au hasard avec prédiction et vraie transcription
    """
    # Afficher 3 images du jeu de données et sa transcription
    img_index = sample(range(0,df.shape[0]),3)
    for i in img_index:
        titre = "top_5: {}, {}, {}, {}, {} \n true transcription : {}".format(df.predict_1[i], 
                                                                              df.predict_2[i], 
                                                                              df.predict_3[i], 
                                                                              df.predict_4[i], 
                                                                              df.predict_5[i], 
                                                                              df.transcript[i])
        image = cv2.imread(df.data_path[i], cv2.IMREAD_GRAYSCALE)
        plt.imshow(image, cmap = 'gray')
        plt.title(titre)
        plt.show()
        
def cer(ligne):
    """
    La fonction cer calcule le Character Error Rate pour le première prédiction
    """
    ref = list(str(ligne.transcript))
    hyp = list(str(ligne.predict_1))
    return round(wer(ref, hyp), 2)  