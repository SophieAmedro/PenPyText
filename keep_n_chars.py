"""
Created on Tue Dec 02 2021

@author: samy_ait-ameur, sophie_amedro, stephane_tchatat
"""

def max_n_chars(dataset, max_length: int):
    """
    La fonction max_n_chars supprime du dataset les données dont la transcription est composée de plus de max_lenght caractères.
    Paramètres:
            dataset : dataframe contenant la transcription
            max_length : int nombre de caractères maximum dans la transcription
    Return
            dataset : dataframe dont les transcriptions ont une taille inférieur ou égale à max_length 
    """
    dataset["count_char"] = dataset.transcript.apply(lambda x: len(list(x)))
    dataset = dataset.loc[dataset.count_char <= max_length]
    dataset.reset_index(inplace=True, drop=True)
    return dataset