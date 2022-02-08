import numpy as np
import pandas as pd

import tensorflow as tf

# Décodage
import tensorflow.keras.backend as K


def calcul_cer(labels, predictions, max_len):
    labels_tensor = tf.cast(tf.sparse.from_dense(labels), dtype=tf.int64)

    # Predictions
    predictions_decoded = K.ctc_decode(predictions, input_length=np.ones(predictions.shape[0]) * predictions.shape[1], 
                                       greedy=True)[0][0][:, :max_len]
    # Convertir les predictions en tensor
    sparse_predictions = tf.cast(tf.sparse.from_dense(predictions_decoded), dtype=tf.int64)

    # Moyenne des distance
    edit_distances = tf.edit_distance(sparse_predictions, labels_tensor, normalize=False)
    return tf.reduce_mean(edit_distances)


class CerCallback(tf.keras.callbacks.Callback):
    def __init__(self, predictions):
        super().__init__()
        self.pred = predictions

    def on_epoch_end(self, epoch):
        edit_distances = []
        
        predictions = self.prediction_model.predict(validation_generator)
        for i in range(len(validationset)):
            labels = validationset[i]
            prediction = predictions[i]
            edit_distances.append(calcul_cer(labels, prediction).numpy())

        #print(f"Mean edit distance for epoch {epoch + 1}: {np.mean(edit_distances):.5f}")
        print("Moyenne CER pour l'epoch {}: {}".format(epoch + 1, round(np.mean(edit_distances), 2)))
