import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def train_nn(weights = None):
    #tf.reset_default_graph()
    #session = tf.Session()
    #tf.keras.backend.set_session(session)

    model = keras.Sequential([
                layers.Dense(64, activation='sigmoid', input_shape=(24,)),
                layers.Dense(128, activation='sigmoid'),
                layers.Dense(128, activation='sigmoid'),
                layers.Dense(1, activation='sigmoid')
            ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    dataset = pd.read_csv('dataset.csv')

    X_train, y_train = dataset, dataset['smoking']

    X_train.drop(['smoking'], inplace=True, axis=1)
    X_train.drop(dataset.columns[0], axis=1, inplace=True)

    if weights is not None:
        new_weights_layer1 = weights["Layer1_weights"]
        new_weights_layer2 = weights["Layer2_weights"]
        new_weights_layer3 = weights["Layer3_weights"]
        new_weights_layer4 = weights["Layer4_weights"]
        new_biases_layer1 = weights["Layer1_biases"]
        new_biases_layer2 = weights["Layer2_biases"]
        new_biases_layer3 = weights["Layer3_biases"]
        new_biases_layer4 = weights["Layer4_biases"]

        model.layers[0].set_weights([np.array(new_weights_layer1), np.array(new_biases_layer1)])
        model.layers[1].set_weights([np.array(new_weights_layer2), np.array(new_biases_layer2)])
        model.layers[2].set_weights([np.array(new_weights_layer3), np.array(new_biases_layer3)])
        model.layers[3].set_weights([np.array(new_weights_layer4), np.array(new_biases_layer4)])
        
    model.fit(X_train, y_train, epochs=10, verbose=1)

    weights = model.get_weights()
    #session.close()
    return weights