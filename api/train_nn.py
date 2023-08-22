import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


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
        model.set_weights(weights)
        
    model.fit(X_train, y_train, epochs=10, verbose=1)


    weights = model.get_weights()

    #session.close()

    return weights