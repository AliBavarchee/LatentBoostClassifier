import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler
from scipy import linalg as la
from sklearn.decomposition import PCA

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

from tqdm import tqdm
import time
seed = 777
np.random.seed(seed)
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import keras_tuner as kt
from multiprocessing import Manager, Process
import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
#plt.style.use('default')
import os
import warnings
warnings.filterwarnings("ignore")
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR') 


# Define a custom loss layer for the CVAE
class CustomLossLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(CustomLossLayer, self).__init__(**kwargs)

    def call(self, inputs):
        inputs, reconstructed, z_mean, z_log_var = inputs
        reconstruction_loss = tf.keras.losses.MeanSquaredError()(inputs, reconstructed)
        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
        total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        self.add_loss(total_loss)
        return reconstructed

# Define the sampling function
@keras.utils.register_keras_serializable()
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.keras.backend.random_normal(shape=(tf.shape(z_mean)[0], tf.shape(z_mean)[1]))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# Define a function to build the CVAE model
def build_cvae(hp, X_test):
    """
    Build a Conditional Variational Autoencoder (CVAE) model.

    Args:
        hp (keras_tuner.HyperParameters): The hyperparameters to use for the model.
        X_test (np.ndarray): Test dataset to infer the input dimensionality.

    Returns:
        tf.keras.Model: The built CVAE model.
    """
    input_dim = X_test.shape[1]  # Dynamically infer input dimension
    latent_dim = hp.Int("latent_dim", min_value=input_dim, max_value=510, step=10)
    dense_units = hp.Int("dense_units", min_value=64, max_value=1032, step=64)
    learning_rate = hp.Choice("learning_rate", values=[1e-4, 1e-3, 1e-2])

    inputs = layers.Input(shape=(input_dim,), name="inputs")
    condition = layers.Input(shape=(1,), name="condition")
    x = layers.Concatenate(name="concat_inputs_condition")([inputs, condition])
    x = layers.Dense(dense_units, activation="relu", name="encoder_dense")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

    z = layers.Lambda(sampling, name="z")([z_mean, z_log_var])

    decoder_input = layers.Input(shape=(latent_dim,), name="decoder_input")
    decoder_condition = layers.Input(shape=(1,), name="decoder_condition")
    x = layers.Concatenate(name="concat_decoder_inputs_condition")([decoder_input, decoder_condition])
    x = layers.Dense(dense_units, activation="relu", name="decoder_dense")(x)
    outputs = layers.Dense(input_dim, activation="sigmoid", name="reconstructed_output")(x)
    decoder = models.Model([decoder_input, decoder_condition], outputs, name="decoder")

    reconstructed = decoder([z, condition])

    final_output = CustomLossLayer(name="custom_loss_layer")([inputs, reconstructed, z_mean, z_log_var])

    cvae = models.Model([inputs, condition], final_output, name="cvae")
    cvae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate))

    return cvae


# Define a function to build the CGAN model
def build_cgan(hp, X_test):
    """
    Build a Conditional Generative Adversarial Network (CGAN) model.

    Args:
        hp (keras_tuner.HyperParameters): The hyperparameters to use for the model.
        X_test (np.ndarray): Test dataset to infer the input dimensionality.

    Returns:
        tf.keras.Model: The built CGAN model.
    """
    input_dim = X_test.shape[1]  # Dynamically infer input dimension
    latent_dim = hp.Int("latent_dim", input_dim, 510, step=10)
    dense_units = hp.Int("dense_units", 64, 1032, step=64)
    learning_rate = hp.Choice("learning_rate", [1e-4, 1e-3, 1e-2])

    # Generator
    noise_input = layers.Input(shape=(latent_dim,), name="noise_input")
    condition_input = layers.Input(shape=(1,), name="condition_input")
    generator_x = layers.Concatenate()([noise_input, condition_input])
    generator_x = layers.Dense(dense_units, activation="relu")(generator_x)
    generator_x = layers.Dense(input_dim, activation="sigmoid")(generator_x)
    generator = models.Model([noise_input, condition_input], generator_x, name="generator")

    # Discriminator
    data_input = layers.Input(shape=(input_dim,), name="data_input")
    condition_input_discriminator = layers.Input(shape=(1,), name="condition_input_discriminator")
    discriminator_x = layers.Concatenate()([data_input, condition_input_discriminator])
    discriminator_x = layers.Dense(dense_units, activation="relu")(discriminator_x)
    discriminator_x = layers.Dense(1, activation="sigmoid")(discriminator_x)
    discriminator = models.Model(
        [data_input, condition_input_discriminator],
        discriminator_x,
        name="discriminator"
    )
    discriminator.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # CGAN Model
    noise = layers.Input(shape=(latent_dim,), name="cgan_noise_input")
    condition = layers.Input(shape=(1,), name="cgan_condition_input")
    generated_data = generator([noise, condition])
    discriminator.trainable = False
    validity = discriminator([generated_data, condition])
    cgan = models.Model([noise, condition], validity, name="cgan")
    cgan.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="binary_crossentropy")

    return cgan

# Define a function to tune the CVAE model
def tune_cvae(X_train, Y_train, X_test, Y_test, shared_dict):
    """
    Tune the CVAE model using the Hyperband tuner.

    Args:
        X_train (np.ndarray): The training data.
        Y_train (np.ndarray): The training labels.
        X_test (np.ndarray): The testing data.
        Y_test (np.ndarray): The testing labels.
        shared_dict (dict): A shared dictionary to store the results.
    """
    try:
        tuner = kt.Hyperband(
            lambda hp: build_cvae(hp, X_test),  # Pass X_test explicitly
            objective="loss",
            max_epochs=30,
            factor=3,
            directory="./cvae_tuning",
            project_name="cvae_tuning_project"
        )
        stop_early = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5)
        tuner.search(
            [X_train, Y_train], None,
            validation_data=([X_test, Y_test], None),
            callbacks=[stop_early]
        )
        best_cvae = tuner.get_best_models(num_models=1)[0]
        shared_dict["best_cvae"] = best_cvae
        logging.info("CVAE training successful.")
    except Exception as e:
        logging.error(f"Error tuning CVAE: {e}")
        best_cvae = build_cvae(kt.HyperParameters(), X_test)
        shared_dict["best_cvae"] = best_cvae
        logging.warning("Using default CVAE model due to training failure.")

# Define a function to tune the CGAN model
def tune_cgan(X_train, Y_train, X_test, Y_test, shared_dict):
    """
    Tune the CGAN model using the Hyperband tuner.

    Args:
        X_train (np.ndarray): The training data.
        Y_train (np.ndarray): The training labels.
        X_test (np.ndarray): The testing data.
        Y_test (np.ndarray): The testing labels.
        shared_dict (dict): A shared dictionary to store the results.
    """
    try:
        tuner = kt.Hyperband(
            lambda hp: build_cgan(hp, X_test),  # Pass X_test explicitly
            objective="loss",
            max_epochs=30,
            factor=3,
            directory="./cgan_tuning",
            project_name="cgan_tuning_project"
        )
        stop_early = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3)
        tuner.search(
            [X_train, Y_train], None,
            validation_data=([X_test, Y_test], None),
            callbacks=[stop_early]
        )
        best_cgan = tuner.get_best_models(num_models=1)[0]
        best_generator = best_cgan.get_layer("generator")
        shared_dict["best_cgan_generator"] = best_generator
        logging.info("CGAN training successful.")
    except Exception as e:
        logging.error(f"Error tuning CGAN: {e}")
        shared_dict["best_cgan_generator"] = build_cgan(kt.HyperParameters(), X_test).get_layer("generator")

# Define a function to tune the Random Forest model
def tune_random_forest(latent_features, synthetic_data, Y_train, shared_dict, X_test):
    """
    Tune the Random Forest model using GridSearchCV.

    Args:
        latent_features (np.ndarray): The latent features extracted from the CVAE.
        synthetic_data (np.ndarray): The synthetic data generated by the CGAN.
        Y_train (np.ndarray): The training labels.
        shared_dict (dict): A shared dictionary to store the results.
        X_test (np.ndarray): The test features used to determine the number of input dimensions.
    """
    try:
        # Determine the number of input dimensions
        input_dim = X_test.shape[1]

        # Combine latent features and synthetic data
        combined_X = np.vstack([latent_features, synthetic_data])
        combined_Y = np.hstack([Y_train, Y_train])  # Duplicate labels for synthetic data
        
        # Initialize the Random Forest Classifier
        rf = RandomForestClassifier(random_state=42)

        # Define the parameter grid
        param_grid = {
            "n_estimators": [input_dim, 50, 100, 200],  # Number of trees
            "max_depth": [5, 10, 20, None],            # Maximum depth of each tree
            "min_samples_split": [2, 5, 10, 50],       # Minimum samples required to split a node
            "min_samples_leaf": [1, 2, 4, 5]           # Minimum samples required in a leaf node
        }

        # GridSearch with cross-validation
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1
        )

        # Perform the grid search
        grid_search.fit(combined_X, combined_Y)

        # Save the best model in the shared dictionary
        best_rf = grid_search.best_estimator_
        shared_dict["rf_model"] = best_rf

        logging.info("Random Forest tuning successful.")
    except Exception as e:
        logging.error(f"Error tuning Random Forest: {e}")
