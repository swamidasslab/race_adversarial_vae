import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import os
import random
import tensorflow as tf
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import backend as K
from matplotlib.lines import Line2D
from keras.optimizers import RMSprop
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import layers
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from keras.constraints import Constraint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import json 
from scipy.stats import sem
import copy
import vae_models

class Encoder(keras.Model):
    def __init__(self, diadv_hidden, psych_hidden, activation, latent_space_dim):
        super(Encoder, self).__init__()
        self.diadv_hidden = diadv_hidden
        self.psych_hidden = psych_hidden
        self.activation = activation
        self.latent_space_dim = latent_space_dim
        
        self.dense_1 = layers.Dense(self.diadv_hidden[0], activation=self.activation)
        self.dense_2 = [layers.Dense(i, activation=self.activation) for i in self.diadv_hidden[1:]]
        self.dense_3 = layers.Dense(self.latent_space_dim)
        self.dense_4 = layers.Dense(self.latent_space_dim)
        
        self.dense_5 = layers.Dense(self.psych_hidden[0], activation=self.activation)
        self.dense_6 = [layers.Dense(i, activation=self.activation) for i in self.psych_hidden[1:]]
        self.dense_7 = layers.Dense(self.latent_space_dim)
        self.dense_8 = layers.Dense(self.latent_space_dim)
        
    def call(self, data):
        # Get data
        disadv_inputs, psych_inputs, race = data[0], data[1], data[2]
        
        # Concatentate Inputs
        I_1 = layers.Concatenate(axis=1)([disadv_inputs, race])
        
        # Run Hidden Layers
        H_1 = self.dense_1(I_1)
        for i in self.dense_2:
            H_1 = i(H_1)
        
        # Calculate Latent Space
        mu_1 = self.dense_3(H_1)
        log_var_1 = self.dense_4(H_1)
        eps_1 = tf.random.normal(shape=[tf.shape(mu_1)[0], self.latent_space_dim], mean=0.0, stddev=1.0)
        out_1 = mu_1 + log_var_1 * eps_1
        
        # Concatentate Inputs
        I_2 = layers.Concatenate(axis=1)([psych_inputs, race])
        
        # Run Hidden Layers
        H_2 = self.dense_5(I_2)
        for i in self.dense_6:
            H_2 = i(H_2)
        
        # Calculate Latent Space
        mu_2 = self.dense_7(H_2)
        log_var_2 = self.dense_8(H_2)
        eps_2 = tf.random.normal(shape=[tf.shape(mu_2)[0], self.latent_space_dim], mean=0.0, stddev=1.0)
        out_2 = mu_2 + log_var_2 * eps_2
        
        out = layers.Concatenate(axis=1)([out_1, out_2])
        mu = layers.Concatenate(axis=1)([mu_1, mu_2])
        log_var = layers.Concatenate(axis=1)([log_var_1, log_var_2])
        
        return out, mu, log_var

class Decoder(keras.Model):
    def __init__(self, hidden, activation, disadv_output_size, disadv_categorical_output_size, psych_output_size):
        super(Decoder, self).__init__()
        self.hidden = hidden
        self.activation = activation
        self.disadv_output_size = disadv_output_size
        self.disadv_categorical_output_size = disadv_categorical_output_size
        self.psych_output_size = psych_output_size
        
        self.dense_1 = layers.Dense(self.hidden[-1], activation=self.activation)
        self.dense_2 = [layers.Dense(i, activation=self.activation) for i in reversed(self.hidden[:-1])]
        self.dense_3 = layers.Dense(self.disadv_output_size)
        self.dense_4 = layers.Dense(self.disadv_categorical_output_size, activation='sigmoid')
        self.dense_5 = layers.Dense(1, activation='sigmoid')
        self.dense_6 = layers.Dense(self.psych_output_size)
        
    def call(self, data):
        # Get Data
        latent_space, race = data[0], data[1]
        
        # Concatentate Inputs
        I = layers.Concatenate(axis=1)([latent_space, race])

        # Run Hidden Layers
        H = self.dense_1(I)
        for i in self.dense_2:
            H = i(H)

        # Calculate Reconstruction
        continuous_outputs = self.dense_3(H)
        categorical_outputs = self.dense_4(H)
        disadv_outputs = layers.Concatenate(axis=1)([continuous_outputs, categorical_outputs])
        
        race_output = self.dense_5(H)
        psych_outputs = self.dense_6(H)
        
        return disadv_outputs, psych_outputs, race_output
        
    
class Predictor(keras.Model):
    def __init__(self, width, kw, okw):
        super(Predictor, self).__init__()
        self.width = 2
        self.kw = {'activation':'elu', 'kernel_initializer':tf.random_uniform_initializer(), 'kernel_regularizer':keras.regularizers.l2()}
        self.okw = {'activation':'sigmoid', 'kernel_initializer':tf.random_uniform_initializer(), 'kernel_regularizer':keras.regularizers.l2()}
        
        self.dense_1 = layers.Dense(width, **kw)
        self.dense_2 = layers.Dense(1, **okw)

    def call(self, I):
        # Train
        H = self.dense_1(I)

        # Compute Output
        race_predictions = self.dense_2(H)

        return race_predictions
    
class RaceControlledAutoencoder(keras.Model):
    def __init__(self, encoder, decoder, predictor, opt1, opt2):
        super(RaceControlledAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.predictor = predictor
        self.opt1 = opt1
        self.opt2 = opt2
        
    def call(self, data):
        # Get Inputs
        disadv_inputs, psych_inputs, race = data[0], data[1], data[2]
        
        # Run Encoder
        out, mu, log_var = self.encoder([disadv_inputs, psych_inputs, race])
        
        # Run Decoder
        hat_disadv_inputs, hat_psych_inputs, hat_race = self.decoder([out, race])
        
        # Calculate Loss
        reconstruction_loss_disadv_inputs = tf.reduce_mean(tf.square(disadv_inputs - hat_disadv_inputs))
        reconstruction_loss_psych_inputs = tf.reduce_mean(tf.square(psych_inputs - hat_psych_inputs))
        reconstruction_loss_race = tf.reduce_mean(tf.square(race - hat_race))
        kl_loss = -0.5 * keras.backend.sum(1.0 + log_var - keras.backend.square(mu) - keras.backend.exp(log_var), axis=1)
        loss = reconstruction_loss_disadv_inputs + reconstruction_loss_psych_inputs + reconstruction_loss_race + kl_loss

        # Run Predictor
        hat_race_predictions = self.predictor(mu)

        # Caclulate Loss
        bce = keras.losses.BinaryCrossentropy(from_logits=True)
        predictor_loss = bce(race, hat_race_predictions) 

        # Combine Losses
        encoder_decoder_loss = loss - predictor_loss
        
        return mu, [hat_disadv_inputs, hat_psych_inputs, hat_race], hat_race_predictions

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
            
        disadv_inputs, psych_inputs, race = data[0], data[1], data[2]
        
        with tf.GradientTape(persistent=True) as tape:

            # Run Encoder
            out, mu, log_var = self.encoder([disadv_inputs, psych_inputs, race])
            
            # Run Decoder
            hat_disadv_inputs, hat_psych_inputs, hat_race = self.decoder([out, race])
            
             # Calculate Loss
            reconstruction_loss_disadv_inputs = tf.reduce_mean(tf.square(disadv_inputs - hat_disadv_inputs))
            reconstruction_loss_psych_inputs = tf.reduce_mean(tf.square(psych_inputs - hat_psych_inputs))
            reconstruction_loss_race = tf.reduce_mean(tf.square(race - hat_race))
            kl_loss = -0.5 * keras.backend.sum(1.0 + log_var - keras.backend.square(mu) - keras.backend.exp(log_var), axis=1)
            loss = reconstruction_loss_disadv_inputs + reconstruction_loss_psych_inputs + reconstruction_loss_race + kl_loss

            # Run Predictor
            hat_race_predictions = self.predictor(mu)

            # Caclulate Loss
            bce = keras.losses.BinaryCrossentropy(from_logits=True)
            predictor_loss = bce(race, hat_race_predictions) 

            # Combine Losses
            encoder_decoder_loss = loss - predictor_loss

        # Take step on Encoder/Decoder using Encoder/Decoder Loss
        encoder_decoder_gradients = tape.gradient(encoder_decoder_loss, self.encoder.trainable_weights + self.decoder.trainable_weights)
        self.opt1.apply_gradients(zip(encoder_decoder_gradients, self.encoder.trainable_weights + self.decoder.trainable_weights))

        # Take step on Predictor using Predictor Loss
        predictor_gradients = tape.gradient(predictor_loss, self.predictor.trainable_weights)
        self.opt2.apply_gradients(zip(predictor_gradients, self.predictor.trainable_weights))
        
        return {'predictor_loss': predictor_loss, 'vae_loss': encoder_decoder_loss}
    