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
import avae_model
import model_bw_w_latent_spaces as model_bw
import model_bw_w_latent_spaces_and_race as model_bw_with_race

def main():
    vae_models.set_seeds()

    DIR = os.environ["PROJECT_DIR"]

    """
    ********************** 
    Get Disadvantage and Psych Latent Spaces
    **********************
    """

    # Reset
    K.clear_session()

    # Create Encoder

    adv_hidden = [15,12,9,6,4]
    psych_hidden = [7,6,5,4,3]
    hidden = [7,6,5,4,3]
    activation = 'elu'
    latent_space_dim = 1

    encoder = avae_model.Encoder(adv_hidden, psych_hidden, activation, latent_space_dim)

    # Create Decoder 

    disadv_output_size = 6
    disadv_categorical_output_size = 10
    psych_output_size = 8

    decoder = avae_model.Decoder(hidden, activation, disadv_output_size, disadv_categorical_output_size, psych_output_size)

    # Create Predictor

    width = 2
    kw = {'activation':'elu', 'kernel_initializer':tf.random_uniform_initializer(), 'kernel_regularizer':keras.regularizers.l2()}
    okw = {'activation':'sigmoid', 'kernel_initializer':tf.random_uniform_initializer(), 'kernel_regularizer':keras.regularizers.l2()}

    predictor = avae_model.Predictor(width, kw, okw)

    # Create Optimizers
    opt1 = keras.optimizers.RMSprop()
    opt2 = keras.optimizers.RMSprop()

    # Create Race Controlled Autoencoder

    RcAE = avae_model.RaceControlledAutoencoder(encoder, decoder, predictor, opt1, opt2)

    RcAE.compile()

    # Load Data
    psych_filepath = DIR + 'data/psych.csv'
    disadv_filepath = DIR + 'data/disadv.csv'
    X_psych, X_race = vae_models.load_data(psych_filepath, race=False)
    X_adv, X_race = vae_models.load_data(disadv_filepath, race=False)
    X_race = pd.DataFrame(X_race).astype(float)

    # Train Race Controlled Autoencoder
    epochs = 500
    batch_size = 128
    final_latent_space = pd.DataFrame()
    final_reconstruction = pd.DataFrame()
    final_race_correlates = pd.DataFrame()
    final_race = pd.DataFrame()
    all_latent_spaces = []

    kf = StratifiedKFold(n_splits=5, random_state=111, shuffle=True)
    count = 1
    for train_index, test_index in kf.split(X_adv, X_race):
        print('Fold %d' % count)
        
        # Split Data
        x_adv_train_inputs, x_adv_test_inputs = X_adv.loc[train_index], X_adv.loc[test_index]
        x_psych_train_inputs, x_psych_test_inputs = X_psych.loc[train_index], X_psych.loc[test_index]
        x_train_race, x_test_race = X_race.loc[train_index], X_race.loc[test_index]

        # Normalize
        adv_scaler = StandardScaler()
        psych_scaler = StandardScaler()
        x_adv_train_inputs, x_adv_test_inputs = vae_models.normalize_data(adv_scaler, x_adv_train_inputs, x_adv_test_inputs)
        x_psych_train_inputs, x_psych_test_inputs = vae_models.normalize_data(psych_scaler, x_psych_train_inputs, x_psych_test_inputs)
            
        # Fit
        history = RcAE.fit([np.array(x_adv_train_inputs), np.array(x_psych_train_inputs), np.array(x_train_race)], epochs=epochs, batch_size=batch_size, verbose=2, shuffle=True)

        # Predict
        latent_space, reconstruction, race_predictions = RcAE.predict([np.array(x_adv_test_inputs), np.array(x_psych_test_inputs), np.array(x_test_race)])
        
        # Reconstruction
        adv_inputs_rec = vae_models.unnormalize_data(adv_scaler, pd.DataFrame(reconstruction[0], columns=x_adv_test_inputs.columns))
        adv_inputs_rec.index = test_index
        psych_inputs_rec = vae_models.unnormalize_data(psych_scaler, pd.DataFrame(reconstruction[1], columns=x_psych_test_inputs.columns))
        psych_inputs_rec.index = test_index
        race_rec = pd.DataFrame(reconstruction[2], index=test_index, columns=x_test_race.columns)
        reconstruction = pd.concat([adv_inputs_rec, psych_inputs_rec, race_rec], axis=1)    
        final_reconstruction = pd.concat([final_reconstruction, reconstruction])

        # Predictions
        final_race = pd.concat([final_race, pd.DataFrame(race_predictions, index=test_index)])
        
        # Latent Space
        final_latent_space = pd.concat([final_latent_space, pd.DataFrame(latent_space, index=test_index)])

        # Get Reconstruction Error
        x_adv_test_inputs.index = test_index
        x_psych_test_inputs.index = test_index
        og_fold_X = pd.concat([x_adv_test_inputs, x_psych_test_inputs, x_test_race], axis=1)
        reconstruction_loss = np.sum((og_fold_X - reconstruction) ** 2, axis=1).mean()
        print('Fold %d Reconstruction Loss: %f' % (count, reconstruction_loss))

        # Get Prediction Error
        race_error = roc_auc_score(x_test_race, pd.DataFrame(race_predictions).sort_index())
        print('Fold %d Race ROC AUC: %f' % (count, race_error))
        
        count += 1
        
        # Permute
        all_latent_spaces = vae_models.permute(all_latent_spaces, x_adv_test_inputs, x_psych_test_inputs, x_test_race, test_index, RcAE)

    # Get Reconstruction Error
    og_X = pd.concat([X_adv, X_psych, X_race], axis=1)
    final_reconstruction = final_reconstruction.sort_index()
    race_blind_reconstruction_loss = np.sum((og_X - final_reconstruction) ** 2, axis=1).mean()
    print('Total Reconstruction Loss: %f' % reconstruction_loss)

    # Get Prediction Error
    final_race = final_race.sort_index()
    race_error = roc_auc_score(X_race, final_race)
    print('Race ROC AUC: %f' % race_error)

    # Latent Space
    race_blind_latent_space = final_latent_space.sort_index()

    # All Latent Spaces Permuted
    race_blind_all_latent_spaces_permuted = []
    for i in all_latent_spaces:
        race_blind_all_latent_spaces_permuted.append(i.sort_index())

    # Save Model
    RcAE.save(DIR + 'models/race_blind_model')

    """
    ********************** 
    Save Latent Spaces
    **********************
    """

    race_blind_latent_space = pd.concat([race_blind_latent_space.iloc[:,0], race_blind_latent_space.iloc[:,1]], axis=1)
    race_blind_latent_space.columns = ['Disadv', 'Psych']
    race_blind_latent_space.to_csv(DIR + 'latent_spaces/race_blind/race_blind_latent_space.csv', sep='\t', index=None)
    
    """
    ********************** 
    Save Permuted Latent Spaces
    **********************
    """

    for i in race_blind_all_latent_spaces_permuted:    
        permuted_col = i.columns[1] if i.columns[0] == 0 else i.columns[0]
        temp = i
        temp.columns = [0,1]
        temp.to_csv(DIR + 'latent_spaces/race_blind/race_blind_permuted_%s.csv' % permuted_col, sep='\t', index=None)

    """
    ********************** 
    Predict Race With CV
    **********************
    """

    # Reset keras
    tf.keras.backend.clear_session()

    # Load Data
    disadv_filepath = DIR + 'data/disadv.csv'
    _, y = vae_models.load_data(disadv_filepath, race=False)
    X = pd.concat([race_blind_latent_space.iloc[:,0], race_blind_latent_space.iloc[:,1]], axis=1)
    y = pd.DataFrame(y)

    # Set Parameters
    width = 2
    depth = 0
    kw = {'activation':'elu', 'kernel_initializer':tf.random_uniform_initializer(), 'kernel_regularizer':keras.regularizers.l2()}
    okw = {'activation':'sigmoid', 'kernel_initializer':tf.random_uniform_initializer(), 'kernel_regularizer':keras.regularizers.l2()}
    epochs = 50
    batch_size = None 
    restarts = 1
    input_size = 2
    output_size = 1

    # Build and Train Model
    race_blind_nn_model = vae_models.build_race_model(input_size, output_size, width, depth, kw, okw)
    race_blind_nn_outputs = vae_models.run_race_model(restarts, X, y, race_blind_nn_model, epochs, batch_size)

    # Get Metrics
    race_blind_nn_accuracy = accuracy_score(y, np.round(race_blind_nn_outputs))
    race_blind_nn_auc = roc_auc_score(y, race_blind_nn_outputs)

    """
    ********************** 
    Predict Race No CV
    **********************
    """

    # Reset keras
    tf.keras.backend.clear_session()

    # Load Data
    disadv_filepath = DIR + 'data/disadv.csv'
    _, y = vae_models.load_data(disadv_filepath, race=False)
    X = pd.concat([race_blind_latent_space.iloc[:,0], race_blind_latent_space.iloc[:,1]], axis=1)
    y = pd.DataFrame(y)

    # Set Parameters
    width = 2
    depth = 0
    kw = {'activation':'elu', 'kernel_initializer':tf.random_uniform_initializer(), 'kernel_regularizer':keras.regularizers.l2()}
    okw = {'activation':'sigmoid', 'kernel_initializer':tf.random_uniform_initializer(), 'kernel_regularizer':keras.regularizers.l2()}
    epochs = 50
    batch_size = None 
    restarts = 1
    input_size = 2
    output_size = 1

    # Build and Train Model
    race_blind_nn_model_nocv = vae_models.build_race_model(input_size, output_size, width, depth, kw, okw)
    race_blind_nn_outputs_nocv = vae_models.run_race_model_nocv(restarts, X, y, race_blind_nn_model_nocv, epochs, batch_size)

    # Get Metrics
    race_blind_nn_accuracy_nocv = accuracy_score(y, np.round(race_blind_nn_outputs_nocv))
    race_blind_nn_auc_nocv = roc_auc_score(y, race_blind_nn_outputs_nocv)

    """
    ********************** 
    Model Birthweight with Latent Spaces
    **********************
    """

    baseline = DIR + 'latent_spaces/race_blind/race_blind_latent_space.csv'
    directory = DIR + 'latent_spaces/race_blind/'
    latent_type = 'race_blind'
    model_bw.model_birthweight(baseline, directory, latent_type)

    """
    ********************** 
    Model Birthweight with Latent Spaces and Race
    **********************
    """

    baseline = DIR + 'latent_spaces/race_blind/race_included_latent_space.csv'
    directory = DIR + 'latent_spaces/race_blind/'
    latent_type = 'race_blind'
    model_bw_with_race.model_birthweight(baseline, directory, latent_type) 
    
if __name__ == "__main__":
    main()