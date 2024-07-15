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
import model_bw_w_latent_spaces as model_bw
import model_bw_w_latent_spaces_and_race as model_bw_with_race

def main():
    vae_models.set_seeds()

    DIR = os.environ["PROJECT_DIR"]

    """
    ********************** 
    Get Disadvantage Latent Space
    **********************
    """

    # Reset keras
    tf.keras.backend.clear_session()

    # Parameters
    input_size = 16
    hidden = [13,11,9,7,5,3]
    cont_output_size = 6
    cat_output_size = 10
    latent_space_dim = 1

    # Build VAE
    encoder, decoder, vae = vae_models.build_control_VAE(input_size=input_size, hidden=hidden, cont_output_size=cont_output_size, cat_output_size=cat_output_size, latent_space_dim=latent_space_dim)

    # Load data
    disadv_filepath = DIR + 'data/disadv.csv'
    og_X, og_y = vae_models.load_data(disadv_filepath, race=False)
    X, y = vae_models.load_data(disadv_filepath, race=False)

    # Train VAE
    race_control_disadv_reconstruction, race_control_disadv_latent_space, race_control_disadv_loss, race_control_disadv_model, race_control_all_disadv_latent_spaces_permuted = vae_models.train_VAE(encoder, decoder, vae, og_X, og_y, X, y, race=False, control=True, latent_type='Disadv')

    # Save Model
    race_control_disadv_model.save(DIR + 'models/race_control_disadv_model')

    """
    ********************** 
    Get Psych Latent Space
    **********************
    """

    # Reset keras
    tf.keras.backend.clear_session()

    # Parameters
    input_size = 8
    hidden = [7,6,5,4,3]
    cont_output_size = 8
    latent_space_dim = 1

    # Build VAE
    encoder, decoder, vae = vae_models.build_control_VAE(input_size=input_size, hidden=hidden, cont_output_size=cont_output_size, latent_space_dim=latent_space_dim)

    # Load data
    psych_filepath = DIR + 'data/psych.csv'
    X, y = vae_models.load_data(psych_filepath, race=False)

    # Train VAE
    race_control_psych_reconstruction, race_control_psych_latent_space, race_control_psych_loss, race_control_psych_model, race_control_all_psych_latent_spaces_permuted = vae_models.train_VAE(encoder, decoder, vae, X, y, X, y, race=False, categorical=False, control=True, latent_type='Psych')

    # Save Model
    race_control_psych_model.save(DIR + 'models/race_control_psych_model')

    """
    ********************** 
    Save Latent Spaces
    **********************
    """

    race_control_latent_space = pd.concat([race_control_disadv_latent_space[0], race_control_psych_latent_space[0]], axis=1)
    race_control_latent_space.columns = ['Disadv','Psych']
    race_control_latent_space.to_csv(DIR + 'latent_spaces/race_control/race_control_latent_space.csv', sep='\t', index=None)

    """
    ********************** 
    Save Permuted Latent Spaces
    **********************
    """

    race_control_all_latent_spaces_permuted = []
    for i in race_control_all_disadv_latent_spaces_permuted:
        race_control_all_latent_spaces_permuted.append(pd.concat([i, race_control_psych_latent_space.iloc[:,0]], axis=1))
    for i in race_control_all_psych_latent_spaces_permuted:
        race_control_all_latent_spaces_permuted.append(pd.concat([race_control_disadv_latent_space.iloc[:,0], i], axis=1))

    for i in race_control_all_latent_spaces_permuted:
        
        permuted_col = i.columns[1] if i.columns[0] == 0 else i.columns[0]
        temp = i
        temp.columns = [0,1]
        temp.to_csv(DIR + 'latent_spaces/race_control/race_control_permuted_%s.csv' % permuted_col, sep='\t', index=None)

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
    X = pd.concat([race_control_latent_space.iloc[:,0], race_control_latent_space.iloc[:,1]], axis=1)
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
    race_control_nn_model = vae_models.build_race_model(input_size, output_size, width, depth, kw, okw)
    race_control_nn_outputs = vae_models.run_race_model(restarts, X, y, race_control_nn_model, epochs, batch_size)

    # Get Metrics
    race_control_nn_accuracy = accuracy_score(y, np.round(race_control_nn_outputs))
    race_control_nn_auc = roc_auc_score(y, race_control_nn_outputs)

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
    X = pd.concat([race_control_latent_space.iloc[:,0], race_control_latent_space.iloc[:,1]], axis=1)
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
    race_control_nn_model_nocv = vae_models.build_race_model(input_size, output_size, width, depth, kw, okw)
    race_control_nn_outputs_nocv = vae_models.run_race_model_nocv(restarts, X, y, race_control_nn_model_nocv, epochs, batch_size)

    # Get Metrics
    race_control_nn_accuracy_nocv = accuracy_score(y, np.round(race_control_nn_outputs_nocv))
    race_control_nn_auc_nocv = roc_auc_score(y, race_control_nn_outputs_nocv)

    """
    ********************** 
    Model Birthweight with Latent Spaces
    **********************
    """

    baseline = DIR + 'latent_spaces/race_control/race_control_latent_space.csv'
    directory = DIR + 'latent_spaces/race_control/'
    latent_type = 'race_control'
    model_bw.model_birthweight(baseline, directory, latent_type)

    """
    ********************** 
    Model Birthweight with Latent Spaces and Race
    **********************
    """   

    baseline = DIR + 'latent_spaces/race_control/race_included_latent_space.csv'
    directory = DIR + 'latent_spaces/race_control/'
    latent_type = 'race_control'
    model_bw_with_race.model_birthweight(baseline, directory, latent_type) 

if __name__ == "__main__":
    main()