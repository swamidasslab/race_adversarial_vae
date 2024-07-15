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
    encoder, decoder, vae = vae_models.build_race_only_VAE(input_size=input_size, hidden=hidden, cont_output_size=cont_output_size, cat_output_size=cat_output_size, latent_space_dim=latent_space_dim)

    # Load data
    disadv_filepath = DIR + 'data/disadv.csv'
    og_X, og_y = vae_models.load_data(disadv_filepath, race=False)
    X, y = vae_models.load_data(disadv_filepath, race=False)

    # Train VAE
    race_only_disadv_reconstruction, race_only_disadv_latent_space, race_only_disadv_loss, race_only_disadv_model, race_only_all_disadv_latent_spaces_permuted = vae_models.train_VAE(encoder, decoder, vae, og_X, og_y, X, y, race=False, latent_type='Disadv', only_race=True)

    # Save Model
    race_only_disadv_model.save(DIR + 'models/race_only_disadv_model')

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
    encoder, decoder, vae = vae_models.build_race_only_VAE(input_size=input_size, hidden=hidden, cont_output_size=cont_output_size, latent_space_dim=latent_space_dim)

    # Load data
    psych_filepath = DIR + 'data/psych.csv'
    X, y = vae_models.load_data(psych_filepath, race=False)

    # Train VAE
    race_only_psych_reconstruction, race_only_psych_latent_space, race_only_psych_loss, race_only_psych_model, race_only_all_psych_latent_spaces_permuted = vae_models.train_VAE(encoder, decoder, vae, X, y, X, y, race=False, categorical=False, latent_type='Psych', control=True, only_race=True)

    # Save Model
    race_only_psych_model.save(DIR + 'models/race_only_psych_model')
    
    """
    ********************** 
    Save Latent Spaces
    **********************
    """

    race_only_latent_space = pd.concat([-race_only_disadv_latent_space, race_only_psych_latent_space], axis=1)
    race_only_latent_space.columns = ['Disadv', 'Psych']
    race_only_latent_space.to_csv(DIR + 'latent_spaces/race_only/race_only_latent_space.csv', sep='\t', index=None)

    """
    ********************** 
    Model Birthweight with Latent Spaces and Race
    **********************
    """  

    color = 'grey'
    directory = DIR + 'latent_spaces/race_only/'

    # Set Parameters
    X, y = model_bw_with_race.load_data()
    input_size = 1
    output_size = 1
    width = 0
    depth = 0
    kw = {'activation':'linear', 'kernel_initializer':tf.random_uniform_initializer(), 'kernel_regularizer':keras.regularizers.l2()}
    okw = {'activation':'linear', 'kernel_initializer':tf.random_uniform_initializer(), 'kernel_regularizer':keras.regularizers.l2()}
    epochs = 100
    batch_size = None 
    hidden = False
    restarts = 5

    X = pd.DataFrame(X['child_race___2'])
    X.columns = ['race']

    # LR
    filename = directory + 'LR/LR_with_race_outputs.csv'
    if os.path.exists(filename):
        lr_outputs = pd.read_csv(filename, sep='\t')
    else:
        lr_model = model_bw_with_race.custom_build_model(input_size, output_size, width, depth, kw, okw, hidden=hidden)        
        lr_outputs = model_bw_with_race.run_model(restarts, X, y, lr_model, 'LR', filename, epochs, batch_size)

    # Set Parameters
    X, y = model_bw_with_race.load_data()
    input_size = 1
    output_size = 1
    width = 2
    depth = 0
    kw = {'activation':'elu', 'kernel_initializer':tf.random_uniform_initializer(), 'kernel_regularizer':keras.regularizers.l2()}
    okw = {'activation':'linear', 'kernel_initializer':tf.random_uniform_initializer(), 'kernel_regularizer':keras.regularizers.l2()}
    epochs = 100
    batch_size = None 
    hidden = True
    restarts = 5

    X = pd.DataFrame(X['child_race___2'])
    X.columns = ['race']

    # NN
    filename = directory + 'NN/NN_with_race_outputs.csv'
    if os.path.exists(filename):
        nn_outputs = pd.read_csv(filename, sep='\t')
    else:
        nn_model = model_bw_with_race.custom_build_model(input_size, output_size, width, depth, kw, okw, hidden=hidden)
        nn_outputs = model_bw_with_race.run_model(restarts, X, y, nn_model, 'NN', filename, epochs, batch_size)

    # Rvals
    latent_type = 'race_only'
    filename = directory + 'rvals/rvals_with_race.csv'
    if os.path.exists(filename):
        rvals = pd.read_csv(filename, sep='\t')
    else:
        rvals = model_bw_with_race.get_rval_data(X, y, lr_outputs, nn_outputs, latent_type, filename, coloring=color, save=True)
        
    model_bw_with_race.get_rval_data(X, y, lr_outputs, nn_outputs, latent_type, filename, coloring=color, save=True)
    
if __name__ == "__main__":
    main()