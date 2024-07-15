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

def set_seeds():
    # Seed value
    seed_value= 0
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

def load_data(filepath, race=True):
    # Load Data
    X = pd.read_csv(filepath)
    y = X['race']
    if not race:
        X = X.drop(columns=['race'])
    return X, y

def normalize(scaler, x_train, x_test):
    # Get continuous columns
    bool_cols = list(x_train.filter(regex='race').columns)
    cat_cols = list(x_train.filter(regex='insur').columns)
    [cat_cols.append(i) for i in x_train.filter(regex='education').columns]
    other_cols = set(x_train.columns) - set(bool_cols) - set(cat_cols)

    # Standardize continuous values
    x_train_cont = pd.DataFrame(scaler.fit_transform(x_train[list(other_cols)]))
    x_train_cont.columns = other_cols
    x_test_cont = pd.DataFrame(scaler.transform(x_test[list(other_cols)]))
    x_test_cont.columns = other_cols
    
    # Combine
    x_train = pd.concat([x_train_cont, x_train[list(cat_cols)].reset_index().iloc[:,1:], x_train[bool_cols].reset_index().iloc[:,1:]], axis=1)
    x_test = pd.concat([x_test_cont, x_test[list(cat_cols)].reset_index().iloc[:,1:], x_test[bool_cols].reset_index().iloc[:,1:]], axis=1)
    
    return x_train, x_test, list(other_cols) + list(cat_cols) + list(bool_cols)

def unnormalize(scaler, x):
    # Get continuous columns
    bool_cols = x.filter(regex='race').columns
    cat_cols = list(x.filter(regex='insur').columns)
    [cat_cols.append(i) for i in x.filter(regex='education').columns]
    other_cols = set(x.columns) - set(bool_cols) - set(cat_cols)

    # Standardize continuous values
    x_cont = pd.DataFrame(scaler.inverse_transform(x[other_cols]))
    x_cont.columns = other_cols
    
    # Combine
    x = pd.concat([x_cont, x[cat_cols].reset_index().iloc[:,1:], x[bool_cols].reset_index().iloc[:,1:]], axis=1)
    
    return x

def build_race_only_VAE(input_size=None, hidden=None, cont_output_size=None, cat_output_size=None, bin_output_size=None, latent_space_dim=None):
    # Encoder
    # Input
    x = layers.Input(shape=(input_size), name="encoder_input")

    # Hidden
    encoder_dense = layers.Dense(units=hidden[0], activation="elu", name='encoder_dense_1')(x)
    for idx, i in enumerate(hidden[1:]):
        encoder_dense = layers.Dense(units=i, activation="elu", name='encoder_dense_%s' % (idx+2))(encoder_dense)

    # Latent
    encoder_mu = layers.Dense(units=latent_space_dim, name="encoder_mu")(encoder_dense)
    encoder_log_variance = layers.Dense(units=latent_space_dim, name="encoder_log_variance")(encoder_dense)
    encoder_output = layers.Lambda(sampling, name="encoder_output")([encoder_mu, encoder_log_variance])

    # Build Encoder
    encoder = keras.Model(x, [encoder_output, encoder_mu], name="encoder_model")

    # Decoder
    # Input
    decoder_input = layers.Input(shape=(1), name="decoder_input")

    # Hidden
    decoder_dense = layers.Dense(hidden[-1], activation="elu", name='decoder_dense_1')(decoder_input)
    for idx, i in enumerate(reversed(hidden[:-1])):
        decoder_dense = layers.Dense(i, activation="elu", name='decoder_dense_%s' % (idx+2))(decoder_dense)

    # Output
    continuous_output = layers.Dense(cont_output_size, name='decoder_continuous_output')(decoder_dense)
    decoder_output = [continuous_output]
    if cat_output_size:
        categorical_output = layers.Dense(cat_output_size, activation="sigmoid", name='decoder_categorical_output')(decoder_dense)
        decoder_output.append(categorical_output)
    if bin_output_size:
        binary_output = layers.Dense(bin_output_size, activation="sigmoid", name='decoder_binary_output')(decoder_dense)
        decoder_output.append(binary_output)

    # Build Decoder
    decoder = keras.Model(decoder_input, decoder_output, name="decoder_model")

    # VAE
    vae_input = layers.Input(shape=(input_size), name="VAE_input")
    race = layers.Input(shape=(1), name="race_input")
    vae_encoder_output, vae_encoder_mu = encoder(vae_input)
    vae_decoder_output = decoder(race)
    vae = keras.Model([vae_input, race], vae_decoder_output, name="VAE")

    # Compile
    vae.compile(optimizer=keras.optimizers.RMSprop(), loss=loss_func(encoder_mu, encoder_log_variance))
    
    return encoder, decoder, vae

def build_VAE(input_size=None, hidden=None, cont_output_size=None, cat_output_size=None, bin_output_size=None, latent_space_dim=None):
    # Encoder
    # Input
    x = layers.Input(shape=(input_size), name="encoder_input")

    # Hidden
    encoder_dense = layers.Dense(units=hidden[0], activation="elu", name='encoder_dense_1')(x)
    for idx, i in enumerate(hidden[1:]):
        encoder_dense = layers.Dense(units=i, activation="elu", name='encoder_dense_%s' % (idx+2))(encoder_dense)

    # Latent
    encoder_mu = layers.Dense(units=latent_space_dim, name="encoder_mu")(encoder_dense)
    encoder_log_variance = layers.Dense(units=latent_space_dim, name="encoder_log_variance")(encoder_dense)
    encoder_output = layers.Lambda(sampling, name="encoder_output")([encoder_mu, encoder_log_variance])

    # Build Encoder
    encoder = keras.Model(x, [encoder_output, encoder_mu], name="encoder_model")

    # Decoder
    # Input
    decoder_input = layers.Input(shape=(latent_space_dim), name="decoder_input")

    # Hidden
    decoder_dense = layers.Dense(hidden[-1], activation="elu", name='decoder_dense_1')(decoder_input)
    for idx, i in enumerate(reversed(hidden[:-1])):
        decoder_dense = layers.Dense(i, activation="elu", name='decoder_dense_%s' % (idx+2))(decoder_dense)

    # Output
    continuous_output = layers.Dense(cont_output_size, name='decoder_continuous_output')(decoder_dense)
    decoder_output = [continuous_output]
    if cat_output_size:
        categorical_output = layers.Dense(cat_output_size, activation="sigmoid", name='decoder_categorical_output')(decoder_dense)
        decoder_output.append(categorical_output)
    if bin_output_size:
        binary_output = layers.Dense(bin_output_size, activation="sigmoid", name='decoder_binary_output')(decoder_dense)
        decoder_output.append(binary_output)

    # Build Decoder
    decoder = keras.Model(decoder_input, decoder_output, name="decoder_model")

    # VAE
    vae_input = layers.Input(shape=(input_size), name="VAE_input")
    vae_encoder_output, vae_encoder_mu = encoder(vae_input)
    vae_decoder_output = decoder(vae_encoder_output)
    vae = keras.Model(vae_input, vae_decoder_output, name="VAE")

    # Compile
    vae.compile(optimizer=keras.optimizers.RMSprop(), loss=loss_func(encoder_mu, encoder_log_variance))
    
    return encoder, decoder, vae

def build_control_VAE(input_size=None, hidden=None, cont_output_size=None, cat_output_size=None, bin_output_size=None, latent_space_dim=None):
    # Encoder
    # Input
    x = layers.Input(shape=(input_size), name="encoder_input")
    race = layers.Input(shape=(1), name='race')
    
    # Hidden
    encoder_dense = layers.Dense(units=hidden[0], activation="elu", name='encoder_dense_1')(x)
    for idx, i in enumerate(hidden[1:]):
        encoder_dense = layers.Dense(units=i, activation="elu", name='encoder_dense_%s' % (idx+2))(encoder_dense)

    # Latent
    encoder_mu = layers.Dense(units=latent_space_dim, name="encoder_mu")(encoder_dense)
    encoder_mu = layers.Concatenate(axis=1)([encoder_mu, race])
    encoder_log_variance = layers.Dense(units=latent_space_dim, name="encoder_log_variance")(encoder_dense)
    encoder_output = layers.Lambda(sampling, name="encoder_output")([encoder_mu, encoder_log_variance])

    # Build Encoder
    encoder = keras.Model([x, race], [encoder_output, encoder_mu], name="encoder_model")

    # Decoder
    # Input
    decoder_input = layers.Input(shape=(latent_space_dim+1), name="decoder_input")

    # Hidden
    decoder_dense = layers.Dense(hidden[-1], activation="elu", name='decoder_dense_1')(decoder_input)
    for idx, i in enumerate(reversed(hidden[:-1])):
        decoder_dense = layers.Dense(i, activation="elu", name='decoder_dense_%s' % (idx+2))(decoder_dense)

    # Output
    continuous_output = layers.Dense(cont_output_size, name='decoder_continuous_output')(decoder_dense)
    decoder_output = [continuous_output]
    if cat_output_size:
        categorical_output = layers.Dense(cat_output_size, activation="sigmoid", name='decoder_categorical_output')(decoder_dense)
        decoder_output.append(categorical_output)
    if bin_output_size:
        binary_output = layers.Dense(bin_output_size, activation="sigmoid", name='decoder_binary_output')(decoder_dense)
        decoder_output.append(binary_output)

    # Build Decoder
    decoder = keras.Model(decoder_input, decoder_output, name="decoder_model")

    # VAE
    data = layers.Input(shape=(input_size), name="VAE_input")
    race = layers.Input(shape=(1), name="race_input")
    vae_input = [data, race]
    vae_encoder_output, vae_encoder_mu = encoder(vae_input)
    vae_decoder_output = decoder(vae_encoder_output)
    vae = keras.Model(vae_input, vae_decoder_output, name="VAE")

    # Compile
    vae.compile(optimizer=keras.optimizers.RMSprop(), loss=loss_func(encoder_mu, encoder_log_variance))
    
    return encoder, decoder, vae

def loss_func(encoder_mu, encoder_log_variance):
    def vae_reconstruction_loss(y_true, y_predict):
        reconstruction_loss_factor = 1000
        reconstruction_loss = keras.backend.mean(keras.backend.square(y_true-y_predict))
        return reconstruction_loss_factor * reconstruction_loss

    def vae_kl_loss(encoder_mu, encoder_log_variance):
        kl_loss = -0.5 * keras.backend.sum(1.0 + encoder_log_variance - keras.backend.square(encoder_mu) - keras.backend.exp(encoder_log_variance), axis=1)
        return kl_loss

    def vae_loss(y_true, y_predict):
        reconstruction_loss = vae_reconstruction_loss(y_true, y_predict)
        kl_loss = vae_kl_loss(y_true, y_predict)
        loss = reconstruction_loss + kl_loss
        return loss

    return vae_loss

def sampling(mu_log_variance):
    mu, log_variance = mu_log_variance
    epsilon = keras.backend.random_normal(shape=keras.backend.shape(mu), mean=0.0, stddev=1.0)
    random_sample = mu + keras.backend.exp(log_variance/2) * epsilon
    return random_sample

def train_VAE(encoder, decoder, vae, og_X, og_y, X, y, race=True, categorical=True, control=False, latent_type=None, normalization=True, permutation=True, only_race=False):
    scaler = StandardScaler()
    skf = StratifiedKFold(n_splits=5)
    
    latent_space = pd.DataFrame()
    reconstruction = pd.DataFrame()
    all_latent_spaces = []
    
    for train_index, test_index in skf.split(X, y):
        x_train, x_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        
        if normalization:
            x_train, x_test, column_ordering = normalize(scaler, x_train, x_test)

        outputs = []
        
        # Get columns
        bool_cols = []
        if race:
            bool_cols = list(x_train.filter(regex='race').columns)
        cat_cols = []
        if categorical:
            cat_cols = list(x_train.filter(regex='insur').columns)
            [cat_cols.append(i) for i in x_train.filter(regex='education').columns]
        other_cols = list(set(x_train.columns) - set(bool_cols) - set(cat_cols))
        
        train_out = [np.array(x_train[other_cols], dtype='float32')]
        test_out = [np.array(x_test[other_cols], dtype='float32')]
        if categorical:
            train_out.append(np.array(x_train[cat_cols], dtype='float32'))
            test_out.append(np.array(x_test[cat_cols], dtype='float32'))
        if race:
            train_out.append(np.array(x_train[bool_cols], dtype='float32'))
            test_out.append(np.array(x_test[bool_cols], dtype='float32'))
            
        x_train = np.array(x_train)
        x_test = np.array(x_test)

        # Fit
        if only_race:
            vae.fit([x_train, y_train], train_out, epochs=100, batch_size=None, shuffle=True, validation_split=0.2, verbose=2)
            encoded_data, encoded_mu = encoder.predict(x_test)

        elif control:
            vae.fit([x_train, y_train], train_out, epochs=100, batch_size=None, shuffle=True, validation_split=0.2, verbose=2)
            encoded_data, encoded_mu = encoder.predict([x_test, y_test])
            all_latent_spaces = permute2(all_latent_spaces, x_test, y_test, test_index, encoder, X.columns, latent_type=latent_type)
            
        else:
            vae.fit(x_train, train_out, epochs=100, batch_size=None, shuffle=True, validation_split=0.2, verbose=2)
            encoded_data, encoded_mu = encoder.predict(x_test)
            if permutation:
                all_latent_spaces = permute3(all_latent_spaces, x_test, encoder, test_index, X.columns, latent_type=latent_type)
        
        if only_race:
            # Predict
            decoded_data = decoder.predict(y_test)
        else:
            # Predict
            decoded_data = decoder.predict(encoded_data)
            
        if normalization:
            fold_reconstruct = np.array(reconstruct_data(scaler, decoded_data, other_cols, race, categorical))
        else:
            fold_reconstruct = decoded_data
        
        latent_space = pd.concat([latent_space, pd.DataFrame(encoded_mu, index=test_index)])
        reconstruction = pd.concat([reconstruction, pd.DataFrame(fold_reconstruct, index=test_index)])
        
    if normalization:
        reconstruction.columns = column_ordering
    else:
        reconstruction.columns = ['child_race___2']
    reconstruction = reconstruction.sort_index()
    latent_space = latent_space.sort_index()
    
    # All Latent Spaces Permuted
    all_latent_spaces_permuted = []
    for i in all_latent_spaces:
        all_latent_spaces_permuted.append(i.sort_index())
        
    # Get reconstruction error
    loss = np.sum((og_X - reconstruction) ** 2, axis=1).mean()
    
    return reconstruction, latent_space, loss, vae, all_latent_spaces_permuted

def reconstruct_data(scaler, decoded_data, other_cols, race, categorical):
    count = 0
    
    if not race and not categorical:
        reconstruction_cont = pd.DataFrame(scaler.inverse_transform(decoded_data))
    else:
        reconstruction_cont = pd.DataFrame(scaler.inverse_transform(decoded_data[count]))
    reconstruction_cont.columns = other_cols
    reconstruction = reconstruction_cont
    count += 1

    if categorical:
        insur = pd.DataFrame(decoded_data[count]).iloc[:,0:5]
        edu = pd.DataFrame(decoded_data[count]).iloc[:,5:]
        reconstruction = pd.concat([reconstruction, insur, edu], axis=1)
        count += 1
        
    if race: 
        race = pd.DataFrame(decoded_data[count])
        race.columns = ['race']
        reconstruction = pd.concat([reconstruction, race], axis=1)
        
    return reconstruction

def build_race_model(input_size, output_size, width, depth, kw, okw, hidden=True):
    # Create Inputs
    I = layers.Input(shape=(input_size,))
    
    # Train
    if hidden:
        H = layers.Dense(width, **kw)(I)
        for i in range(depth):
            H = layers.Dense(width, **kw)(H)

        # Compute Output
        output = layers.Dense(output_size, **okw)(H)
        
    else:
        # Compute Output
        output = layers.Dense(output_size, **okw)(I)

    # Build Model
    model = keras.Model(inputs=I, outputs=output)
    
    # Compile the model
    model.compile(optimizer=keras.optimizers.RMSprop(), loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(), 'acc'])
    
    return model

def train_eval(model, x_train, y_train, x_test, y_test, epochs, batch_size):
    # Fit the model
    history = model.fit(x_train, y_train, epochs=epochs, validation_split=0.2, verbose=2, batch_size=batch_size, shuffle=True)
    plot_metric(history, 'loss')
    plot_metric(history, 'mean_squared_error')
    plot_metric(history, 'mean_absolute_error')
    
    # Return predictions
    return model.predict(x_test)

def plot_metric(history, metric):
    font = {'family' : 'DejaVu Sans',
            'weight' : 'normal',
            'size'   : 12}
    matplotlib.rc('font', **font)
    
    plt.plot(history.history[metric])
    plt.plot(history.history['val_%s' % metric])
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def run_race_model(restarts, X, y, model, epochs=100, batch_size=None):
    # 10 K-fold Cross-validation
    kf = KFold(n_splits=5, random_state=111, shuffle=True)

    final_results = pd.DataFrame()
    for i in range(restarts):
        # Train
        outputs = pd.DataFrame()
        for train_index, test_index in kf.split(X, y):

            # Set Scaler
            scaler = StandardScaler()

            # Split Data
            x_train, x_test = X.loc[train_index].reset_index().iloc[:,1:], X.loc[test_index].reset_index().iloc[:,1:]
            y_train, y_test = y.loc[train_index].reset_index().iloc[:,1:], y.loc[test_index].reset_index().iloc[:,1:]

            # Normalize Data
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            # Predict
            output = train_eval(model, x_train, y_train, x_test, y_test, epochs, batch_size)
            outputs = pd.concat([outputs, pd.DataFrame(output, index=test_index)])
            
        # Save Output
        outputs = outputs.sort_index()
        final_results = pd.concat([final_results, outputs])
        
    final_results = final_results.groupby(level=0).mean()    
    
    return final_results

def run_race_model_nocv(restarts, X, y, model, epochs=100, batch_size=None):
   # Set Scaler
    scaler = StandardScaler()

    # Normalize Data
    X = scaler.fit_transform(X)

    # Predict
    output = train_eval(model, X, y, X, y, epochs, batch_size)
    
    return output

def permute(all_latent_spaces, x_adv, x_psych, x_race, test_index, model):

    count = 0
    
    adv_permutes = [['HEI2015_TOTAL_SCORE'], ['ADI'], ['disc_avg'],
                   ['log_income_needs_1t', 'log_income_needs_2t', 'log_income_needs_3t'], 
                   ['insur_1.0', 'insur_2.0', 'insur_3.0', 'insur_4.0', 'insur_5.0'],
                   ['education_intake_-99.0', 'education_intake_1.0', 'education_intake_2.0', 'education_intake_3.0', 'education_intake_4.0']]
    adv_permutes_names = ['Diet', 'ADI', 'Discrimination', 'Income', 'Insurance', 'Education']

    for columns, column_name in zip(adv_permutes, adv_permutes_names):

        temp = copy.deepcopy(x_adv)

        col = column_name
        new_col_names = [col, col]
        temp[columns] = temp[columns].mean()

        latent_space, reconstruction, race_predictions = model.predict([np.array(temp), np.array(x_psych), np.array(x_race)])
        latent_space = pd.DataFrame(latent_space, index=test_index)
        latent_space.columns = new_col_names

        try:
            all_latent_spaces[count] = pd.concat([all_latent_spaces[count], latent_space])
        except:
            all_latent_spaces.append(latent_space)
        count += 1
        
    psych_permutes = [['epds_trim1_total', 'epds_trim2_total', 'epds_trim3_total'], 
                  ['pss_trim1_total', 'pss_trim2_total', 'pss_trim3_total'],
                  ['diffth', 'diffct']]
    psych_permutes_names = ['EPDS', 'PSS', 'Strain']

    for columns, column_name in zip(psych_permutes, psych_permutes_names):

        temp = copy.deepcopy(x_psych)

        col = column_name
        new_col_names = [col, col]
        temp[columns] = temp[columns].mean()

        latent_space, reconstruction, race_predictions = model.predict([np.array(x_adv), np.array(temp), np.array(x_race)])
        latent_space = pd.DataFrame(latent_space, index=test_index)
        latent_space.columns = new_col_names

        try:
            all_latent_spaces[count] = pd.concat([all_latent_spaces[count], latent_space])
        except:
            all_latent_spaces.append(latent_space)
        count += 1

    return all_latent_spaces

def permute2(all_latent_spaces, x_inputs, x_race, test_index, model, columns, latent_type=None):

    count = 0
    x_inputs = pd.DataFrame(x_inputs)
    x_inputs.columns = columns
    
    if latent_type == 'Disadv':
        permutes = [['HEI2015_TOTAL_SCORE'], ['ADI'], ['disc_avg'],
                       ['log_income_needs_1t', 'log_income_needs_2t', 'log_income_needs_3t'], 
                       ['insur_1.0', 'insur_2.0', 'insur_3.0', 'insur_4.0', 'insur_5.0'],
                       ['education_intake_-99.0', 'education_intake_1.0', 'education_intake_2.0', 'education_intake_3.0', 'education_intake_4.0']]
        permutes_names = ['Diet', 'ADI', 'Discrimination', 'Income', 'Insurance', 'Education']

    if latent_type == 'Psych':
        permutes = [['epds_trim1_total', 'epds_trim2_total', 'epds_trim3_total'], 
                      ['pss_trim1_total', 'pss_trim2_total', 'pss_trim3_total'],
                      ['diffth', 'diffct']]
        permutes_names = ['EPDS', 'PSS', 'Strain']
    
    for columns, column_name in zip(permutes, permutes_names):

        temp = copy.deepcopy(x_inputs)

        col = column_name
        new_col_names = [col]
        temp[columns] = temp[columns].mean()

        _, latent_space = model.predict([np.array(temp), np.array(x_race).reshape(-1, 1)])
        
        latent_space = pd.DataFrame(latent_space)
        latent_space = pd.DataFrame(latent_space.iloc[:,0])
        latent_space.index = test_index
        latent_space.columns = new_col_names

        try:
            all_latent_spaces[count] = pd.concat([all_latent_spaces[count], latent_space])
        except:
            all_latent_spaces.append(latent_space)
        count += 1

    return all_latent_spaces

def permute3(all_latent_spaces, x_inputs, model, test_index, columns, latent_type=None):

    count = 0
    x_inputs = pd.DataFrame(x_inputs)
    x_inputs.columns = columns

    if latent_type == 'Disadv':
        permutes = [['HEI2015_TOTAL_SCORE'], ['ADI'], ['disc_avg'],
                       ['log_income_needs_1t', 'log_income_needs_2t', 'log_income_needs_3t'], 
                       ['insur_1.0', 'insur_2.0', 'insur_3.0', 'insur_4.0', 'insur_5.0'],
                       ['education_intake_-99.0', 'education_intake_1.0', 'education_intake_2.0', 'education_intake_3.0', 'education_intake_4.0']]
        permutes_names = ['Diet', 'ADI', 'Discrimination', 'Income', 'Insurance', 'Education']

    if latent_type == 'Psych':
        permutes = [['epds_trim1_total', 'epds_trim2_total', 'epds_trim3_total'], 
                      ['pss_trim1_total', 'pss_trim2_total', 'pss_trim3_total'],
                      ['diffth', 'diffct']]
        permutes_names = ['EPDS', 'PSS', 'Strain']
    
    for columns, column_name in zip(permutes, permutes_names):

        temp = copy.deepcopy(x_inputs)

        col = column_name
        new_col_names = [col]
        temp[columns] = temp[columns].mean()

        _, latent_space = model.predict(np.array(temp))
        latent_space = pd.DataFrame(latent_space, index=test_index)
        latent_space.columns = new_col_names

        try:
            all_latent_spaces[count] = pd.concat([all_latent_spaces[count], latent_space])
        except:
            all_latent_spaces.append(latent_space)
        count += 1

    return all_latent_spaces

def main():
    set_seeds()

if __name__ == "__main__":
    main()