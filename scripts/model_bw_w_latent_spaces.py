import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import keras
import os
import random
import tensorflow as tf
from keras import backend as K
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from keras.models import Model
from keras.constraints import Constraint
from keras.layers import Input, Dense, Concatenate, BatchNormalization, Add
from keras.optimizers import Adam, RMSprop
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats, polyval
from os import listdir
from os.path import isfile, join

DIR = os.environ["PROJECT_DIR"]

def set_seeds():
    # Seed value
    seed_value= 0
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

def load_data(filepath):
    # Load Data
    X = pd.read_csv(filepath)
    y = pd.read_csv(DIR + 'data/child_birthweight.csv')
    return X, y

def normalize_data(scaler, x_train, x_test):
    
    # Get continuous columns
    bool_cols = x_train.filter(regex='race').columns
    if len(bool_cols) >= 1:
        other_cols = set(x_train.columns) - set(bool_cols)
    else:
        other_cols = set(x_train.columns)
    if len(other_cols) >= 1:
        # Standardize continuous values
        x_train_cont = pd.DataFrame(scaler.fit_transform(x_train[other_cols]))
        x_train_cont.columns = other_cols
        x_test_cont = pd.DataFrame(scaler.transform(x_test[other_cols]))
        x_test_cont.columns = other_cols
        # Combine
        x_train = pd.concat([x_train_cont, x_train[bool_cols].reset_index().iloc[:,1:]], axis=1)
        x_test = pd.concat([x_test_cont, x_test[bool_cols].reset_index().iloc[:,1:]], axis=1)  
    return x_train, x_test

def unnormalize_data(scaler, outputs, test_index):
    # Unnormalize
    outputs = pd.DataFrame(scaler.inverse_transform(outputs), index=test_index)
    return outputs

def custom_build_model(input_size, output_size, width, depth, kw, okw, hidden=True):
    # Create Inputs
    I = Input(shape=(input_size,))
    
    # Train
    if hidden:
        H = Dense(width, **kw)(I)
        for i in range(depth):
            H = Dense(width, **kw)(H)

        # Compute Output
        output = Dense(output_size, **okw)(H)
        
    else:
        # Compute Output
        output = Dense(output_size, **okw)(I)

    # Build Model
    model = Model(inputs=I, outputs=output)
    
    # Compile the model
    opt = RMSprop()
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mse', 'mae'])
    
    return model

def train_eval(model, x_train, y_train, x_test, y_test, epochs, batch_size):
    # Fit the model
    history = model.fit(x_train, y_train, epochs=epochs, validation_split=0.2, verbose=2, batch_size=batch_size, shuffle=True)

    return model.predict(x_test)

def run_model(restarts, X, y, model, model_name, filename, epochs=100, batch_size=None):
    
    # Set Seeds
    set_seeds()

    # 10 K-fold Cross-validation
    kf = KFold(n_splits=5, random_state=111, shuffle=True)

    final_results = pd.DataFrame()
    for i in range(restarts):
        # Train
        outputs = pd.DataFrame()
        for train_index, test_index in kf.split(X, y):

            # Set Scaler
            x_scaler = StandardScaler()
            y_scaler = StandardScaler()

            # Split Data
            x_train, x_test = X.loc[train_index].reset_index().iloc[:,1:], X.loc[test_index].reset_index().iloc[:,1:]
            y_train, y_test = y.loc[train_index].reset_index().iloc[:,1:], y.loc[test_index].reset_index().iloc[:,1:]

            # Normalize Data
            x_train, x_test = normalize_data(x_scaler, x_train, x_test)
            x_train = np.array(x_train)
            x_test = np.array(x_test)

            y_train, y_test = normalize_data(y_scaler, y_train, y_test)
            y_train = np.array(y_train)
            y_test = np.array(y_test)

            # Predict
            output = train_eval(model, x_train, y_train, x_test, y_test, epochs, batch_size)
            output = unnormalize_data(y_scaler, output, test_index)
            outputs = pd.concat([outputs, output])
            
        # Save Output
        outputs = outputs.sort_index()
        final_results = pd.concat([final_results, outputs])
        
    final_results = final_results.groupby(level=0).mean()
    final_results.to_csv(filename, sep='\t', index=False)
    
    return final_results

def get_rval_and_pval(x_vals, y_vals):
    xrng = np.array([np.nanmin(x_vals), np.nanmax(x_vals)])
    a_s, b_s = np.polyfit(x_vals, y_vals, 1)
    r, p = stats.pearsonr(x_vals, y_vals)
    xr = polyval([a_s, b_s], xrng)
        
    return r*r, p

def get_rval_data(X, y, lr_outputs, nn_outputs, latent_type, filename, coloring='k', title='Birthweight', save=False):
    columns = ['LR', 'NN']
    
    targets = ['Birthweight']
    
    all_rvals = pd.DataFrame()
    for idx, i in enumerate(y):

        lr_rval, lr_pval = get_rval_and_pval(y.iloc[:,idx], lr_outputs.iloc[:,idx])
        nn_rval, nn_pval = get_rval_and_pval(y.iloc[:,idx], nn_outputs.iloc[:,idx])
        rvals = [lr_rval, nn_rval]

        all_rvals = pd.concat([all_rvals, pd.DataFrame(rvals).T])

        if save:
            font = {'family' : 'DejaVu Sans',
            'weight' : 'normal',
            'size'   : 12}
            matplotlib.rc('font', **font)
            
            fig, ax = plt.subplots(figsize=(5,4), dpi=300)
            
            graph = plt.bar(columns, rvals, color=[coloring, coloring], hatch=[None, '\\\\'])
            plt.xlabel('Model')
            plt.ylabel('$R^2$ Value')
            plt.ylim([0, 0.15])
            
            for p in graph:
                width = p.get_width()
                height = p.get_height()
                x1, y1 = p.get_xy()
                plt.text(x1+width/2,
                         y1+height+0.005,
                         str(np.round(height, 3)),
                         ha='center')

            plt.tight_layout()
            plt.savefig(DIR + 'figure/VAE/%s_%s.pdf' % (latent_type, targets[idx]))
            plt.show()

    all_rvals.columns = ['LR', 'NN']
    all_rvals.index = y.columns
    all_rvals = all_rvals.round(3)
    all_rvals.to_csv(filename, sep='\t', index=False)
    return all_rvals

def perform_holdout_test(rvals, all_data, index, targets, data_type, coloring, title='Birthweight'):
    rvals = rvals.replace(0.00, 0.01)
    
    avg_improvement = []
    for i in all_data:
        temp = i - rvals
        avg_improvement.append(temp/-rvals*100)

    for idx1 in range(len(targets)):
        lr_ordered_avg_improvement = pd.DataFrame([i.iloc[idx1,0] for i in avg_improvement], index=index).sort_values(by=0)
        df = lr_ordered_avg_improvement.reset_index()
        df.columns = ['Name', 'Value']
        
        font = {'family' : 'DejaVu Sans',
            'weight' : 'normal',
            'size'   : 12}
        matplotlib.rc('font', **font)
        
        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(10,4), dpi=300)
        
        color = []
        hatch = []
        for i in df['Name']:
            color.append(coloring)
            hatch.append(None)
                
        df.plot.barh(x='Name', y='Value', rot=0, color=color, ax=ax1, edgecolor='k')
        ax1.axvline(linewidth=1, color='black', linestyle='--')
        ax1.grid(axis = 'x',color = 'black', linestyle = '--', linewidth = 0.5)
        ax1.set_xlabel('Variable Importance\n(Holdout $\Delta R^2$ as %)')
        ax1.get_legend().remove()
        ax1.set_ylabel('')
        bars = ax1.patches
        for bar, h in zip(bars, hatch):
            bar.set_hatch(h)
            
        lr_rvals = df
        lr_rvals.index = lr_rvals['Name']
        lr_rvals = lr_rvals['Value']
        lr_rvals = pd.DataFrame(lr_rvals).T

        nn_ordered_avg_improvement = pd.DataFrame([i.iloc[idx1,1] for i in avg_improvement], index=index).sort_values(by=0) 
        df = nn_ordered_avg_improvement.reset_index()
        df.columns = ['Name', 'Value']
        
        color = []
        hatch = []
        for i in df['Name']:
            color.append(coloring)
            hatch.append('\\\\')
                
        df.plot.barh(x='Name', y='Value', rot=0, color=color, ax=ax2, edgecolor='k')
        ax2.axvline(linewidth=1, color='black', linestyle='--')
        ax2.grid(axis = 'x',color = 'black', linestyle = '--', linewidth = 0.5)
        ax2.set_xlabel('Variable Importance\n(Holdout $\Delta R^2$ as %)')
        ax2.get_legend().remove()
        ax2.set_ylabel('')
        bars = ax2.patches
        for bar, h in zip(bars, hatch):
            bar.set_hatch(h)
            
        nn_rvals = df
        nn_rvals.index = nn_rvals['Name']
        nn_rvals = nn_rvals['Value']
        nn_rvals = pd.DataFrame(nn_rvals).T

        plt.tight_layout()
        plt.savefig(DIR + 'figure/VAE/holdout_test_%s_%s.pdf' % (data_type, targets[idx1]))
        plt.show()
        
        return lr_rvals, nn_rvals
    
def differences_from_baseline(lr_rvals, lr_rvals_comp, nn_rvals, nn_rvals_comp, filename):
    
    df1 = (lr_rvals_comp[lr_rvals.columns.tolist()] - lr_rvals).T
    df1['Name'] = df1.index
    
    df2 = (nn_rvals_comp[nn_rvals.columns.tolist()] - nn_rvals).T
    df2['Name'] = df2.index
    
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(10,4), dpi=300)
    
    color = '#FF6600'
    df1.plot.barh(x='Name', y='Value', rot=0, color=color, ax=ax1, edgecolor='white')
    ax1.axvline(linewidth=1, color='black', linestyle='--')
    ax1.grid(axis = 'x',color = 'black', linestyle = '--', linewidth = 0.5)
    ax1.set_xlabel('Delta from Baseline')
    ax1.get_legend().remove()
    ax1.set_ylabel('')

    color = '#6D40BD'
    df2.plot.barh(x='Name', y='Value', rot=0, color=color, ax=ax2, edgecolor='white')
    ax2.axvline(linewidth=1, color='black', linestyle='--')
    ax2.grid(axis = 'x',color = 'black', linestyle = '--', linewidth = 0.5)
    ax2.set_xlabel('Delta From Baseline')
    ax2.get_legend().remove()
    ax2.set_ylabel('')
    
    plt.tight_layout()
    plt.savefig(DIR + 'figure/VAE/%s.pdf' % (filename))
    plt.show()
    
def model_birthweight(baseline, directory, latent_type, color='k'):
    # Set Parameters
    X, y = load_data()
    input_size = 2
    output_size = 1
    width = 0
    depth = 0
    kw = {'activation':'linear', 'kernel_initializer':tf.random_uniform_initializer(), 'kernel_regularizer':keras.regularizers.l2()}
    okw = {'activation':'linear', 'kernel_initializer':tf.random_uniform_initializer(), 'kernel_regularizer':keras.regularizers.l2()}
    epochs = 100
    batch_size = None 
    hidden = False
    restarts = 5
    
    X = pd.read_csv(baseline, sep='\t')
    X.columns = ['adv', 'psych']
    
    # LR
    filename = directory + 'LR/LR_outputs.csv'
    if os.path.exists(filename):
        lr_outputs = pd.read_csv(filename, sep='\t')
    else:
        lr_model = custom_build_model(input_size, output_size, width, depth, kw, okw, hidden=hidden)        
        lr_outputs = run_model(restarts, X, y, lr_model, 'LR', filename, epochs, batch_size)
        
    # Set Parameters
    X, y = load_data()
    input_size = 2
    output_size = 1
    width = 2
    depth = 0
    kw = {'activation':'elu', 'kernel_initializer':tf.random_uniform_initializer(), 'kernel_regularizer':keras.regularizers.l2()}
    okw = {'activation':'linear', 'kernel_initializer':tf.random_uniform_initializer(), 'kernel_regularizer':keras.regularizers.l2()}
    epochs = 100
    batch_size = None 
    hidden = True
    restarts = 5
    
    X = pd.read_csv(baseline, sep='\t')
    X.columns = ['adv', 'psych']

    # NN
    filename = directory + 'NN/NN_outputs.csv'
    if os.path.exists(filename):
        nn_outputs = pd.read_csv(filename, sep='\t')
    else:
        nn_model = custom_build_model(input_size, output_size, width, depth, kw, okw, hidden=hidden)
        nn_outputs = run_model(restarts, X, y, nn_model, 'NN', filename, epochs, batch_size)  
    
    # Rvals
    filename = directory + 'rvals/rvals.csv'
    if os.path.exists(filename):
        rvals = pd.read_csv(filename, sep='\t')
    else:
        rvals = get_rval_data(X, y, lr_outputs, nn_outputs, latent_type, filename, coloring=color, save=True) 
    
    # Permuted
    onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f)) and 'permuted' in f]
    all_data = []
    index = []
    for i in onlyfiles:
        
        # Set Parameters
        X, y = load_data()
        input_size = 2
        output_size = 1
        width = 0
        depth = 0
        kw = {'activation':'linear', 'kernel_initializer':tf.random_uniform_initializer(), 'kernel_regularizer':keras.regularizers.l2()}
        okw = {'activation':'linear', 'kernel_initializer':tf.random_uniform_initializer(), 'kernel_regularizer':keras.regularizers.l2()}
        epochs = 100
        batch_size = None 
        hidden = False
        restarts = 5
        
        X = pd.read_csv(directory + i, sep='\t')
        X.columns = ['adv', 'psych']
        
        # LR
        filename = directory + 'LR/LR_' + i
        if os.path.exists(filename):
            lr_permuted_outputs = pd.read_csv(filename, sep='\t')
        else:
            lr_permuted_model = custom_build_model(input_size, output_size, width, depth, kw, okw, hidden=hidden)        
            lr_permuted_outputs = run_model(restarts, X, y, lr_permuted_model, 'LR', filename, epochs, batch_size)
        
        # Set Parameters
        X, y = load_data()
        input_size = 2
        output_size = 1
        width = 2
        depth = 0
        kw = {'activation':'elu', 'kernel_initializer':tf.random_uniform_initializer(), 'kernel_regularizer':keras.regularizers.l2()}
        okw = {'activation':'linear', 'kernel_initializer':tf.random_uniform_initializer(), 'kernel_regularizer':keras.regularizers.l2()}
        epochs = 100
        batch_size = None 
        hidden = True
        restarts = 5
        
        X = pd.read_csv(directory + i, sep='\t')
        X.columns = ['adv', 'psych']
        
        # NN
        filename = directory + 'NN/NN_' + i
        if os.path.exists(filename):
            nn_permuted_outputs = pd.read_csv(filename, sep='\t')
        else:
            nn_permuted_model = custom_build_model(input_size, output_size, width, depth, kw, okw, hidden=hidden)
            nn_permuted_outputs = run_model(restarts, X, y, nn_permuted_model, 'NN', filename, epochs, batch_size)
    
        # Rvals
        filename = directory + 'rvals/rvals_' + i
        if os.path.exists(filename):
            rvals_permuted = pd.read_csv(filename, sep='\t')
        else:
            rvals_permuted = get_rval_data(X, y, lr_permuted_outputs, nn_permuted_outputs, latent_type, filename, save=False)
        all_data.append(rvals_permuted)
        index.append(i.split('_')[-1].split('.')[0])
        
    targets = ['Birthweight']

    lr_rvals, nn_rvals = perform_holdout_test(rvals, all_data, index, targets, latent_type, color)