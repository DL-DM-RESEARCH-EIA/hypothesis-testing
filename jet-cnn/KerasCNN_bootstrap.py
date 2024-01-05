#!/usr/bin/env python3

#Purpose: This is a CNN classification code for the jet-images data set
#Original Source: Taken from https://gist.github.com/ilmonteux
#Modified By: Charanjit K. Khosa, University of Genova, Italy, Michael Soughton, University of Sussex, UK
#Date: 09.02.2021
import sys, os
import numpy as np
from numpy import expand_dims
import pandas as pd
import catboost as ctb

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from main_NNDM import *


data_dir = 'Data/'
# =========================== Take in arguments ================================
import argparse


import pandas as pd
import numpy as np
import keras
from numpy import expand_dims

set_name = ['DUNE']

# This is how the program will develop

# In prepare data, set_name == "HEPMASS", proceed as already established
# If set_name=="DUNE", then create the data to save on Data/MLTrainData/TrainData/
#  based on the dict constrained.


def prepare_data(file_path, dict_contrained, signal_label_threshold=1e-1, set_name='HEPMASS'):
    if set_name == 'HEPMASS':
        mass_value = dict_constrained['mass_value']
        prepare_data_HEPMASS(file_path, mass_value, signal_label_threshold=signal_label_threshold)     
    elif set_name == 'DUNE':
        prepare_data_DUNE(file_path, dict_constrained, signal_label_threshold=signal_label_threshold)

def prepare_data_DUNE(file_path, dict_constrained, signal_label_threshold=1e-1):
    print("Preparing data to be written")

    n_data = 1e6  # Set the size of your dataset

    # Create balanced training and validation datasets
    df_train = create_dataset_dataframe_balanced(file_path, dict_constrained, n_data, dataset_type='df_train')
    df_val = create_dataset_dataframe_balanced(file_path, dict_constrained, int(n_data * 0.2), dataset_type='df_val')

    # Here, apply any additional processing you need, for example:
    # Filtering, feature engineering, scaling, etc.

    # Save the prepared datasets
    base_dir = "hypothesis-testing/jet-cnn/Data"
    target_dir_train = os.path.join(base_dir, "MLTrainData", "TrainData")

    # Create the directories if they don't exist
    os.makedirs(target_dir_train, exist_ok=True)

    # Save the dataframes
    save_dataframe(df_train, "train", dict_constrained, target_dir_train)
    save_dataframe(df_val, "val", dict_constrained, target_dir_train)

    return df_train, df_val


def prepare_data_HEPMASS(file_path, mass_value, signal_label_threshold=1e-1):
    print("Preparing data")

    # Load and preprocess data
    data = pd.read_csv(file_path)
    data = data.astype({'# label': 'int32'})
    data.rename(columns={"# label": "label"}, inplace=True)
    background = data.loc[data.label == 0] 
    signal = data.loc[data.label == 1]
    selected_signal = signal.loc[abs(signal.mass - mass_value) < signal_label_threshold]

    values = ['f20', 'f21', 'f22', 'f23', 'f24', 'f25', 'f26']

    # Loading input data
    data0 = background[values].to_numpy()
    data1 = selected_signal[values].to_numpy()

    print("data0", data0.shape)
    print("data1", data1.shape)

    # Balancing the dataset
    size_signal = min(len(data1), len(data0))
    data0 = data0[:size_signal]
    data1 = data1[:size_signal]

    print("data0", data0.shape)
    print(f'We have {len(data0)} background events and {len(data1)} signal m={mass_value} events')

    # Preparing objects and labels
    x_data = np.concatenate((data0, data1))
    y_data = np.array([0]*len(data0) + [1]*len(data1))

    print("x_data shape", x_data.shape)
    print("x_data shape before reshuffle", len(x_data))

    # Shuffling the data
    np.random.seed(0)  # for reproducibility
    perm = np.random.permutation(y_data.shape[0])
    x_data, y_data = x_data[perm], y_data[perm]

    # Reshaping
    x_data = np.stack(x_data)
    x_data = expand_dims(x_data, axis=2)

    # One-hot encoding
    y_data = keras.utils.to_categorical(y_data, 2)

    # Saving preprocessed data
    np.save("prepped_x_data", x_data)
    np.save("prepped_y_data", y_data)

    print("Data preparation complete.")

parser = argparse.ArgumentParser(description='These are the arguments that will be passed to the script')

parser.add_argument("--smear_target",
                    type=str,
                    default="neither",
                    help="str: The jet image type that is to be smeared. Either 'neither', 'top', 'qcd', 'both'. Default is 'neither'")

parser.add_argument("--sigma",
                    type=float,
                    default=0,
                    help="float: The sigma value for smearing. Default is 0.")

parser.add_argument("--n_iter",
                    type=int,
                    default=5,
                    help="int: The number of bootstrap iterations. Default is 5.")

parser.add_argument("--n_epoch",
                    type=int,
                    default=18,
                    help="int: The number of training iterations. Default is 18.")

args = parser.parse_args()

smearing = args.sigma
n_iterations = args.n_iter
n_epochs = args.n_epoch
print("Smear target = " + str(args.smear_target) + " sigma = " + str(smearing) + " n iterations = " + str(n_iterations))

# ==============================================================================

from skimage import filters

# Preparing the data is very memory intensive and won't run on the sussex cluster in a batch so prepare it beforehand
prep_data = "load"
if prep_data == "prepare":
    # Usage

    if set_name[0] == 'HEPMASS': 
        dict_constrained = {'mass_value': 1250}
        path = 'Data/all_train.csv'
        prepare_data('Data/all_train.csv', dict_constrained)
    elif set_name[0] == 'DUNE':
        # TODO make all the process to create the df separately, combined, then sampled
        # DF of the data is assumed to aredy exist
        file_path = "hypothesis-testing/jet-cnn/Data/MLTrainData/TrainDataDist/"
        dict_constrained = {"mk1": 0.02514990210703923, "delta": 1.05,  "eps2": 1.4550810518824755e-08} # input
        prepare_data_DUNE(file_path, dict_constrained)

elif prep_data == "load":
    print("Loading data")
    x_data, y_data = None, None
    df_train, df_val = None, None
    if set_name[0] == 'HEPMASS': 
        x_data = np.load("prepped_x_data.npy")
        y_data = np.load("prepped_y_data.npy")
    elif set_name[0] == 'DUNE':
        base_dir = "hypothesis-testing/jet-cnn/Data/MLTrainData/"
        target_dir_train = "TrainData"

        file_path_train = os.path.join(base_dir, "TrainData")
        dict_constrained = {"mk1": 0.02514990210703923, "delta": 1.05,  "eps2": 1.4550810518824755e-08} # input

        all_files_train = read_names_constrained(file_path_train, dict_constrained, ext='.pkl')

        # Extract specific file paths
        df_train_path = find_string_with_substring(all_files_train, 'train')
        df_val_path = find_string_with_substring(all_files_train, 'val')

        # Check if the files exist and then load them
        if df_train_path and os.path.exists(df_train_path):
            df_train = pd.read_pickle(df_train_path)
        else:
            raise FileNotFoundError("Training data file not found.")

        if df_val_path and os.path.exists(df_val_path):
            df_val = pd.read_pickle(df_val_path)
        else:
            raise FileNotFoundError("Validation data file not found.")

test_size = 1/3

def create_model():
    model_CBC = ctb.CatBoostClassifier(custom_loss=['Logloss', 'Accuracy'], metric_period=1)
    return model_CBC


# Do the bootstrap
from sklearn.metrics import accuracy_score

y_test_list = []
predictions_list = []
score_list = []

for i in range(n_iterations):
    model_catboost = ctb.CatBoostClassifier(custom_loss=['Logloss', 'Accuracy'], metric_period=1)
    print("bootstrap iteration", i+1)

    # TODO 4: change the way this is read depending on the case
    if set_name[0] == 'HEPMASS':
        x_train, x_test, y_train, y_test = train_test_split(np.squeeze(x_data), np.squeeze(y_data), test_size=test_size)
    if set_name[0] == 'DUNE':
        y_train = df_train['label']
        x_train = df_train.drop('label', axis=1)

        y_test = df_val['label']
        x_test = df_val.drop('label', axis=1)


    # This is going back and forth, but it is made like this for the mean time for compatibility
    #   for the code already written for HEPMASS
    y_train = keras.utils.to_categorical(y_train, 2)
    y_test = keras.utils.to_categorical(y_test, 2)

    # correct size of data
    train_data = ctb.Pool(data=x_train, label=np.argmax(y_train, axis=1))
    valid_data = ctb.Pool(data=x_test, label=np.argmax(y_test, axis=1))

    qcd_train_indices = np.where(y_train[:,0]==1)[0]
    top_train_indices = np.where(y_train[:,1]==1)[0]
    both_train_indices = np.where(y_train[:,0]>=0)[0]

    qcd_test_indices = np.where(y_test[:,0]==1)[0]
    top_test_indices = np.where(y_test[:,1]==1)[0]
    both_test_indices = np.where(y_test[:,0]>=0)[0]

    # If jet type to smear is qcd, top or both, smear them appropriately
    if args.smear_target != "neither":
        if args.smear_target == "qcd":
            # TODO: chosen index to smear
            pass
        elif args.smear_target == "top":
            # TODO: chosen index to smear
            pass
        elif args.smear_target == "both":
            # TODO: chosen index to smear
            pass
        # TODO: smeaning application
    elif args.smear_target == "neither":
        print("Not smearing either")

    model_catboost.fit(
        train_data,
        eval_set=valid_data,
        verbose=False,
        early_stopping_rounds=100
    )

    predictions_cnn = model_catboost.predict(x_test)
    y_test_list.append(y_test)
    predictions_list.append(predictions_cnn)

    score = accuracy_score(np.argmax(y_test, axis=1), np.round(predictions_cnn))
    score_list.append(score)

    model_dir='model_catboost/'
    if not os.path.isdir(model_dir): os.system('mkdir '+model_dir)
    model_catboost.save_model(f'{model_dir}cnn_bootstrap00{i}.h5')
    np.savez(model_dir+f'training_history00{i}.npz', [model_catboost.get_evals_result()])

    # Clear model and memory
    from keras import backend as K
    import gc
    del model_catboost
    K.clear_session()
    gc.collect()
    print("Cleared session and memory")


y_test_arr = np.stack((y_test_list))
predictions_arr = np.stack((predictions_list))
score_arr = np.stack((score_list))

array_dir = 'bootstrap_arrays/'
# extension = str(args.smear_target) + '_' + str(smearing) + 'smeared_' + str(n_iterations) + '_bootstraps'
extension = str(n_iterations) + '_bootstraps'
os.makedirs(array_dir, exist_ok=True)

np.save(array_dir + 'y_test_arr' + extension, y_test_arr)
np.save(array_dir + 'predictions_arr' + extension, predictions_arr)
np.save(array_dir + 'score_arr' + extension, score_arr)

# To load
y_test_arr = np.load(array_dir + 'y_test_arr' + extension + '.npy')
predictions_arr = np.load(array_dir + 'predictions_arr' + extension + '.npy')
score_arr = np.load(array_dir + 'score_arr' + extension + '.npy')


"""
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('Accuracy005.pdf')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('Loss005.pdf')
plt.show()


#Save the model and training history
"""
