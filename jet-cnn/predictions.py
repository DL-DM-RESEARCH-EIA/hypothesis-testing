#Purpose: This is a CNN classification code for the jet-images data set
#Original Source: Charanjit K. Khosa, University of Genova, Italy
#Modified By: Charanjit K. Khosa, University of Genova, Italy, Michael Soughton, University of Sussex, UK
#Date: 09.02.2021
# import sys, os
# import numpy as np
# from numpy import expand_dims

# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D

# data_dir = 'Data/'

# def pad_image(image, max_size = (25,25)):
#     """
#     Simply pad an image with zeros up to max_size.
#     """
#     size = np.shape(image)
#     px, py = (max_size[0]-size[0]), (max_size[1]-size[1])
#     a1=int(np.floor(px/2.0))
#     a2=int(np.ceil(px/2.0))
#     a3=int(np.floor(py/2.0))
#     a4=int(np.ceil(py/2.0))
#     image = np.pad(image, ((a1, a2), (a3, a4)), 'constant', constant_values=(0))
#     #image = np.pad(image, (map(int,((np.floor(px/2.), np.ceil(px/2.)))), map(int,(np.floor(py/2.), np.ceil(py/2.)))), 'constant')
#     return image

# def normalize(histo, multi=255):
#     """
#     Normalize picture in [0,multi] range, with integer steps. E.g. multi=255 for 256 steps.
#     """
#     return (histo/np.max(histo)*multi).astype(int)

# #Loading input data
# data0 = np.load(data_dir + 'qcd_leading_jet.npz',allow_pickle=True,encoding = 'latin1')['arr_0']
# data1 = np.load(data_dir + 'top_leading_jet.npz',allow_pickle=True,encoding = 'latin1')['arr_0']

# print("data0",data0.shape)
# print("data1",data1.shape)

# #I want to use 60K events from each sample (total-x=30K)
# data0 = np.delete(data0,np.s_[1:714],0)
# data1 = np.delete(data1,np.s_[1:1762],0)

# print("data0",data0.shape)
# print('We have {} QCD jets and {} top jets'.format(len(data0), len(data1)))

# # objects and labels
# x_data = np.concatenate((data0, data1))
# y_data = np.array([0]*len(data0)+[1]*len(data1))


# print("xdatashape",x_data.shape)

# # pad and normalize images
# x_data = list(map(pad_image, x_data))
# #print("xdatashape",x_data.shape)
# x_data = list(map(normalize, x_data))
# print("xdatashape-afterNorm",x_data[1][17:21][:])


# # shapeuffle
# np.random.seed(4) # for reproducibility
# x_data, y_data = np.random.permutation(np.array([x_data, y_data]).T).T

# # the data coming out of previous commands is a list of 2D arrays. We want a 3D np array (n_events, xpixels, ypixels)
# x_data = np.stack(x_data)
# #y_data= np.stack(y_data)

# print("xshape-after stack",x_data.shape)
# #print("x-after stack",x_data)

# x_data=x_data /255.
# x_data = expand_dims(x_data, axis=3)
# #print("xdatashape-afterNorm255",x_data[1][0][10:21][:])


# #y_data = keras.utils.to_categorical(y_data, 2)

# n_train = 80000
# (x_train, x_test) = x_data[:n_train], x_data[n_train:]
# (y_train, y_test) = y_data[:n_train], y_data[n_train:]

# print("x_train",x_train.shape)
# ytestone=y_test

# y_train = keras.utils.to_categorical(y_train, 2)
# y_test = keras.utils.to_categorical(y_test, 2)


import warnings
import matplotlib.pyplot as plt
import catboost as ctb
import numpy as np
from sklearn.model_selection import train_test_split

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], "size":18})
rc('text', usetex=True)


model_dir='model_catboost/'

#history_cnn = np.load(model_dir+'training_histories.npz')['arr_0']
model_catboost = ctb.CatBoostClassifier()
model_catboost.load_model(model_dir+"cnn_bootstrap000.h5")

x_data = np.load("prepped_x_data.npy")
y_data = np.load("prepped_y_data.npy")

test_size = 1/3

x_train, x_test, y_train, y_test = train_test_split(np.squeeze(x_data), np.squeeze(y_data), test_size=test_size)
ytestone=np.argmax(y_test, axis=1)

predictions_cnn = model_catboost.predict(x_test)

#print("predictions_cnn",predictions_cnn[10])

from sklearn.metrics import roc_curve

fpr_cnn, tpr_cnn, thresholds = roc_curve(np.argmax(y_test, axis=1), predictions_cnn)

from sklearn.metrics import auc

auc = auc(fpr_cnn, tpr_cnn)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_cnn, tpr_cnn, label='(AUC = {:.3f})'.format(auc))
plt.gca().set(xlabel='False positive rate', ylabel='True positive rate', title='ROC curve', xlim=(-0.01,1.01), ylim=(-0.01,1.01))
plt.grid(True, which="both")
plt.legend(loc='lower right');
plt.savefig('ROC_curve.png')


y_top=predictions_cnn

print("y_top",y_top)
print("y_test",y_test)

#ytestnew= y_test.flatten()

print("ytestone",ytestone)
print("y_top.shape",y_top.shape)

top_probs = y_top[np.where(ytestone == 1)]
qcd_probs = y_top[np.where(ytestone == 0)]


np.savetxt("top_probs005.txt",top_probs)
np.savetxt("qcd_probs005.txt",qcd_probs)

print("top_probs",top_probs)


import seaborn as sns; sns.set(style="white", color_codes=True)
# Make KDE plot
fig, ax = plt.subplots(figsize=(8, 8))
#ax = plt.gca()
susy_pdf_plot = sns.kdeplot(top_probs,label="BSM")
other_sig_pdf_plot = sns.kdeplot(qcd_probs,label="SM")
# Set title with fontsize 18
ax.set_title("BSM vs SM", fontsize=18)

# Set xlabel and    ylabel
ax.set_xlabel(r"P(BSM)", fontsize=16)
ax.set_ylabel("PDF", fontsize=16)

# Hide the y ticks
ax.set_yticks([])

# Set legend with fontsize 14
ax.legend(fontsize=16)

# Set tick parameters for both axes
ax.tick_params(axis='x', labelsize=16)  # Set x ticks fontsize to 12
# ax.tick_params(axis='y', labelsize=12)  # Uncomment if y ticks are present
plt.savefig("PDF_topqcd1.pdf")

import pandas as pd
path_to_xs="cross_section/data"

columns1 = ["cross", "delta-xs", "mzp"]
columns2 = ["cross", "delta-xs"]

xs_signal = pd.read_csv(f"{path_to_xs}/data_signal_cln_kl_2.txt", names=columns1)
xs_bkg = pd.read_csv(f"{path_to_xs}/data_background_cln.txt", names=columns2)

signal_xs_df = xs_signal[xs_signal["mzp"] == 1500]
# sample from a poisson distribution sigma on  uncertainty
xs_signal = float(signal_xs_df["cross"])
# Bckg is a normal distribution
xs_bkg = float(xs_bkg["cross"])

top_cross_section = xs_signal # 53.1684
qcd_cross_section = xs_bkg # 48829.2
top_to_qcd_ratio = top_cross_section/qcd_cross_section

import random
qcd_probs = np.array(list(qcd_probs) * 100)
top_probs = random.choices(np.array(list(top_probs) * 100), k=int(len(qcd_probs) * top_to_qcd_ratio))
fig, ax = plt.subplots(figsize=(8, 8))
#ax = plt.gca()
susy_pdf_plot = sns.kdeplot(top_probs,label="BSM")
other_sig_pdf_plot = sns.kdeplot(qcd_probs,label="SM")
ax.set_title("BSM vs SM")
ax.set_xlabel(r"P(BSM)")
ax.set_ylabel("PDF")
ax.set_yticks([])
ax.legend()
plt.savefig("PDF_topqcd2.pdf")
