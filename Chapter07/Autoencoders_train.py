#!/usr/bin/env python
# coding: utf-8

# # Pan-cancer gene prediction using Autoencoders

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import Input, Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split

# Load RNAseq data
pancan_rnaseq_df = pd.read_csv('data/pancan_scaled_zeroone_rnaseq.tsv', index_col=0, sep="\t", low_memory=False)
pancan_rnaseq_df.iloc[:2,:5].head()

# Train Test split
pancan_rnaseq_df_train, pancan_rnaseq_df_test = train_test_split(pancan_rnaseq_df, test_size=0.1, shuffle=True)
pancan_rnaseq_df_train.shape, pancan_rnaseq_df_test.shape

# This is the size of our encoded representations
encoding_dim = 100
numb_of_features = pancan_rnaseq_df.shape[1]

# Defining the 'Autoencoder' full model
autoencoder = Sequential()
autoencoder.add(Dense(encoding_dim, activation="relu", input_shape=(numb_of_features, )))
autoencoder.add(Dense(numb_of_features, activation="sigmoid"))
autoencoder.compile(optimizer="adam", loss='mse')
autoencoder.summary()

%%time
hist = autoencoder.fit(np.array(pancan_rnaseq_df_train), np.array(pancan_rnaseq_df_train),
                       shuffle=True,
                       epochs=10,
                       batch_size=50,
                       validation_split=0.2)

# Visualize training performance
history_df = pd.DataFrame(hist.history)
ax = history_df.plot()
ax.set_xlabel('Epochs')
ax.set_ylabel('Reconstruction Loss')


# Reconstruction
input_sample = pancan_rnaseq_df_test[:1]

reconstruction = autoencoder.predict([input_sample])
reconstruction

# Reconstruct input RNAseq
reconstruction = autoencoder.predict(np.array(pancan_rnaseq_df))

reconstructed_df = pd.DataFrame(reconstruction, index=pancan_rnaseq_df.index,
                                columns=pancan_rnaseq_df.columns)

reconstruction_fidelity = reconstructed_df - pancan_rnaseq_df

gene_mean = reconstruction_fidelity.mean(axis=0)
gene_abssum = reconstruction_fidelity.abs().sum(axis=0).divide(pancan_rnaseq_df.shape[0])
gene_summary = pd.DataFrame([gene_mean, gene_abssum], index=['gene mean', 'gene abs(sum)']).T
gene_summary.sort_values(by='gene abs(sum)', ascending=False).head()

# Mean of gene reconstruction vs. absolute reconstructed difference per sample
g = sns.jointplot('gene mean', 'gene abs(sum)', data=gene_summary)