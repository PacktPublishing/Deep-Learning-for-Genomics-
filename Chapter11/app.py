import streamlit as st
import numpy as np
from utils import dnaseq_features
from keras.models import load_model

st.set_page_config(page_title = "A Simple App for predicting TFBS in a DNA sequence")

model = load_model('best_model.h5')

with st.container():
    st.title('Simple Model Serving Web App for TFBS prediction')
    st.caption('Get TFBS Predictions From The Latest Model.')

# Create a horizontal line, and then a new container.
st.markdown("---")

with st.container():

    dna_seq = st.text_area("Input DNA sequence", 'ATAGAGAC...')

    dna_ohe_feat, ds_index, ds_val = dnaseq_features(seq=dna_seq)

    trigger = st.button('Make Prediction')

    if trigger:

        st.info("Loading the data for predictions")

        predicted_labels = model.predict(dna_ohe_feat)

        print(predicted_labels)
        print(ds_val)

        for i, j in zip(ds_val, predicted_labels):
            st.write(i)
            if np.argmax(j) == 1:
                st.success("TFBS found :thumbsup:")
            else:
                st.error('TFBS not found :thumbsdown:')