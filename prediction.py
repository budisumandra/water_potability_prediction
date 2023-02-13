import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
import pandas as pd

with open('water_potability.bin', 'rb') as f_in:
    dv, rf = pickle.load(f_in)


def predict_single(potability_parameters):
    X = dv.transform([potability_parameters])
    y_pred = rf.predict_proba(X)[:, 1]
    return y_pred[0]


def predict_water_potability(potability_parameters):
    prediction = predict_single(potability_parameters)
    potability = prediction >= 0.5
    result = {
        'Potability Probability': float(prediction),
        'Is Potable?': 'Potable' if potability else 'Not Potable'}
    return pd.DataFrame(result, index=[0])
