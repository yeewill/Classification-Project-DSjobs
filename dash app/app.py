# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import pandas as pd
import copy

from sklearn.metrics import roc_curve , roc_auc_score , precision_recall_curve, confusion_matrix , f1_score
from collections import Counter


def predict_proba_output(X_val_proba, new_threshold):
    empty = []
    for values in X_val_proba[:,0]:
        if values <= new_threshold:
            empty.append(0)
        if values > new_threshold:
            empty.append(1)
    return(empty)

X_test = pickle.load(open('X_test_processed.p','rb'))
y_test = pickle.load(open('y_test.p','rb'))
vote_soft = pickle.load(open('voting_classifer_soft.p','rb'))

hola = str(Counter(vote_soft.predict(X_test)))


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Input(id='my-id', value='input threshold', type='text'),
    html.Div(id='my-div')
])


@app.callback(
    Output(component_id='my-div', component_property='children'),
    [Input(component_id='my-id', component_property='value')]
)
def update_output_div(input_value):
    input = float(input_value)
    hello =f1_score(y_test,predict_proba_output(vote_soft.predict_proba(X_test),input))
    return 'Your f1_score is : {}'.format(hello)


if __name__ == '__main__':
    app.run_server(debug=True)