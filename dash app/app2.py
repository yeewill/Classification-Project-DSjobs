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
from sklearn.metrics import confusion_matrix
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
    html.H1(children='Threshold Cost Estimator'),
    html.Div(children ='Input threshold level for Ensemble model'),
    dcc.Input(id='threshold', value='input threshold', type='text'),
    html.Div(children ='Input cost of Data Scientist Acquisition'),
    dcc.Input(id='FP', value='USD', type='text'),
    html.Div(children ='Input cost of Employee Retention Program'),
    dcc.Input(id='FN', value='USD', type='text'),
    html.Div(id='new-cost')
])


@app.callback(
    Output(component_id='new-cost', component_property='children'),
    [
    Input(component_id='threshold', component_property='value'),
    Input(component_id='FP', component_property ='value'),
    Input(component_id='FN', component_property ='value')
    ]
)
def update_output_div(thold, fp_cost, fn_cost):
    thold = float(thold)
    fp_cost = float(fp_cost)
    fn_cost = float(fn_cost)
    
    y_pred = predict_proba_output(vote_soft.predict_proba(X_test),thold)
    
    
    
    cost_reten = int(confusion_matrix(y_test,y_pred)[1][0] * fn_cost)
    cost_hire = int(confusion_matrix(y_test,y_pred)[0][1] * fp_cost)
    return 'Your cost of Employee Retention is : {} Your cost of Data Science Acqusition is : {}'.format(cost_reten,cost_hire)


if __name__ == '__main__':
    app.run_server(debug=True)