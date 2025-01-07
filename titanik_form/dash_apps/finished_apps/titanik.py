from django_plotly_dash import DjangoDash
import numpy as np
import pandas as pd
from dash import Dash, callback, Input, Output, dcc, html

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import  sys
import os
from django.conf import settings


os.environ["OPENBLAS_NUM_THREADS"] = "1"




print(sys.prefix)
print(sys.base_prefix)
ages = []
for age in range(1, 101):
    d = {}
    d['label'] = age
    d['value'] = age
    ages.append(d)

labelStyle = {
    'cursor': 'pointer',
    'margin-right': '10px',
    'margin-top':'5px',
    'font-family': 'Calibri',
    'font-size': '1.2em',
}
style_dd = {
    'cursor': 'pointer',
    'font-family': 'Calibri',
    'font-size': '1.2em', }
inputStyle = {"width": "20px",
              "height": "20px",
              "margin-right": "10px",
              'background-color': 'pink !important',
              'border-color': 'pink !important',
              }
survive_css = {'font-family': 'Calibri',
               'color': 'pink',
               'font-size': '6em',
               'line-height': 100}
question_css = {'font-family': 'Calibri',
                'font-size': '2em',
                'font-weight': 'bold',
                'color': 'turquoise',
}
file_path = os.path.join(settings.BASE_DIR, 'titanik_form', 'dash_apps', 'finished_apps', 'train.csv')
data = pd.read_csv(file_path)

dummy = pd.get_dummies(data[['Embarked', 'Sex']])
data_dummy = pd.concat([data, dummy], axis=1)
data_dummy.drop(['Embarked', 'Sex'], axis=1, inplace=True)
features = data_dummy[['Sex_female', 'Pclass', 'SibSp', 'Parch', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Age']]
features['Age'] = round(features.Age.replace(np.nan, features.Age.mean()))
features.Age = features.Age.astype(int)

file_path = os.path.join(settings.BASE_DIR, 'titanik_form', 'dash_apps', 'finished_apps', 'Survived.csv')
survived = pd.read_csv(file_path)

print(features.shape, survived.shape)

rf_classifier = RandomForestClassifier(n_estimators=500, max_depth=5, min_samples_split=4)
rf_classifier.fit(features, survived)

app = DjangoDash('titanik') # Create app

app.layout = html.Div([
    html.H2("Choose your gender.", style = question_css),
    dcc.RadioItems(id='sex',
                   options=[{'label': 'male', 'value': 0},
                            {'label': 'female', 'value': 1}],
                   inline=True, labelStyle=labelStyle,
                   inputStyle=inputStyle, className='custom-radio'),
    html.Br(),

    html.H2("Which class are you travelling?", style = question_css),
    dcc.RadioItems(id='class',
                   options=[{'label': '1st class', 'value': 1},
                            {'label': '2nd class', 'value': 2},
                            {'label': '3rd class', 'value': 3}],
                   inline=True, labelStyle=labelStyle, inputStyle=inputStyle, className='custom-radio'),
    html.Br(),

    html.H2("Do you have sibling or spouse aboard the Titanik?", style = question_css),
    dcc.RadioItems(id='spouse',
                   options=[{'label': 'yes', 'value': 1},
                            {'label': 'no', 'value': 0},
                            ],
                   inline=True, labelStyle=labelStyle, inputStyle=inputStyle, className='custom-radio'),
    html.Br(),

    html.H2("Do you have parents aboard the Titanik?", style = question_css),
    dcc.RadioItems(id='parents', options=[{'label': 'yes', 'value': 1},
                                          {'label': 'no', 'value': 0},
                            ],
                   inline=True, labelStyle=labelStyle, inputStyle=inputStyle, className='custom-radio'),
    html.Br(),

    html.H2("At what city have you boarded in?", style = question_css),
    dcc.RadioItems(id='port', options=[{'label': 'Cherbourg', 'value': '100'},
                            {'label': 'Queenstown', 'value': '010'},
                            {'label': 'Southampton', 'value': '001'}],
                   inline=True, labelStyle=labelStyle, inputStyle=inputStyle, className='custom-radio'),
    html.Br(),

    dcc.Dropdown(id='age', options=ages, placeholder='Your age', style=style_dd),
    html.Br(),

    html.H1(id='survived', style=survive_css)
], style={'width': '80%'})

@app.callback(
        Output(component_id='survived', component_property='children'),
        [Input(component_id='sex', component_property='value'),
         Input(component_id='class', component_property='value'),
         Input(component_id='spouse', component_property='value'),
         Input(component_id='parents', component_property='value'),
         Input(component_id='port', component_property='value'),
         Input(component_id='age', component_property='value'),]
          )
def viz(sex, ticket_class, spouse, parents, port, age):
    x = pd.DataFrame([sex, ticket_class, spouse, parents, port[0], port[1], port[2], age]).T
    yhat = rf_classifier.predict(x)
    return 'YES!!!' if yhat == 1 else 'I am sorry...'
