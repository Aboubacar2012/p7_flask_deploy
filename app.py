from flask import Flask,render_template,request
import pandas as pd
import numpy as np
import pickle
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from io import StringIO
import os
from os.path import join, dirname, realpath
import lime
from lime import lime_tabular
from pathlib import Path
import matplotlib.pyplot as plt
from wtforms.fields import StringField
from wtforms import Form,TextField, BooleanField, PasswordField, TextAreaField, validators
from wtforms.widgets import TextArea
import time 

app=Flask(__name__, template_folder='template')


#load model
model = pickle.load(open('Credit_model_reg.pkl','rb'))

# Load Scaler
file_scaler= open('file_scale.dat','rb')
std_scale=pickle.load(file_scaler)
file_scaler.close()

#Télécharger le explainer
file_explain=open('Explainer.dat','rb')
explainer=open('features.pkl','rb')
file_explain.close()

#liste des identifiants
#liste_id = df['SK_ID_CURR'].values


@app.route('/',methods=["GET"])

def hello_word():
    return render_template("index.html")


# Load Dataframe
df = pd.read_csv("TEST_FINAL_SCALE_2.csv")

#print(df.shape)



@app.route('/',methods=["POST"])
def predict():
    for x in request.form.values():
        Identifiant = int(x)
    print(Identifiant)
    print(type(Identifiant))
    print(df.columns)
    print(df.head(10))
    #print(df['SK_ID_CURR'].dtypes)
    if Identifiant in df['SK_ID_CURR'].values:

        i=df['SK_ID_CURR'] == Identifiant

        Y = df[i]
        Y = Y.drop(['SK_ID_CURR'], axis=1)
        num = np.array(std_scale.transform(Y))
        pr = model.predict_proba(num)[:, 1]

        if pr > 0.5:
            classification = 'Rejet de la demande de crédit'


        else:
            classification = 'Acceptation de la demande de crédit'

        #return render_template('index.html', valeur=(pr[0]*100).round(0),prediction=classification)
        return render_template('index.html', prediction=classification)





@app.route('/')
@app.route('/', methods=['POST'])

@app.route('/')
@app.route('/index')

#
def interpretation ():
    exp = " "
    for x in request.form.values():
        Identifiant = int(x)
    ID = int(Identifiant)
    class_names = {0: 'Client Fiable', 1: 'Client Pas Fiable'}
    X1 = df[df['SK_ID_CURR'] == Identifiant]
    print('ID client: {}'.format(ID))
    print('Classe réelle : {}'.format(class_names[X1['LABELS'].values[0]]))
    print('Temps initialisation : ', time.time() - start_time)
    start_time = time.time()
    X1 = X1.drop(['SK_ID_CURR'], axis=1)
    dataframe = df.drop(['SK_ID_CURR'], axis=1)
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=np.array(std_scale.transform(dataframe)),
        #training_data=df.columns,
        feature_names=dataframe.columns,
        training_labels=dataframe.columns.tolist(),
        verbose=1,
        random_state=20,
        mode='classification')
    exp = explainer.explain_instance(data_row=X1.sort_index(axis=1).iloc[0:1, :].to_numpy().ravel(), \
                                     predict_fn=model.predict_proba)
    exp.save_to_file('output_filename.html')
    exp = exp.as_html()
    
    #buf = BytesIO()
    plt.savefig(exp,
                format="png",
                dpi=150,
                bbox_inches='tight')
    #dataToTake = base64.b64encode(buf.getbuffer()).decode("ascii")
    #return dataToTake
    
    #return render_template('index.html', exp=exp)
    return render_template("index.html", user_image=print(exp))


#




if __name__ == "__main__":
    app.run(port=5000,debug=True)

