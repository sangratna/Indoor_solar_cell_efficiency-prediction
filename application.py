import pickle
from flask import Flask, request, jsonify,render_template
import numpy as np
import pandas as pd

application = Flask(__name__)
app=application
#postgres://fwvfb_user:rwkgXudN8zPVVUTEp7ZcaQCEqEYkenx3@dpg-cjgsobk1ja0c73chlcj0-a.oregon-postgres.render.com/fwvfb
dtr_model = pickle.load(open('model/regressor.pkl', 'rb'))


@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=="POST":
        Bandgap=float(request.form.get('Bandgap'))
        HOMO=float(request.form.get('HOMO'))
        LUMO=float(request.form.get('LUMO'))
        Light_source=int(request.form.get('Light_source'))
        Illuminance=float(request.form.get('Illuminance'))

        new_data=([[Bandgap,HOMO,LUMO,Light_source,Illuminance]])
        result=dtr_model.predict(new_data)
        return render_template('home.html',results=result[0])
        

    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0")
