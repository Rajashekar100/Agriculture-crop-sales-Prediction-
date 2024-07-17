from flask  import Flask,request,render_template
import numpy as np
import pandas as pd
import pickle
#loading models
lr=pickle.load(open("lr.pkl","rb"))
preprocessor=pickle.load(open("preprocessor.pkl","rb"))

#creating flask app
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict",methods=["POST"])
def predict():
    if request.method=="POST":
        State=request.form["State"]
        Year=request.form["Year"]
        District=request.form["District"]
        Crop=request.form["Crop"]
        Area=request.form["Area"]
        Yield=request.form["Yield"]


        features=np.array([[State,District,Crop,Year,Area,Yield]])
        transformed_features=preprocessor.transform(features)
        predicted_value=lr.predict(transformed_features).reshape(1,-1)

        return render_template("index.html",predicted_value=abs(predicted_value)/1000)

#python main
if __name__=="__main__":
    app.run(debug=True)




