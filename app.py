
from flask import Flask
from flask import Flask, render_template, request
import sklearn
import pickle
# from assembleModel.modelcombine import combineModel
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import pandas as pd

random = pickle.load(open('heart_random.pkl','rb'))
randoms = pickle.load(open('heart_bagging.pkl','rb'))
xg = pickle.load(open('heart_xgb.pkl','rb'))
lgb = pickle.load(open('heart_lgb.pkl','rb'))

app = Flask(__name__)


@app.route("/")
@app.route('/index')
def index():
    return render_template("index.html")


@app.route('/login')
def login():
    return render_template("login.html")


@app.route('/upload')
def upload():
    return render_template("upload.html")

@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset)
        return render_template("preview.html",df_view = df)


@app.route("/heart_disease")
def heart_disease():
    return render_template("heart_disease.html")


@app.route("/heart-disease-predictor", methods=['POST', 'GET'])
def heartDiseasePredictor():
    if request.method == 'POST':
        result = request.form.to_dict()
        age = result['age']
        gender = result['gender']
        chest_pain_type = result['chest-pain-type']
        resting_blood_pressure = result['resting-blood-pressure']
        serum_cholestrol_value =result['serum-cholestrol-value']
        fasting_blood_sugar = result['fasting-blood-sugar']
        resting_ecg =result['resting-ecg']
        heart_rate_value = result['heart-rate-value']
        induced_agina = result['induced-agina']
        st_depression_value = result['st-depressed-value']
        peak_exercise_st = result['peak-exercise-st']
         
         
         
         
        model = result['model']
        print(model)
        element = [age, gender, chest_pain_type,resting_blood_pressure,serum_cholestrol_value,fasting_blood_sugar,resting_ecg,heart_rate_value,induced_agina,st_depression_value,peak_exercise_st]
        print(element)
        int_feature = [float(i) for i in element]
        print(int_feature)
    

		# Reshape the Data as a Sample not Individual Features
        
        ex1 = np.array(int_feature).reshape(1,-1)
        print(ex1)
        # prediction,prediction_prob = combineModel(models=[model1],element=element)
        # k = 0
        # if prediction>=0.5:
        #     k=1
        # result['prediction'] = k 
        # result['prediction-prob']=prediction_prob

        if model == 'Randomforset':
           result_prediction = random.predict(ex1)
           
            
        elif model == 'BaggingClassifier':
          result_prediction = randoms.predict(ex1)
        

        elif model == 'xgboost':
            result_prediction = xg.predict(ex1)
         

        elif model == 'lightgbm':
            result_prediction = lgb.predict(ex1)
        
         
     
    return render_template("result.html", prediction=result_prediction[0],model=model )
    
@app.route("/performance")
def performance():
    return render_template("performance.html")

@app.route("/chart")
def chart():
    return render_template("chart.html")
if __name__ == "__main__":
    app.run()
