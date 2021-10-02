import pandas as pd  
import xgboost
import shap
import sys

#import dataset

file = "framingham.csv"
dataset = pd.read_csv(file, sep=";")


#dataset describe
dataset.info()
dataset.describe()

#male = nominal: 0 or 1
# age from 32 to 70
# education = nominal 1,2,3,4
#currentSmoker = nominal 0 or 1
#cigsPerDay = from 20 to 70
# BPMeds = nominal 0 or 1 (blood pressure medication)
#prevalentStroke = whether or not the patient had previously had a stroke = nominal 0 or 1
#PrevalentHyp = whether or not the patient was hypertensive = nominal 0 or 1
#diabetes = whether or not the patient had diabetes = nominal = 0 or 1 
#totChol = total cholesterol level = from 107 to 696
#Sys BP: systolic blood pressure (Continuous)
#Dia BP: diastolic blood pressure (Continuous)
#BMI: Body Mass Index (Continuous)
#Heart Rate: heart rate (Continuous
#Glucose: glucose level (Continuous)
#TenYearCHD = label 0 = No, 1 = yes 

dataset["male"].value_counts()

#0 = 2419
#1 = 1819


dataset["education"].value_counts()
#1.0    1720
#2.0    1253
#3.0     687
#4.0     473

dataset["currentSmoker"].value_counts()
#0    2144
#1    2094

dataset["BPMeds"].value_counts()
#0.0    4061
#1.0     124

dataset["prevalentStroke"].value_counts()
#0    4213
#1      25

dataset["prevalentHyp"].value_counts()
#0    2922
#1    1316

dataset["diabetes"].value_counts()
#0    4129
#1     109

dataset["TenYearCHD"].value_counts()
#0    3594
#1     644


print(dataset.isnull().sum())

#separate X e Y (label)  data
X = dataset.iloc[:, 0:15].values
Y = dataset.iloc[:,-1 ].values


# train an XGBoost model
model = xgboost.XGBRegressor().fit(X, Y)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shap.Explainer(model)
shap_values = explainer(X)

# visualize the first prediction's explanation
shap.plots.waterfall(shap_values[0])

# summarize the effects of all the features
shap.plots.beeswarm(shap_values)

shap.plots.bar(shap_values)
