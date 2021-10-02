import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt


from sklearn.impute import KNNImputer
from  sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import Normalizer

from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import confusion_matrix

#import dataset

file = "framingham.csv"
dataset = pd.read_csv(file, sep=";")


dataset["male"].fillna(9999, inplace=True)
dataset["education"].fillna(9999, inplace=True)
dataset["currentSmoker"].fillna(9999, inplace=True)
dataset["BPMeds"].fillna(9999, inplace=True)
dataset["prevalentStroke"].fillna(9999, inplace=True)
dataset["prevalentHyp"].fillna(9999, inplace=True)
dataset["diabetes"].fillna(9999, inplace=True)

dataset["age"].fillna(999, inplace=True)
dataset["totChol"].fillna(999, inplace=True)
dataset["sysBP"].fillna(999, inplace=True)
dataset["diaBP"].fillna(999, inplace=True)
dataset["BMI"].fillna(999, inplace=True)
dataset["heartRate"].fillna(999, inplace=True)
dataset["glucose"].fillna(999, inplace=True)
dataset["cigsPerDay"].fillna(999, inplace=True)

X = dataset.iloc[:, 0:15].values
Y = dataset.iloc[:,-1 ].values




imputer_categorical = KNNImputer(missing_values=9999, n_neighbors=1)
imputer_ordinal = KNNImputer(missing_values=999, n_neighbors=20)


# for feature evaluating with boxplot 
X = imputer_categorical.fit_transform(X)
X = imputer_ordinal.fit_transform(X)

X = pd.DataFrame(X, columns=["male", "age", "education", "currentSmoker", "cigsPerDay", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"])



stand_standardScaler = StandardScaler()
stand_MinMaxScaler = MinMaxScaler()
stand_MaxAbsScaler = MaxAbsScaler()
stand_RobustScaler = RobustScaler()
stand_quantileTransformNormal = QuantileTransformer(output_distribution='normal')
stand_quantileTransform_Uniform = QuantileTransformer(output_distribution='uniform')
stand_Power = PowerTransformer()
stand_normalizer = Normalizer()

enc = OneHotEncoder(handle_unknown='ignore')

enc_male = pd.DataFrame(enc.fit_transform(X[["male"]]).toarray())
X = X.drop("male", axis=1)
enc_education = pd.DataFrame(enc.fit_transform(X[["education"]]).toarray())
X = X.drop("education", axis=1)
enc_currentSmoker = pd.DataFrame(enc.fit_transform(X[["currentSmoker"]]).toarray())
X = X.drop("currentSmoker", axis=1)
enc_BPMeds = pd.DataFrame(enc.fit_transform(X[["BPMeds"]]).toarray())
X = X.drop("BPMeds", axis=1)
enc_prevalentStroke = pd.DataFrame(enc.fit_transform(X[["prevalentStroke"]]).toarray())
X = X.drop("prevalentStroke", axis=1)
enc_prevalentHyp = pd.DataFrame(enc.fit_transform(X[["prevalentHyp"]]).toarray())
X = X.drop("prevalentHyp", axis=1)
enc_diabetes = pd.DataFrame(enc.fit_transform(X[["diabetes"]]).toarray())
X = X.drop("diabetes", axis=1)

X = X.join(enc_male, rsuffix="_male")
X = X.join(enc_education, rsuffix="_education")
X = X.join(enc_currentSmoker, rsuffix="_currentSmoker")
X = X.join(enc_BPMeds, rsuffix="_BPMeds")
X = X.join(enc_prevalentHyp, rsuffix="_prevalentHyp")
X = X.join(enc_prevalentStroke, rsuffix="_prevalentStroke")
X = X.join(enc_diabetes, rsuffix="_diabetes")


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

X_train_standard_scaler = pd.DataFrame.copy(X_train)
X_train_minMax = pd.DataFrame.copy(X_train)
X_train_maxAbs = pd.DataFrame.copy(X_train)
X_train_robust = pd.DataFrame.copy(X_train)
X_train_quantileNormal = pd.DataFrame.copy(X_train)
X_train_quantile_uniform = pd.DataFrame.copy(X_train)
X_train_power = pd.DataFrame.copy(X_train)
X_train_normalize = pd.DataFrame.copy(X_train)

X_test_standard_scaler = pd.DataFrame.copy(X_test)
X_test_minMax = pd.DataFrame.copy(X_test)
X_test_maxAbs = pd.DataFrame.copy(X_test)
X_test_robust = pd.DataFrame.copy(X_test)
X_test_quantileNormal = pd.DataFrame.copy(X_test)
X_test_quantile_uniform = pd.DataFrame.copy(X_test)
X_test_power = pd.DataFrame.copy(X_test)
X_test_normalize = pd.DataFrame.copy(X_test)


#######
####### STANDARD SCALER
#######

X_train_standard_scaler[["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]] = stand_standardScaler.fit_transform(X_train_standard_scaler[["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]])
X_test_standard_scaler[["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]] = stand_standardScaler.fit_transform(X_test_standard_scaler[["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]])

classifier_logistic_regression_standardScaler = LogisticRegression()
classifier_logistic_regression_standardScaler.fit(X_train_standard_scaler, Y_train)
print("model score Logistic Regression Standard scaler: %.3f" % classifier_logistic_regression_standardScaler.score(X_test_standard_scaler, Y_test))

classifier_SVC_standardScaler = SVC()
classifier_SVC_standardScaler.fit(X_train_standard_scaler, Y_train)
print("model score SVC Standard scaler: %.3f" % classifier_SVC_standardScaler.score(X_test_standard_scaler, Y_test))

classifier_DT_standardScaler = DecisionTreeClassifier()
classifier_DT_standardScaler.fit(X_train_standard_scaler, Y_train)
print("model score DT Standard scaler: %.3f" % classifier_DT_standardScaler.score(X_test_standard_scaler, Y_test))

classifier_RFC_standardScaler = RandomForestClassifier()
classifier_RFC_standardScaler.fit(X_train_standard_scaler, Y_train)
print("model score RF Standard scaler: %.3f" % classifier_RFC_standardScaler.score(X_test_standard_scaler, Y_test))

classifier_MLP_standardScaler = MLPClassifier()
classifier_MLP_standardScaler.fit(X_train_standard_scaler, Y_train)
print("model score MLP Standard scaler: %.3f" % classifier_MLP_standardScaler.score(X_test_standard_scaler, Y_test))



#######
####### MIN MAX SCALER
#######


X_train_minMax[["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]] = stand_MinMaxScaler.fit_transform(X_train_minMax[["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]])
X_test_minMax[["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]] = stand_MinMaxScaler.fit_transform(X_test_minMax[["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]])


classifier_logistic_regression_minMax = LogisticRegression()
classifier_logistic_regression_minMax.fit(X_train_minMax, Y_train)
print("model score Logistic Regression minMax Scaler: %.3f" % classifier_logistic_regression_minMax.score(X_test_minMax, Y_test))


classifier_SVC_minMax = SVC()
classifier_SVC_minMax.fit(X_train_minMax, Y_train)
print("model score SVC minMax Scaler: %.3f" % classifier_SVC_minMax.score(X_test_minMax, Y_test))

classifier_DT_minMax = DecisionTreeClassifier()
classifier_DT_minMax.fit(X_train_minMax, Y_train)
print("model score DT minMax Scaler: %.3f" % classifier_DT_minMax.score(X_test_minMax, Y_test))

classifier_RFC_minMax = RandomForestClassifier()
classifier_RFC_minMax.fit(X_train_minMax, Y_train)
print("model score RF minMax Scaler: %.3f" % classifier_RFC_minMax.score(X_test_minMax, Y_test))

classifier_MLP_minMax = MLPClassifier()
classifier_MLP_minMax.fit(X_train_minMax, Y_train)
print("model score MLP minMax Scaler: %.3f" % classifier_MLP_minMax.score(X_test_minMax, Y_test))






#######
####### MAX ABS SCALER
#######


X_train_maxAbs[["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]] = stand_MaxAbsScaler.fit_transform(X_train_maxAbs[["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]])
X_test_maxAbs[["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]] = stand_MaxAbsScaler.fit_transform(X_test_maxAbs[["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]])



classifier_logistic_regression_maxAbs = LogisticRegression()
classifier_logistic_regression_maxAbs.fit(X_train_maxAbs, Y_train)
print("model score Logistic Regression maxAbs Scaler: %.3f" % classifier_logistic_regression_maxAbs.score(X_test_maxAbs, Y_test))


classifier_SVC_maxAbs = SVC()
classifier_SVC_maxAbs.fit(X_train_maxAbs, Y_train)
print("model score SVC maxAbs Scaler: %.3f" % classifier_SVC_maxAbs.score(X_test_maxAbs, Y_test))

classifier_DT_maxAbs = DecisionTreeClassifier()
classifier_DT_maxAbs.fit(X_train_maxAbs, Y_train)
print("model score DT maxAbs Scaler: %.3f" % classifier_DT_maxAbs.score(X_test_maxAbs, Y_test))

classifier_RFC_maxAbs = RandomForestClassifier()
classifier_RFC_maxAbs.fit(X_train_maxAbs, Y_train)
print("model score RF maxAbs Scaler: %.3f" % classifier_RFC_maxAbs.score(X_test_maxAbs, Y_test))

classifier_MLP_maxAbs = MLPClassifier()
classifier_MLP_maxAbs.fit(X_train_maxAbs, Y_train)
print("model score MLP maxAbs Scaler: %.3f" % classifier_MLP_maxAbs.score(X_test_maxAbs, Y_test))






#######
####### ROBUST SCALER
#######

X_train_robust[["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]] = stand_RobustScaler.fit_transform(X_train_robust[["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]])
X_test_robust[["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]] = stand_RobustScaler.fit_transform(X_test_robust[["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]])



classifier_logistic_regression_robust = LogisticRegression()
classifier_logistic_regression_robust.fit(X_train_robust, Y_train)
print("model score Logistic Regression robust Scaler: %.3f" % classifier_logistic_regression_robust.score(X_test_robust, Y_test))


classifier_SVC_robust = SVC()
classifier_SVC_robust.fit(X_train_robust, Y_train)
print("model score SVC robust Scaler: %.3f" % classifier_SVC_robust.score(X_test_robust, Y_test))

classifier_DT_robust = DecisionTreeClassifier()
classifier_DT_robust.fit(X_train_robust, Y_train)
print("model score DT robust Scaler: %.3f" % classifier_DT_robust.score(X_test_robust, Y_test))

classifier_RFC_robust = RandomForestClassifier()
classifier_RFC_robust.fit(X_train_robust, Y_train)
print("model score RF robust Scaler: %.3f" % classifier_RFC_robust.score(X_test_robust, Y_test))

classifier_MLP_robust = MLPClassifier()
classifier_MLP_robust.fit(X_train_robust, Y_train)
print("model score MLP robust Scaler: %.3f" % classifier_MLP_robust.score(X_test_robust, Y_test))








#######
####### QUANTILE NORMAL SCALER
#######

X_train_quantileNormal[["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]] = stand_quantileTransformNormal.fit_transform(X_train_quantileNormal[["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]])
X_test_quantileNormal[["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]] = stand_quantileTransformNormal.fit_transform(X_test_quantileNormal[["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]])



classifier_logistic_regression_quantileNormal = LogisticRegression()
classifier_logistic_regression_quantileNormal.fit(X_train_quantileNormal, Y_train)
print("model score Logistic Regression quantile normal Scaler: %.3f" % classifier_logistic_regression_quantileNormal.score(X_test_quantileNormal, Y_test))


classifier_SVC_quantileNormal = SVC()
classifier_SVC_quantileNormal.fit(X_train_quantileNormal, Y_train)
print("model score SVC quantile normal Scaler: %.3f" % classifier_SVC_quantileNormal.score(X_test_quantileNormal, Y_test))

classifier_DT_quantileNormal = DecisionTreeClassifier()
classifier_DT_quantileNormal.fit(X_train_quantileNormal, Y_train)
print("model score DT quantile normal Scaler: %.3f" % classifier_DT_quantileNormal.score(X_test_quantileNormal, Y_test))

classifier_RFC_quantileNormal = RandomForestClassifier()
classifier_RFC_quantileNormal.fit(X_train_quantileNormal, Y_train)
print("model score RF quantile normal Scaler: %.3f" % classifier_RFC_quantileNormal.score(X_test_quantileNormal, Y_test))

classifier_MLP_quantileNormal = MLPClassifier()
classifier_MLP_quantileNormal.fit(X_train_quantileNormal, Y_train)
print("model score MLP quantile normal Scaler: %.3f" % classifier_MLP_quantileNormal.score(X_test_quantileNormal, Y_test))







#######
####### QUANTILE UNIFORM SCALER
#######

X_train_quantile_uniform[["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]] = stand_quantileTransform_Uniform.fit_transform(X_train_quantile_uniform[["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]])
X_test_quantile_uniform[["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]] = stand_quantileTransform_Uniform.fit_transform(X_test_quantile_uniform[["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]])



classifier_logistic_regression_quantileUniform = LogisticRegression()
classifier_logistic_regression_quantileUniform.fit(X_train_quantile_uniform, Y_train)
print("model score Logistic Regression quantile Uniform Scaler: %.3f" % classifier_logistic_regression_quantileUniform.score(X_test_quantile_uniform, Y_test))


classifier_SVC_quantileUniform = SVC()
classifier_SVC_quantileUniform.fit(X_train_quantile_uniform, Y_train)
print("model score SVC quantile Uniform Scaler: %.3f" % classifier_SVC_quantileUniform.score(X_test_quantile_uniform, Y_test))

classifier_DT_quantileUniform = DecisionTreeClassifier()
classifier_DT_quantileUniform.fit(X_train_quantile_uniform, Y_train)
print("model score DT quantile Uniform Scaler: %.3f" % classifier_DT_quantileUniform.score(X_test_quantile_uniform, Y_test))

classifier_RFC_quantileUniform = RandomForestClassifier()
classifier_RFC_quantileUniform.fit(X_train_quantile_uniform, Y_train)
print("model score RF quantile Uniform Scaler: %.3f" % classifier_RFC_quantileUniform.score(X_test_quantile_uniform, Y_test))

classifier_MLP_mquantileUniform= MLPClassifier()
classifier_MLP_mquantileUniform.fit(X_train_quantile_uniform, Y_train)
print("model score MLP quantile Uniform Scaler: %.3f" % classifier_MLP_mquantileUniform.score(X_test_quantile_uniform, Y_test))










#######
####### POWER SCALER
#######

X_train_power[["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]] = stand_Power.fit_transform(X_train_power[["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]])
X_test_power[["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]] = stand_Power.fit_transform(X_test_power[["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]])



classifier_logistic_regression_standPower= LogisticRegression()
classifier_logistic_regression_standPower.fit(X_train_power, Y_train)
print("model score Logistic Regression power  Scaler: %.3f" % classifier_logistic_regression_standPower.score(X_test_power, Y_test))


classifier_SVC_standPower = SVC()
classifier_SVC_standPower.fit(X_train_power, Y_train)
print("model score SVC power  Scaler: %.3f" % classifier_SVC_standPower.score(X_test_power, Y_test))

classifier_DT_standPower = DecisionTreeClassifier()
classifier_DT_standPower.fit(X_train_power, Y_train)
print("model score DT power  Scaler: %.3f" % classifier_DT_standPower.score(X_test_power, Y_test))

classifier_RFC_standPower = RandomForestClassifier()
classifier_RFC_standPower.fit(X_train_power, Y_train)
print("model score RF power  Scaler: %.3f" % classifier_RFC_standPower.score(X_test_power, Y_test))

classifier_MLP_standPower = MLPClassifier()
classifier_MLP_standPower.fit(X_train_power, Y_train)
print("model score MLP power  Scaler: %.3f" % classifier_MLP_standPower.score(X_test_power, Y_test))






#######
####### NORMALIZE SCALER
#######

X_train_normalize[["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]] = stand_normalizer.fit_transform(X_train_normalize[["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]])
X_test_normalize[["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]] = stand_normalizer.fit_transform(X_test_normalize[["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]])



classifier_logistic_regression_normalize= LogisticRegression()
classifier_logistic_regression_normalize.fit(X_train_normalize, Y_train)
print("model score Logistic Regression normalize Scaler: %.3f" % classifier_logistic_regression_normalize.score(X_test_normalize, Y_test))


classifier_SVC_normalize = SVC()
classifier_SVC_normalize.fit(X_train_normalize, Y_train)
print("model score SVC normalize Scaler: %.3f" % classifier_SVC_normalize.score(X_test_normalize, Y_test))

classifier_DT_normalize = DecisionTreeClassifier()
classifier_DT_normalize.fit(X_train_normalize, Y_train)
print("model score DT normalize Scaler: %.3f" % classifier_DT_normalize.score(X_test_normalize, Y_test))

classifier_RFC_normalize = RandomForestClassifier()
classifier_RFC_normalize.fit(X_train_normalize, Y_train)
print("model score RF normalize Scaler: %.3f" % classifier_RFC_normalize.score(X_test_normalize, Y_test))

classifier_MLP_normalize = MLPClassifier()
classifier_MLP_normalize.fit(X_train_normalize, Y_train)
print("model score MLP normalize Scaler: %.3f" % classifier_MLP_normalize.score(X_test_normalize, Y_test))




#LOGISTIC REGRESSION#
print("#LOGISTIC REGRESSION#")
print("_____________________")
print("model score Logistic Regression Standard scaler: %.3f" % classifier_logistic_regression_standardScaler.score(X_test_standard_scaler, Y_test))
print("model score Logistic Regression minMax Scaler: %.3f" % classifier_logistic_regression_minMax.score(X_test_minMax, Y_test))
print("model score Logistic Regression maxAbs Scaler: %.3f" % classifier_logistic_regression_maxAbs.score(X_test_maxAbs, Y_test))
print("model score Logistic Regression robust Scaler: %.3f" % classifier_logistic_regression_robust.score(X_test_robust, Y_test))
print("model score Logistic Regression quantile normal Scaler: %.3f" % classifier_logistic_regression_quantileNormal.score(X_test_quantileNormal, Y_test))
print("model score Logistic Regression quantile Uniform Scaler: %.3f" % classifier_logistic_regression_quantileUniform.score(X_test_quantile_uniform, Y_test))
print("model score Logistic Regression power  Scaler: %.3f" % classifier_logistic_regression_standPower.score(X_test_power, Y_test))
print("model score Logistic Regression normalize Scaler: %.3f" % classifier_logistic_regression_normalize.score(X_test_normalize, Y_test))



#SVC#
print("#SVC#")
print("_____________________")
print("model score SVC Standard scaler: %.3f" % classifier_SVC_standardScaler.score(X_test_standard_scaler, Y_test))
print("model score SVC minMax Scaler: %.3f" % classifier_SVC_minMax.score(X_test_minMax, Y_test))
print("model score SVC maxAbs Scaler: %.3f" % classifier_SVC_maxAbs.score(X_test_maxAbs, Y_test))
print("model score SVC robust Scaler: %.3f" % classifier_SVC_robust.score(X_test_robust, Y_test))
print("model score SVC quantile normal Scaler: %.3f" % classifier_SVC_quantileNormal.score(X_test_quantileNormal, Y_test))
print("model score SVC quantile Uniform Scaler: %.3f" % classifier_SVC_quantileUniform.score(X_test_quantile_uniform, Y_test))
print("model score SVC power  Scaler: %.3f" % classifier_SVC_standPower.score(X_test_power, Y_test))
print("model score SVC normalize Scaler: %.3f" % classifier_SVC_normalize.score(X_test_normalize, Y_test))
print("_____________________")
print("_____________________")



#DT#
print("#DT#")
print("_____________________")
print("model score DT Standard scaler: %.3f" % classifier_DT_standardScaler.score(X_test_standard_scaler, Y_test))
print("model score DT minMax Scaler: %.3f" % classifier_DT_minMax.score(X_test_minMax, Y_test))
print("model score DT maxAbs Scaler: %.3f" % classifier_DT_maxAbs.score(X_test_maxAbs, Y_test))
print("model score DT robust Scaler: %.3f" % classifier_DT_robust.score(X_test_robust, Y_test))
print("model score DT quantile normal Scaler: %.3f" % classifier_DT_quantileNormal.score(X_test_quantileNormal, Y_test))
print("model score DT quantile Uniform Scaler: %.3f" % classifier_DT_quantileUniform.score(X_test_quantile_uniform, Y_test))
print("model score DT power  Scaler: %.3f" % classifier_DT_standPower.score(X_test_power, Y_test))
print("model score DT normalize Scaler: %.3f" % classifier_DT_normalize.score(X_test_normalize, Y_test))
print("_____________________")
print("_____________________")



#RF#
print("#RF#")
print("_____________________")
print("model score RF Standard scaler: %.3f" % classifier_RFC_standardScaler.score(X_test_standard_scaler, Y_test))
print("model score RF minMax Scaler: %.3f" % classifier_RFC_minMax.score(X_test_minMax, Y_test))
print("model score RF maxAbs Scaler: %.3f" % classifier_RFC_maxAbs.score(X_test_maxAbs, Y_test))
print("model score RF robust Scaler: %.3f" % classifier_RFC_robust.score(X_test_robust, Y_test))
print("model score RF quantile normal Scaler: %.3f" % classifier_RFC_quantileNormal.score(X_test_quantileNormal, Y_test))
print("model score RF quantile Uniform Scaler: %.3f" % classifier_RFC_quantileUniform.score(X_test_quantile_uniform, Y_test))
print("model score RF power  Scaler: %.3f" % classifier_RFC_standPower.score(X_test_power, Y_test))
print("model score RF normalize Scaler: %.3f" % classifier_RFC_normalize.score(X_test_normalize, Y_test))
print("_____________________")
print("_____________________")


#MLP#
print("#MLP#")
print("_____________________")
print("model score MLP Standard scaler: %.3f" % classifier_MLP_standardScaler.score(X_test_standard_scaler, Y_test))
print("model score MLP minMax Scaler: %.3f" % classifier_MLP_minMax.score(X_test_minMax, Y_test))
print("model score MLP maxAbs Scaler: %.3f" % classifier_MLP_maxAbs.score(X_test_maxAbs, Y_test))
print("model score MLP robust Scaler: %.3f" % classifier_MLP_robust.score(X_test_robust, Y_test))
print("model score MLP quantile normal Scaler: %.3f" % classifier_MLP_quantileNormal.score(X_test_quantileNormal, Y_test))
print("model score MLP quantile Uniform Scaler: %.3f" % classifier_MLP_mquantileUniform.score(X_test_quantile_uniform, Y_test))
print("model score MLP power  Scaler: %.3f" % classifier_MLP_standPower.score(X_test_power, Y_test))
print("model score MLP normalize Scaler: %.3f" % classifier_MLP_normalize.score(X_test_normalize, Y_test))
print("_____________________")
print("_____________________")



import tensorflow as tf
print (("Tensorflow version: {0}").format(tf.__version__))

tensor_classifier = tf.keras.models.Sequential()

#layer
tensor_classifier.add(tf.keras.layers.Dense(50, activation="relu", input_dim = X_train_power.shape[1]))


tensor_classifier.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

#compile tensorflow
tensor_classifier.compile(optimizer="adam",
                            loss = tf.losses.binary_crossentropy,
                            metrics =["accuracy"])

model_summary = tensor_classifier.summary()
model_config = tensor_classifier.get_config()

#train tensorflow

history = tensor_classifier.fit(X_train_power, Y_train, epochs=50, batch_size=32)


#train tensorflow evaluation
print("evalutate on train data")

y_train_predicted_tensor = tensor_classifier.predict(X_train_power)
y_train_predicted_tensor_approx = (np.rint(y_train_predicted_tensor))
cm = confusion_matrix(y_train_predicted_tensor_approx, Y_train)
print(cm)

results = tensor_classifier.evaluate(X_train_power, Y_train, batch_size = 128)
print("test loss, test acc:", results)


#test tensorflow evaluation
print("evalutate on test data")

y_test_tensor = tensor_classifier.predict(X_test_power)
y_test_predicted_tensor_approx = (np.rint(y_test_tensor))
cm = confusion_matrix(y_test_predicted_tensor_approx, Y_test)
print(cm)

results = tensor_classifier.evaluate(X_test_power, Y_test, batch_size = 128)
print("test loss, test acc:", results)
