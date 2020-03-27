import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import matplotlib as plt

heart_rates = np.array([50,70,90]).reshape(-1,1)
logit = np.array([-1.1, 0, 1.3]).reshape(-1,1)

predictionVar = np.array([60]).reshape(-1,1)

LR = LinearRegression()
LR.fit(heart_rates, logit)
print("Weight", LR.coef_)
print("Bias", LR.intercept_)
print("prediction", LR.predict(predictionVar))


all_heart_rates = np.array([50,50,50,50,70,70,90,90,90,90,90]).reshape(-1,1)
heart_attacks = np.array([1,0,0,0,0,1,1,1,0,1,1,])

LOR = LogisticRegression(random_state=0, solver='lbfgs')
LOR.fit(all_heart_rates, heart_attacks)
myArray = LOR.predict_proba(all_heart_rates)
print("Weight", LOR.coef_)
print("Bias", LOR.intercept_)
print("prediction", LOR.predict_proba(predictionVar))