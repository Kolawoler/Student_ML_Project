import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score , mean_absolute_error


from src.exception import CustomException


def save_obj(file_path, obj):

    try:

        dir_name = os.path.dirname(file_path)

        os.makedirs(dir_name, exist_ok= True)

        with open(file_path, "wb") as file_obj:

            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluation_model (X_train,Y_train, X_test, Y_test, models):

    report = {}

    try:
        for i in range(len(list(models))):

            model = list(models.values())[i]
            model.fit(X_train, Y_train)
            Y_train_predict = model.predict(X_train)
            Y_test_predict = model.predict(X_test)
            train_model_score = r2_score(Y_train , Y_train_predict)
            test_model_score = r2_score(Y_test , Y_test_predict)

            report[list(models.keys())[i]] = test_model_score

        return report
        
    except Exception as e:
        raise CustomException(e, sys)
            


