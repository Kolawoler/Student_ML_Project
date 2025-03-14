import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score , mean_absolute_error
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_obj(file_path, obj):

    try:

        dir_name = os.path.dirname(file_path)

        os.makedirs(dir_name, exist_ok= True)

        with open(file_path, "wb") as file_obj:

            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluation_model (x_train,y_train, X_test, y_test, models, param):

    report = {}

    try:
        for i in range(len(list(models))):

            for i in range(len(list(models))):
                model = list(models.values())[i]
                para= param[list(models.keys())[i]]

                gs = GridSearchCV(model,para,cv=3)
                gs.fit(x_train,y_train)
                best_params = gs.best_params_
                if best_params:
                    model.set_params(**best_params)

                    model.set_params(**gs.best_params_)

                    best_params = gs.best_params_
        
                    model.set_params(**best_params)
                    model.fit(x_train,y_train)


                #model.fit(x_train, y_train)  # Train model


                # model = list(models.values())[i]
                # model.fit(x_train, y_train)
                    y_train_predict = model.predict(x_train)
                    Y_test_predict = model.predict(X_test)
                    train_model_score = r2_score(y_train , y_train_predict)
                    test_model_score = r2_score(y_test , Y_test_predict)

                    report[list(models.keys())[i]] = test_model_score

            return report
        
    except Exception as e:
        raise CustomException(e, sys)
            


