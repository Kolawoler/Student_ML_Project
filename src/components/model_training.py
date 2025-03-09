from dataclasses import dataclass
from sklearn.ensemble import ( RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor )
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from src.logger import logging
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from src.exception import CustomException
import os, sys
# from catboost import CatBoostRegressor
from src.utils import save_obj
from src.utils import evaluation_model

@dataclass

class Modeltransformmer_config:
     trained_model_file_path = os.path.join("artificats", "model.pkl")

class model_training:
     
     def __init__(self):
          
          
          self.model_trainer = Modeltransformmer_config()
          
     def get_model_training(self, train_array, test_array):
          
          try:
               
               
               
               logging.info("splittingg has started ")
               X_train, X_test, Y_train, Y_test = train_array[:,:-1], test_array[:, :-1],train_array[:, -1], test_array[:, -1]

               Model_list = {"linear_model": LinearRegression(),"Random_forest": RandomForestRegressor(),
                    "Ridge" : Ridge(),
                    "Lasso": Lasso(),
                    "AdaBoost" : AdaBoostRegressor(),
                    "Decision_Tree": DecisionTreeRegressor(),
                    "Xgboost" : XGBRegressor() }
                        

               reports_value = evaluation_model(X_train, Y_train, X_test, Y_test, Model_list)
               print(f"the reprots of ths model is {reports_value}")


               best_model = sorted(reports_value.items() , key= lambda x : x [1], reverse= True) [0]
               print(f" thes best model is {best_model} ")


               best_model_name, best_model_score = best_model
               print( best_model_name)



               best_model = Model_list[best_model_name]

               if best_model_score < .6:
                    raise CustomException("No best model found")
               
               logging.info("best model found on both training and test")

               save_obj(file_path= self.model_trainer.trained_model_file_path, obj= best_model
                        )
               
               predict_value = best_model.predict(X_test)

               r_scores = r2_score(Y_test, predict_value)

               return r_scores
          
          except Exception as e:
               raise CustomException (e, sys)
          





          
     

          

          









