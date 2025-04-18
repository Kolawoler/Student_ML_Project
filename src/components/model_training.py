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
from src.utils.utils import save_obj
from src.utils.utils import evaluation_model

@dataclass


class Modeltransformmer_config:
     trained_model_file_path = os.path.join("artificats", "model.pkl")

class model_training:
     
     def __init__(self):

          
          
          self.model_trainer = Modeltransformmer_config()
          
     def initiate_model_trainer(self, train_array, test_array):

          try:

               logging.info("splittingg has started ")
               X_train, X_test, Y_train, Y_test = train_array[:,:-1], test_array[:, :-1],train_array[:, -1], test_array[:, -1]

               Model_list = {"linear_model": LinearRegression(),"Random_forest": RandomForestRegressor(),
                    "Ridge" : Ridge(),
                    "Lasso": Lasso(),
                    "AdaBoost" : AdaBoostRegressor(),
                    "Decision_Tree": DecisionTreeRegressor(),
                    "Xgboost" : XGBRegressor() }
               

               params={
               "Decision_Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
               },

               "Lasso" : {"alpha" : [.2, .6, .9]},

               "Ridge" : {"alpha" : [ 1.2, .6]},

               "Random_forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
               
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
               },
             
               "linear_model":{},
               "Xgboost":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
               },
               "AdaBoost":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
               }
               
          }

               # model_report:dict=evaluation_model(x_train=X_train,y_train=Y_train,x_test=X_test,y_test=Y_test,
               #                                    model= Model_list, param=params)



               reports_value = evaluation_model(X_train, Y_train, X_test, Y_test, Model_list, param= params)
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
          
          except Exception as e :
               raise CustomException ( e, sys)





     
          





     

     
     
     
          





          
     

          

          









