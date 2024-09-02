import numpy as np
import pandas as pd
import sys
import os
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from src.logger import logging
from src.exception import CustomException
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.utils import save_obj
from src.components.data_ingestion import Data_ingestion

# from src.components.data_transformation import Data_transformer
# from src.components.data_transformation import DataTransformerconfig

@dataclass
class DataTransformerconfig :
    preprocessor_obj_file_path = os.path.join("artifacts", "Preprocessor.pkl")


class Data_transformer : 
    def __init__(self):
        self.data_transfomer_config = DataTransformerconfig()

    def get_data_transfomer_obj(self):
        "here, is where the transformer happens"

        try:
            numerical_columns = ["reading score", "writing score"]
            categorical_columns = ["gender","race/ethnicity","parental level of education","lunch",
                                   "test preparation course"]
            numerical_pipeline = Pipeline(
                steps=[("imputter",SimpleImputer(strategy = 'median')),
                        ("scaler",StandardScaler(with_mean=False))])
            
            categorical_pipeline = Pipeline(
                steps=[("imputter",SimpleImputer(strategy = "most_frequent")),
                        ("one_hot_encoder",OneHotEncoder()),
                        ("scaler",StandardScaler(with_mean=False))])
            
            logging.info(f"categorical columns : {categorical_columns}")
            logging.info(f"Numerical columns : {numerical_columns}")
        

            Preprocessor = ColumnTransformer([("num_pipeline", numerical_pipeline,numerical_columns),
                                ("cat_pipelines", categorical_pipeline, categorical_columns)])
            
            
            return Preprocessor

        except Exception as e :
            raise CustomException(e, sys)  
        
    def initiate_data_trasformation(self,train_path, test_path):

        try:

            logging.info("Read train and test data")
            logging.info("getting the transformer objects")


            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            target_columns = ['math score']

            target_feature_train_df = train_data[target_columns]
            input_feature_train = train_data.drop(target_columns, axis = 1)


            target_feature_test_df = test_data[target_columns]
            input_feature_test_ = test_data.drop(target_columns, axis = 1)

            logging.info("Applying processing object on testing and training data")

            obj_transformer = self.get_data_transfomer_obj()

            train_transformed_data_arr = obj_transformer.fit_transform(input_feature_train)
            test_transformed_data_arr = obj_transformer.transform(input_feature_test_)

            logging.info("adding the data and target feature")

            train_array = np.c_[train_transformed_data_arr, np.array(target_feature_train_df)]
            test_array = np.c_[test_transformed_data_arr, np.array(target_feature_test_df)]


            logging.info("saved preprocessing object")

            

            return (train_transformed_data_arr, test_transformed_data_arr,
                    self.data_transfomer_config.preprocessor_obj_file_path 

            )



        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = Data_ingestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = Data_transformer()
    data_transformation.initiate_data_trasformation(train_path=train_data, test_path= test_data)










