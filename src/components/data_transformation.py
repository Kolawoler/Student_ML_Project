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
from src.utils.utils import save_obj

@dataclass
class DataTransformerconfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "Preprocessor.pkl")


class Data_transformer:
    def __init__(self):
        self.data_transfomer_config = DataTransformerconfig()

    def get_data_transfomer_obj(self):
        "here, is where the transformer happens"

        try:
            # Define the original feature names and their mapped names
            rename_mapping = {
                "reading_score": "reading score",
                "writing_score": "writing score",
                "gender": "gender",
                "race_ethnicity": "race/ethnicity",
                "parental_level_of_education": "parental level of education",
                "lunch": "lunch",
                "test_preparation_course": "test preparation course"
            }

            numerical_columns = ["reading score", "writing score"]
            categorical_columns = ["gender", "race/ethnicity", "parental level of education", "lunch",
                                   "test preparation course"]
            numerical_pipeline = Pipeline(
                steps=[("imputter", SimpleImputer(strategy='median')),
                       ("scaler", StandardScaler(with_mean=False))])

            categorical_pipeline = Pipeline(
                steps=[("imputter", SimpleImputer(strategy="most_frequent")),
                       ("one_hot_encoder", OneHotEncoder()),
                       ("scaler", StandardScaler(with_mean=False))])

            logging.info(f"categorical columns : {categorical_columns}")
            logging.info(f"Numerical columns : {numerical_columns}")

            Preprocessor = ColumnTransformer([("num_pipeline", numerical_pipeline, numerical_columns),
                                             ("cat_pipelines", categorical_pipeline, categorical_columns)])

            return Preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def rename_features(self, df):
        """
        Rename columns to match the expected naming convention (space-separated).
        """
        rename_mapping = {
            "reading_score": "reading score",
            "writing_score": "writing score",
            "gender": "gender",
            "race_ethnicity": "race/ethnicity",
            "parental_level_of_education": "parental level of education",
            "lunch": "lunch",
            "test_preparation_course": "test preparation course"
        }
        df.rename(columns=rename_mapping, inplace=True)
        return df

    def initiate_data_trasformation(self, train_path, test_path):
        try:
            logging.info("Read train and test data")
            logging.info("getting the transformer objects")

            # Read data
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            target_columns = ['math score']

            target_feature_train_df = train_data[target_columns]
            input_feature_train = train_data.drop(target_columns, axis=1)

            target_feature_test_df = test_data[target_columns]
            input_feature_test_ = test_data.drop(target_columns, axis=1)

            logging.info("Renaming columns to match space-separated convention")
            input_feature_train = self.rename_features(input_feature_train)
            input_feature_test_ = self.rename_features(input_feature_test_)

            logging.info("Applying processing object on testing and training data")

            # Get transformer object
            obj_transformer = self.get_data_transfomer_obj()

            # Apply transformations
            train_transformed_data_arr = obj_transformer.fit_transform(input_feature_train)
            test_transformed_data_arr = obj_transformer.transform(input_feature_test_)

            logging.info("adding the data and target feature")

            # Combine the transformed features with the target variable
            train_array = np.c_[train_transformed_data_arr, np.array(target_feature_train_df)]
            test_array = np.c_[test_transformed_data_arr, np.array(target_feature_test_df)]

            logging.info("saving preprocessing object")

            # Save the preprocessing object
            save_obj(file_path=self.data_transfomer_config.preprocessor_obj_file_path,
                     obj=obj_transformer)

            return train_array, test_array, self.data_transfomer_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
