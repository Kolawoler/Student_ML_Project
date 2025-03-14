import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.model_training import model_training
# from src.components.model_training import Modeltransformmer_config

# from src.components.data_transformation import DataTransformerconfig
from src.components.data_transformation import Data_transformer


@dataclass
class Data_ingestion_config :

    train_data_path :  str= os.path.join("artifacts", "train.csv")
    test_data_path :  str= os.path.join("artifacts", "test.csv")
    raw_data_data_path :  str= os.path.join("artifacts", "data.csv")


class Data_ingestion:

    def __init__(self):
        self.ingestion_config = Data_ingestion_config()
    


    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or compponent")


        try :
            df = pd.read_csv("C:/Users/Rotim/OneDrive/Documents/Data_Science_Projects/Student_ML_Project/Notebook/StudentsPerformance.csv") 
            
            print(df)
            logging.info("reading the dataset as dataframe")

            # # ensure the directory exist
            # artifacts_ = os.path.dirname(self.ingestion_config.train_data_path)
            # print(f'The directory does not exists {artifacts_}')
            # os.makedirs(artifacts_, exist_ok=True)



            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok= True)

            df.to_csv(self.ingestion_config.raw_data_data_path, index = False, header= True)

            logging.info("data splitting initiated")
            train_set , test_set = train_test_split(df, test_size = .2, random_state = 42 )
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)
            logging.info("data save to the respective folder")

            return(self.ingestion_config.train_data_path,
                   self.ingestion_config.test_data_path)
                   
        except Exception as e:

            print("Failed to read dataset:", e)  # Debugging output
            logging.error("Error reading dataset", exc_info=True)
            raise CustomException(sys, e)
        

if __name__ == "__main__":
    obj = Data_ingestion()
    train_data, test_data = obj.initiate_data_ingestion()

    Transformer = Data_transformer()
    train_array_, test_array_,_ = Transformer.initiate_data_trasformation(train_data, test_data)
    print("transformation completeted")

    Model_training = model_training()
    print(Model_training.initiate_model_trainer(train_array= train_array_, test_array= test_array_))






    

    




    
