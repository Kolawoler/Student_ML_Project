from dataclasses import dataclass
from sklearn.ensemble import ( RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor )
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from src.logger import logging
from src.exception import CustomException
import os, sys
from catboost import CatBoostRegressor

