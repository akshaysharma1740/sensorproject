import sys
import os
import numpy as np
import pandas as pd
from typing import list, Tuple, Generator
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    artifact_folder: str = os.path.join(artifact_folder)
    trained_model_path: str = os.path.join(artifact_folder, "model.pkl")
    expected_accuracy: float = 0.45
    model_config_file_path: str = os.path.join("config", "model.yaml")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.utils = MainUtils()

        self.models = {
            'XGBClassifier': XGBClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'SVC': SVC(),
            'RandomForestClassifier': RandomForestClassifier()
        }   
    def evaluate_models(self, X, y, models):
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            report = {}

            for i in range(len(list(models))):
                model=list(models.value())[i]
                model.fit(X_train, y_train)  
                y_pred = model.predict(X_test)  
                train_model_score = accuracy_score(y_train, y_train_pred) 
                test_model_score=accuracy_score(y_test, y_test_pred)

                report[list(models.keys())[i]] = test_model_score

            return report

        except Exception as e:
            raise CustomException(e, sys)   
    def get_best_model(self,
                         X_train:np.array,
                         y_test:np.array,
                         x_test:np.array,
                         y_test:np.array):
        try:
            model_report: dict=self.evaluate_models(
                X_train=X_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test
            )
            print(model_report)
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model_object=self.models[best_model_name]

            return best_model_name,best_model_object,best_model_score
            except Exception as e:
                raise CustomException(e,sys)

    def finetune_best_model(self,
                            best_model_object=object,
                            best_model_name,
                            x_train,
                            y_train,
                            ) -> object:
        try:
            model_param_grid=self.utils.read_yaml_file(self.model_trainer_config.model_config_file_path)["model_selection"]["model"][best_model_name]["search_param_grid"]
            grid_search=GridSearchCV(
                best_model_object,param_grid=model_param_grid,cv=5,n_jobs=-1,verboose=1
            )
            grid_search.fit(X_train,y_train)
            best_params=grid_search.best_parmas_
            print("best params are:",best_params)
            finetuned_model=best_model_object.set_params(**best_params)
            return finetuned_model
        excpet Exception as e:
        raise CustomException(e,sys)

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and testing inpuy and traget feature")
            x_train,y_train,x_test,y_test=(
                 train_array[:,:-1],
                 train_array[:,-1],
                 test_array[:,:-1],
                 test_array[:,-1]
            )
            logging.info(f"Extracting model config file path")
            logging.info(f"Extracting model config file path")
            model_report:dict=self.evaluate_models(X=x_train,y=y_train,models=self.models)
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = self.finetune_best_model(
        best_model_name=best_model_name,
        best_model_object=best_model,
        X_train=x_train,
        y_train=y_train)

        best_model.fit(x_train, y_train)
        y_pred = best_model.predict(x_test)
        best_model_score = accuracy_score(y_test, y_pred)

        print(f"best model name: {best_model_name} and score: {best_model_score}")
        if best_model_score<0.5:
            raise Exception("no best model found")
        logging.info("best found model on both training and testing dataset")
        os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_path), exist_ok=True)

        self.utils.save_object(
        file_path=self.model_trainer_config.trained_model_path,
        obj=best_model
        )

        return self.model_trainer_config.trained_model_path
        except Exception as e:
            raise CustomException(e,sys)

