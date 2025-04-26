
import subprocess
import sys

# Install requirements
subprocess.check_call([sys.executable, "-m", "pip", "install", "sagemaker"])

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score    
import sklearn 
import joblib 
import pandas as pd
import os
import argparse
import numpy as np
import logging
import json
import sys
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.sklearn import SKLearn

def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, 'model.joblib'))
    return clf

if __name__ == '__main__':

    print("[Info] Extracting arguments from the command line")
    parser = argparse.ArgumentParser()
    parser.add_argument('n_estimators', type=int, default = 100, help='number of trees in the forest')
    parser.add_argument('--random_state', type=int, default = 0, help='random state')

    parser.add_argument('--train', type =str, default = os.environ.get('SM_CHANNEL_TRAIN'), help='training data location')
    parser.add_argument('--test', type = str, default = os.environ.get('SM_CHANNEL_TEST'), help ='test data location')
    parser.add_argument('--model-dir', type = str, default =os.environ.get('SM_MODEL_DIR') , help ='model location')
    parser.add_argument('--train-file', type = str, default = 'trainX-v1.csv', help ='train file name')
    parser.add_argument('--test-file', type = str, default = 'testX-v1.csv', help ='test file name')
    

    args, _ = parser.parse_known_args()
    print(f"Arguments: {args}")

    print(f"sklearn_version: {sklearn.__version__}")
    

    print(f"reading training data")
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    print(f"train_df shape: {train_df.shape}")
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))
    print(f"test_df shape: {test_df.shape}")

    features = list(train_df.columns)
    lable = features.pop(-1)
    x_train = train_df[features]
    x_test = test_df[features]
    y_train = train_df[lable]
    y_test = test_df[lable]
    
    print("Buidling training and test datasets")
    
    print(f'training random forest classfifier')
    model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state)
    model.fit(x_train, y_train)
    print(f"model training complete")

    model_path = os.path.join(args.model_dir, 'model.joblib')
    print(f"Saving model to {model_path}")
    joblib.dump(model, model_path)  

    y_pred = model.predict(x_test)
    print(f"model prediction complete")
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"test accuracy: {test_accuracy}")
    print(f"test classification report: {classification_report(y_test, y_pred)}")
    print(f"test confusion matrix: {confusion_matrix(y_test, y_pred)}")
    print(f"test precision: {precision_score(y_test, y_pred, average='weighted')}") 


    


