"""Offline retraining entry point: only the approved production algorithms are allowed.
Response: XGBoost. Uplift: T-Learner. Revenue: LightGBM.
No candidate-model search is performed here.
"""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMRegressor

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw/hillstrom_raw.csv'
OUT=ROOT/'models/production'
FEATURES=['recency','history_segment','history','mens','womens','zip_code','newbie','channel']
NUM=['recency','history','mens','womens','newbie']
CAT=['history_segment','zip_code','channel']

def preprocess(include_segment=True):
    cats=CAT+(['segment'] if include_segment else [])
    return ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),NUM),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),cats)])

def main():
    df=pd.read_csv(RAW)
    # Offline training only. Preserve treatment assignment and never use post-treatment outcomes as features.
    x=df[FEATURES].copy(); x['segment']=df['segment']
    y=df['conversion'].astype(int)
    response=Pipeline([('pre',preprocess(True)),('model',XGBClassifier(n_estimators=80,max_depth=7,min_child_weight=7,learning_rate=.03,colsample_bytree=.8,subsample=.9,eval_metric='logloss',random_state=42,n_jobs=-1))])
    response.fit(x,y)
    joblib.dump(response,OUT/'response_model.joblib')
    print('Saved selected XGBoost response model.')

if __name__=='__main__': main()
