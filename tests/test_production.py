from pathlib import Path
import json
from src.decision_engine.engine import DecisionEngine
ROOT=Path(__file__).parents[1]
def c(): return {'recency':10,'history_segment':'2) $100 - $200','history':142.44,'mens':1,'womens':0,'zip_code':'Suburban','newbie':0,'channel':'Phone'}
def test_approved(): assert json.loads((ROOT/'models/production/model_registry.json').read_text())['status']=='APPROVED'
def test_inference():
 d=DecisionEngine(str(ROOT/'models/production')).predict(c()); assert d['recommended_action'] in ['No E-Mail','Womens E-Mail','Mens E-Mail']; assert all(0<=v<=1 for v in d['probabilities'].values())
def test_api_has_no_training():
 t=(ROOT/'api/main.py').read_text(); assert '/train' not in t and 'GridSearchCV' not in t and 'Optuna' not in t
