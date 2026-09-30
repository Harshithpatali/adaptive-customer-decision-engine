from dataclasses import dataclass
@dataclass(frozen=True)
class DecisionConfig: treatment_costs:dict; version:str='value_max_v1'
def choose_action(action_values,cfg):
 utility={a:float(v)-cfg.treatment_costs.get(a,0) for a,v in action_values.items()}; action=max(utility,key=utility.get); vals=sorted(utility.values(),reverse=True); return {'recommended_action':action,'utility':utility,'decision_margin':vals[0]-vals[1],'policy_version':cfg.version}
