from src.policy.decision import choose_action,DecisionConfig
def test_policy(): assert choose_action({'No E-Mail':1,'Womens E-Mail':2,'Mens E-Mail':1.5},DecisionConfig({'No E-Mail':0,'Womens E-Mail':.02,'Mens E-Mail':.02}))['recommended_action']=='Womens E-Mail'
