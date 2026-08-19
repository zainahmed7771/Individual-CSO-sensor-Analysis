import pandas as pd, pytest
from cso_spatial_drivers.predictors.join import join_predictors
def test_join_preserves_rows_and_rejects_duplicates():
 o=pd.DataFrame({'company':['x','x'],'uwwCode':['a','a'],'y':[1,2]}); p=pd.DataFrame({'company':['x'],'uwwCode':['a'],'v':[3]}); assert len(join_predictors(o,p))==2
 with pytest.raises(ValueError): join_predictors(o,pd.concat([p,p]))
