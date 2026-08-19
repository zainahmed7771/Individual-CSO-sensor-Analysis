import pandas as pd
from cso_spatial_drivers.spatial.assignment import nearest_same_company
def test_same_company_assignment():
 s=pd.DataFrame([{'sensor_uid':'a','company':'x','easting':0,'northing':0}]); w=pd.DataFrame([{'uwwCode':'wrong','company':'y','easting':0,'northing':0},{'uwwCode':'right','company':'x','easting':10,'northing':0}]); assert nearest_same_company(s,w).uwwCode.iloc[0]=='right'
