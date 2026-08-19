import pandas as pd
from cso_spatial_drivers.cleaning.events import clean_events
def test_duration_minutes_and_end_order():
 d=pd.DataFrame([{'company':'x','permit_number':'1','location_name':'x','start_time':'2024-01-01T00:00Z','stop_time':'2024-01-01T01:30Z'},{'company':'x','permit_number':'2','location_name':'x','start_time':'2024-01-02T02:00Z','stop_time':'2024-01-02T01:00Z'}]); x=clean_events(d); assert len(x)==1 and x.duration_minutes.iloc[0]==90
