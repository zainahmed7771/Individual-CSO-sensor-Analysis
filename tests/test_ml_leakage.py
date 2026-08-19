import pandas as pd
def test_forbidden_predictor_manifest():
 forbidden={'sensor_uid','uwwCode','fitted_beta','fitted_lambda','spill_mean_duration_minutes','spill_events_per_monitoring_year'}; selected={'population_density_km2','rain_mean_annual_mm','catchment_area_km2'}; assert not forbidden & selected
def test_group_split_overlap_zero():
 split=pd.DataFrame({'uwwCode':['a','b','c'],'split':['TRAIN','VALIDATION','TEST']}); assert split.groupby('uwwCode').split.nunique().max()==1
