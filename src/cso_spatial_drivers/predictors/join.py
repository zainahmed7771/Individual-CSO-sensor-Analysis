from __future__ import annotations
import pandas as pd

def join_predictors(outcomes: pd.DataFrame, predictors: pd.DataFrame) -> pd.DataFrame:
    keys=["company","uwwCode"]
    if predictors.duplicated(keys).any(): raise ValueError("predictor keys are not unique")
    result=outcomes.merge(predictors,on=keys,how="left",validate="many_to_one")
    if len(result)!=len(outcomes): raise AssertionError("predictor join changed row count")
    return result
