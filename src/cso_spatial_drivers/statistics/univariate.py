from __future__ import annotations
import numpy as np
import statsmodels.api as sm

def fit_log_outcome(y, x) -> dict:
    y=np.asarray(y,float); x=np.asarray(x,float); ok=np.isfinite(y)&(y>0)&np.isfinite(x)
    model=sm.OLS(np.log(y[ok]),sm.add_constant(x[ok])).fit(cov_type="HC3")
    return {"n":int(ok.sum()),"slope":float(model.params[1]),"hc3_se":float(model.bse[1]),"p_value":float(model.pvalues[1]),"r_squared":float(model.rsquared)}
