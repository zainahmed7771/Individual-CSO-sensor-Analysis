from __future__ import annotations
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def fit_locked_demo(X_train,y_train,X_validation,y_validation,X_test,y_test):
    candidates=[]
    for alpha in (.1,1.0,10.0):
        m=Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",Ridge(alpha=alpha))])
        m.fit(X_train,y_train); candidates.append((mean_squared_error(y_validation,m.predict(X_validation))**.5,alpha,m))
    _,alpha,_=min(candidates,key=lambda z:z[0])
    X_fit=np.vstack([X_train,X_validation]); y_fit=np.r_[y_train,y_validation]
    model=Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",Ridge(alpha=alpha))]).fit(X_fit,y_fit)
    dummy=DummyRegressor().fit(X_fit,y_fit); pred=model.predict(X_test); dp=dummy.predict(X_test)
    rmse=mean_squared_error(y_test,pred)**.5; dr=mean_squared_error(y_test,dp)**.5
    return {"alpha":alpha,"rmse":rmse,"dummy_rmse":dr,"improvement_pct":100*(dr-rmse)/dr,"r2":r2_score(y_test,pred),"predictions":pred}
