"""Create compact cross-model comparison figures from frozen final results."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]; NEW=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'PROJECT_RESULTS'/'machine_learning'/'_runtime'))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

d=pd.read_csv(NEW/'cross_model'/'tables'/'four_model_summary.csv')
out=NEW/'cross_model'/'figures'; out.mkdir(parents=True,exist_ok=True)
labels=['Beta','Scale','Duration','Frequency']; blue='#174A6E'; orange='#D97706'; grey='#64748B'

fig,axs=plt.subplots(1,2,figsize=(10,4.5))
x=np.arange(4); w=.36
axs[0].bar(x-w/2,d.dummy_rmse_model_scale,w,label='Dummy',color=grey)
axs[0].bar(x+w/2,d.rmse_model_scale,w,label='Selected model',color=blue)
axs[0].set_xticks(x,labels); axs[0].set_ylabel('RMSE on model scale'); axs[0].set_title('Within-target locked-test RMSE'); axs[0].legend(frameon=False)
axs[1].bar(x,d.rmse_improvement_pct,color=[blue,blue,blue,orange]); axs[1].axhline(0,color=grey,lw=.8); axs[1].set_xticks(x,labels); axs[1].set_ylabel('RMSE improvement over Dummy (%)'); axs[1].set_title('Baseline-relative predictive gain')
fig.suptitle('Four-model final scorecard',weight='bold',color=blue); fig.tight_layout(); fig.savefig(out/'four_model_performance.pdf',bbox_inches='tight'); plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(10,4.5))
axs[0].bar(x,d.dynamic_rainfall_delta_rmse,color=[blue if v<0 else orange for v in d.dynamic_rainfall_delta_rmse]); axs[0].axhline(0,color=grey,lw=.8); axs[0].set_xticks(x,labels); axs[0].set_ylabel('Delta test RMSE'); axs[0].set_title('Annual NIMROD value added')
axs[1].bar(x,d.company_delta_rmse,color=[blue if v<0 else orange for v in d.company_delta_rmse]); axs[1].axhline(0,color=grey,lw=.8); axs[1].set_xticks(x,labels); axs[1].set_ylabel('Delta test RMSE'); axs[1].set_title('Company value added')
fig.suptitle('Pre-specified contextual sensitivities (negative is better)',weight='bold',color=blue); fig.tight_layout(); fig.savefig(out/'dynamic_rainfall_and_company_value.pdf',bbox_inches='tight'); plt.close(fig)

for r in d.itertuples():
 p=NEW/r.folder/'results'
 source=p/'validation_model_comparison.csv'
 pd.read_csv(source).to_csv(p/'validation_metrics.csv',index=False)
 pd.read_csv(source).to_csv(p/'model_comparison.csv',index=False)
print('cross-model figures and exact CSV handoffs created')
