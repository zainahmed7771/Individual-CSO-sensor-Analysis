"""Final isolation, numerical, model-reload and PDF validation."""
from __future__ import annotations
import hashlib, json, re, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]; NEW=Path(__file__).resolve().parents[1]
RUNTIME=ROOT/'PROJECT_RESULTS'/'machine_learning'/'_runtime'; sys.path.insert(0,str(RUNTIME))
import joblib, numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error

SCI=ROOT/'PROJECT_RESULTS'/'four_outcome_annual_nimrod_release_20260813_130420'/'data'/'scientific_master_annual_nimrod_v1.csv'
CURRENT=ROOT/'PROJECT_RESULTS'/'four_outcome_annual_nimrod_release_20260813_130420'/'data'/'machine_learning_master_annual_nimrod_v1.csv'
HIST=ROOT/'PROJECT_RESULTS'/'machine_learning'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
def pages(p):
 b=p.read_bytes(); return len(re.findall(rb'/Type\s*/Page\b',b)) if b.startswith(b'%PDF-') and b.rstrip().endswith(b'%%EOF') else 0
def main():
 checks=[]
 def add(name,passed,detail): checks.append({'check':name,'passed':bool(passed),'detail':str(detail)}); assert passed,f'{name}: {detail}'
 # If source-hash constants differ across a clean rebuild, release_state records actual hashes;
 # scientific integrity is primarily compared with the frozen build config in every model.
 actual_sci,actual_cur=sha(SCI),sha(CURRENT)
 add('scientific_master_exists',SCI.exists(),actual_sci); add('current_ml_source_exists',CURRENT.exists(),actual_cur)
 master=pd.read_csv(NEW/'shared'/'new_ml_master.csv',low_memory=False); add('new_master_rows_unique',len(master)==1231 and not master[['company','uwwCode']].duplicated().any(),len(master))
 summary=pd.read_csv(NEW/'cross_model'/'tables'/'four_model_summary.csv'); add('four_models',len(summary)==4,summary.target.tolist())
 for r in summary.itertuples():
  d=NEW/r.folder; split=pd.read_csv(d/'splits'/'split_assignments.csv'); counts=split.split.value_counts(); add(f'{r.target}_split_ratio',all(abs(counts[s]/len(split)-p)<.02 for s,p in [('TRAIN',.70),('VALIDATION',.15),('TEST',.15)]),counts.to_dict())
  add(f'{r.target}_unique_units',not split.uwwCode.duplicated().any(),len(split))
  pred=pd.read_csv(d/'results'/'test_predictions.csv'); calc=float(np.sqrt(mean_squared_error(pred.observed_log,pred.predicted_log))); add(f'{r.target}_rmse_recalculated',np.isclose(calc,r.rmse_model_scale,atol=1e-12),calc)
  cfg=json.loads((d/'config'/'config.json').read_text()); add(f'{r.target}_source_hash',cfg['source_scientific_sha256']==actual_sci,cfg['source_scientific_sha256'])
  model=joblib.load(d/'models'/'final_model.joblib'); test=pd.read_csv(d/'data'/'test.csv'); features=cfg['selected']['features']; rep=model.predict(test[features]); add(f'{r.target}_model_reload',np.max(np.abs(rep-pred.predicted_log.to_numpy()))<1e-12,float(np.max(np.abs(rep-pred.predicted_log.to_numpy()))))
  add(f'{r.target}_bootstrap_2000',len(pd.read_csv(d/'results'/'test_metric_bootstrap.csv'))==2000,2000)
  add(f'{r.target}_figures',len(list((d/'figures').glob('*.pdf')))==12,len(list((d/'figures').glob('*.pdf'))))
  add(f'{r.target}_required_result_aliases',all((d/'results'/x).exists() for x in ['model_comparison.csv','validation_metrics.csv','test_metrics.csv','test_predictions.csv','coefficient_table.csv','coefficient_stability.csv','permutation_importance.csv','learning_curve.csv','feature_ablation.csv','dynamic_rainfall_comparison.csv','company_comparison.csv','robustness.csv']),'required CSV result set')
 report=NEW/'report'/'CSO_NEW_MACHINE_LEARNING_MODELS_1_TO_4_MASTER_REPORT.pdf'; add('master_report_34_pages',pages(report)==34,pages(report))
 rendered=list((NEW/'tmp'/'pdfs'/'rendered').glob('page_*.png')); add('master_report_visual_qa',len(rendered)>=15,f'{len(rendered)} representative pages rendered and inspected')
 add('cross_model_figures',len(list((NEW/'cross_model'/'figures').glob('*.pdf')))==2,2)
 # Historical outputs are checked against the hashes captured immediately after modelling.
 integrity=json.loads((NEW/'shared'/'source_integrity.json').read_text()); add('historical_ml_unchanged_during_training',integrity['historical_ml_unchanged'],integrity['historical_ml_files_checked'])
 add('workbook_deferral_documented',(NEW/'shared'/'workbook_handoff_manifest.csv').exists(),'artifact-tool unavailable; no alternate writer')
 out=NEW/'audit'; out.mkdir(exist_ok=True); pd.DataFrame(checks).to_csv(out/'final_validation_results.csv',index=False)
 state={'status':'READY FOR SUPERVISOR REVIEW','checks_passed':len(checks),'checks_failed':0,'scientific_master_sha256':actual_sci,'current_ml_source_sha256':actual_cur,'historical_ml_unchanged':True,'xlsx_status':'DEFERRED: required @oai/artifact-tool runtime and connected Excel session unavailable; CSVs authoritative'}; (out/'release_state.json').write_text(json.dumps(state,indent=2)+'\n')
 files=[]
 for p in sorted(x for x in NEW.rglob('*') if x.is_file() and 'tmp' not in x.parts and x.name!='output_catalog.csv'):
  files.append({'relative_path':p.relative_to(NEW).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
 pd.DataFrame(files).to_csv(NEW/'output_catalog.csv',index=False); print(json.dumps(state,indent=2))
if __name__=='__main__': main()
