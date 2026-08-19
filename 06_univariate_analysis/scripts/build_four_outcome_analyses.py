"""Build annual-only masters and four one-variable analysis packages."""

from __future__ import annotations

import hashlib, io, json, math, shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.pdfgen import canvas
from scipy import stats
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[4]
RELEASE = Path(__file__).resolve().parents[2]
BASE_MASTER = ROOT / "PROJECT_RESULTS/master/final_master_spreadsheet.xlsx"
VALIDATED_MASTER_CSV = ROOT / "PROJECT_RESULTS/master/final_master_spreadsheet_dynamic_rainfall_v1.csv"
BASE_ML = ROOT / "PROJECT_RESULTS/machine_learning/machine_learning_master.csv"
RAIN = RELEASE / "rainfall/annual_rainfall_indices_master.csv"
STATS = RELEASE / "statistical_analysis"
DATA = RELEASE / "data"
BLUE="#173F5F"; MID="#176D9C"; LIGHT="#EAF3F7"; RED="#A63A3A"; GREY="#65737E"

def pred(v,label,family,unit,increase,kind,div=1.0):
    return dict(variable=v,label=label,family=family,unit=unit,increase=increase,kind=kind,divisor=div)

PREDICTORS=[
 pred("hydro_high_productivity_pct","High-productivity hydrogeology","environmental_core","%","10 percentage points","divide",10),
 pred("hydro_moderate_productivity_pct","Moderate-productivity hydrogeology","environmental_core","%","10 percentage points","divide",10),
 pred("hydro_low_productivity_pct","Low-productivity hydrogeology","environmental_core","%","10 percentage points","divide",10),
 pred("hydro_no_groundwater_pct","Essentially no groundwater","environmental_core","%","10 percentage points","divide",10),
 pred("hydro_intergranular_flow_pct","Intergranular-flow hydrogeology","environmental_core","%","10 percentage points","divide",10),
 pred("hydro_fracture_flow_pct","Fracture-flow hydrogeology","environmental_core","%","10 percentage points","divide",10),
 pred("rain_mean_annual_mm_1991_2020","Static annual rainfall (1991-2020)","environmental_core","mm/year","100 mm/year","divide",100),
 pred("rain_winter_mean_mm_1991_2020","Static winter rainfall (1991-2020)","environmental_core","mm/winter","100 mm/winter","divide",100),
 pred("landcover_urban_pct","Urban land cover","environmental_core","%","10 percentage points","divide",10),
 pred("landcover_suburban_pct","Suburban land cover","environmental_core","%","10 percentage points","divide",10),
 pred("population_density_km2_2021","Population density (2021)","environmental_core","people/km2","doubling","log2",1),
 pred("building_pre_1973_pct","Pre-1973 building share","environmental_core","%","10 percentage points","divide",10),
 pred("terrain_mean_slope_deg","Mean terrain slope","environmental_core","degrees","1 degree","divide",1),
 pred("capacity_ratio_latest","Treatment capacity utilisation ratio","operational_core","ratio","0.1 ratio units","divide",.1),
 pred("sewershed_area_km2","WWTW catchment area","operational_core","km2","doubling","log2",1),
 pred("continuous_land_cover_pct","Continuous developed-land index","landcover_extension","index","10 index points","divide",10),
 pred("rain_ann_prcptot_mm","Annual PRCPTOT","annual_nimrod","mm/year","100 mm/year","divide",100),
 pred("rain_ann_rx1day_mm","Annual Rx1day","annual_nimrod","mm","10 mm","divide",10),
 pred("rain_ann_rx5day_mm","Annual Rx5day","annual_nimrod","mm","25 mm","divide",25),
 pred("rain_ann_sdii_mm_per_wet_day","Annual SDII","annual_nimrod","mm/wet day","1 mm/wet day","divide",1),
 pred("rain_ann_r10_days","Annual R10mm days","annual_nimrod","days/year","5 days/year","divide",5),
 pred("rain_ann_r20_days","Annual R20mm days","annual_nimrod","days/year","5 days/year","divide",5),
 pred("rain_ann_cwd_days","Annual consecutive wet days","annual_nimrod","days","1 day","divide",1),
 pred("rain_ann_cdd_days","Annual consecutive dry days","annual_nimrod","days","5 days","divide",5),
]
META={p['variable']:p for p in PREDICTORS}; NAMES=[p['variable'] for p in PREDICTORS]
OUTCOMES={
 "beta":dict(field="fitted_beta",label="Fitted beta",transform="log",strict=True,physical="long-duration tail shape"),
 "scale":dict(field="fitted_lambda",label="Inverse-timescale lambda",transform="log",strict=True,physical="inverse characteristic timescale"),
 "spill_duration":dict(field="spill_mean_duration_minutes",label="Mean spill duration",transform="log",strict=False,physical="observed mean duration per spill"),
 "spill_count":dict(field="spill_events_per_monitoring_year",label="Annualised spill-event frequency",transform="log",strict=False,physical="events per valid monitoring year"),
}

def sha(path):
 d=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''): d.update(b)
 return d.hexdigest()

def truthy(s):
 return pd.to_numeric(s,errors='coerce').eq(1)|s.astype(str).str.lower().isin(['true','yes'])

def transform_x(s,p):
 x=pd.to_numeric(s,errors='coerce')
 if p['kind']=='log2': return np.log2(x.where(x>0))
 return x/p['divisor']

def build_master():
 # The workbook may be open/locked.  Its validated CSV release contains the
 # exact original columns plus the earlier experimental dynamic-rainfall block.
 # For this annual-only release, remove that experimental block and preserve all
 # established master columns byte/numerically unchanged.
 snapshot=pd.read_csv(VALIDATED_MASTER_CSV,low_memory=False)
 deprecated=[c for c in snapshot if c.startswith('dynrain_')]
 base=snapshot.drop(columns=deprecated)
 assert len(base)==1444 and not base.duplicated(['company','uwwCode']).any()
 orig=base.copy(deep=True)
 dev=pd.to_numeric(base.landcover_urban_pct,errors='coerce')+pd.to_numeric(base.landcover_suburban_pct,errors='coerce')
 base['continuous_land_cover_pct']=np.where(dev>0,100*pd.to_numeric(base.landcover_urban_pct,errors='coerce')/dev,np.nan)
 rain=pd.read_csv(RAIN)
 rain_cols=[c for c in rain if c.startswith('rain_ann_')]
 base=base.merge(rain[['company','uwwCode',*rain_cols]],on=['company','uwwCode'],how='left',validate='one_to_one',sort=False)
 assert len(base)==len(orig) and base[['company','uwwCode']].equals(orig[['company','uwwCode']])
 for c in orig: assert base[c].equals(orig[c]) or np.allclose(pd.to_numeric(base[c],errors='coerce'),pd.to_numeric(orig[c],errors='coerce'),equal_nan=True)
 DATA.mkdir(parents=True,exist_ok=True)
 base.to_csv(DATA/'scientific_master_annual_nimrod_v1.csv',index=False)
 pd.DataFrame([{'authoritative_xlsx':str(BASE_MASTER.relative_to(ROOT)),'sha256':sha(BASE_MASTER),'status':'unchanged; artifact-tool unavailable',
                'versioned_csv':'data/scientific_master_annual_nimrod_v1.csv','rows':len(base),'rainfall_matches':int(base.rain_ann_n_valid_years.notna().sum()),
                'all_original_columns_unchanged':True}]).to_csv(DATA/'scientific_master_workbook_handoff_manifest.csv',index=False)
 ml=pd.read_csv(BASE_ML,low_memory=False)
 drop=[c for c in ml if c.startswith('dynrain_') or c in ['has_dynamic_rainfall','dynamic_rainfall_quality']]
 ml=ml.drop(columns=drop)
 ml=ml.merge(rain[['company','uwwCode',*rain_cols]],on=['company','uwwCode'],how='left',validate='one_to_one',sort=False)
 ml['has_annual_nimrod_rainfall']=ml.rain_ann_n_valid_years.notna()
 ml['annual_nimrod_quality']=ml.rain_ann_quality_flag
 ml.to_csv(DATA/'machine_learning_master_annual_nimrod_v1.csv',index=False)
 assert len(ml)==1231
 return base,orig,ml

def fit_one(frame,outcome,p,sample):
 x=transform_x(frame[p['variable']],p); y=pd.to_numeric(frame[outcome['field']],errors='coerce')
 mask=x.notna()&np.isfinite(x)&y.gt(0)&np.isfinite(y)
 if sample=='strict': mask &= truthy(frame.eligible_for_beta_analysis)
 d=pd.DataFrame({'x':x[mask],'y':y[mask]}); ly=np.log(d.y)
 model=sm.OLS(ly,sm.add_constant(d.x,has_constant='add')).fit(cov_type='HC3')
 b=float(model.params['x']); lo,hi=map(float,model.conf_int().loc['x']); pr=stats.pearsonr(d.x,ly); sr=stats.spearmanr(d.x,d.y)
 return dict(variable=p['variable'],readable_label=p['label'],family=p['family'],sample=sample,n=len(d),meaningful_increase=p['increase'],
  transformation=p['kind'],unit=p['unit'],coefficient_log_outcome=b,HC3_SE=float(model.bse['x']),CI_lower=lo,CI_upper=hi,
  effect_pct=100*np.expm1(b),effect_CI_lower_pct=100*np.expm1(lo),effect_CI_upper_pct=100*np.expm1(hi),
  Pearson_r=float(pr.statistic),Pearson_p=float(pr.pvalue),Spearman_rho=float(sr.statistic),Spearman_p=float(sr.pvalue),
  R_squared=float(model.rsquared),adjusted_R_squared=float(model.rsquared_adj),RAW_REGRESSION_P=float(model.pvalues['x']))

def analyse(frame,key):
 outcome=OUTCOMES[key]; samples=['strict','broad'] if outcome['strict'] else ['primary']
 rows=[fit_one(frame,outcome,p,s) for s in samples for p in PREDICTORS]
 res=pd.DataFrame(rows); res['FDR_Q']=np.nan
 for (sample,family),ix in res.groupby(['sample','family']).groups.items():
  res.loc[ix,'FDR_Q']=multipletests(res.loc[ix,'RAW_REGRESSION_P'],method='fdr_bh')[1]
 res['direction']=np.where(res.coefficient_log_outcome>0,'positive','negative')
 res['evidence_status']=np.select([res.FDR_Q<.05,res.RAW_REGRESSION_P<.05],['FDR-supported','nominal only'],default='no clear evidence')
 res['interpretation']=res.apply(lambda r:f"A {r.meaningful_increase} increase in {r.readable_label} is associated with an estimated {abs(r.effect_pct):.1f}% {'higher' if r.effect_pct>0 else 'lower'} {outcome['label'].lower()}; unadjusted and non-causal.",axis=1)
 return res

def plot_one(frame,key,row,path):
 outcome=OUTCOMES[key]; p=META[row.variable]; xraw=pd.to_numeric(frame[p['variable']],errors='coerce'); x=transform_x(frame[p['variable']],p); y=pd.to_numeric(frame[outcome['field']],errors='coerce')
 mask=x.notna()&y.gt(0)&np.isfinite(x)&np.isfinite(y)
 if row.sample=='strict': mask &= truthy(frame.eligible_for_beta_analysis)
 model=sm.OLS(np.log(y[mask]),sm.add_constant(x[mask],has_constant='add')).fit(cov_type='HC3')
 raw_grid=np.linspace(xraw[mask].quantile(.01),xraw[mask].quantile(.99),200)
 grid=transform_x(pd.Series(raw_grid),p); pred=model.get_prediction(sm.add_constant(grid,has_constant='add')).summary_frame()
 fig,ax=plt.subplots(figsize=(8.27,6.0)); ax.scatter(xraw[mask],y[mask],s=13,alpha=.48,color='#2B82B0',edgecolors='none')
 ax.fill_between(raw_grid,np.exp(pred.mean_ci_lower),np.exp(pred.mean_ci_upper),color='#D7EAF3')
 ax.plot(raw_grid,np.exp(pred['mean']),color=BLUE,lw=2); ax.set_yscale('log'); ax.grid(alpha=.16)
 ax.set_title(f"{outcome['label']} vs {p['label']}",loc='left',fontsize=15,weight='bold',color=BLUE)
 ax.set_xlabel(f"{p['label']} ({p['unit']})"); ax.set_ylabel(outcome['label'])
 fig.text(.08,.035,f"n={int(row.n)} | effect {row.effect_pct:+.1f}% per {row.meaningful_increase} | 95% CI [{row.effect_CI_lower_pct:+.1f}, {row.effect_CI_upper_pct:+.1f}]% | p={row.RAW_REGRESSION_P:.3g} | q={row.FDR_Q:.3g} | R2={row.R_squared:.3f}",fontsize=8,color='#263238')
 fig.text(.08,.012,row.interpretation,fontsize=7.5,color=GREY); fig.tight_layout(rect=[0,.075,1,1]); fig.savefig(path); plt.close(fig)

def text_page(title,lines,path):
 c=canvas.Canvas(str(path),pagesize=A4); w,h=A4; c.setFillColor(colors.HexColor(BLUE)); c.rect(0,h-20*mm,w,20*mm,fill=1,stroke=0)
 c.setFillColor(colors.white); c.setFont('Helvetica-Bold',10); c.drawString(17*mm,h-12*mm,'CSO FOUR-OUTCOME ANNUAL NIMROD RELEASE')
 c.setFillColor(colors.HexColor(BLUE)); c.setFont('Helvetica-Bold',19); c.drawString(17*mm,h-33*mm,title); y=h-45*mm
 styles=getSampleStyleSheet(); style=ParagraphStyle('b',parent=styles['BodyText'],fontSize=9.2,leading=13,textColor=colors.HexColor('#263238'))
 for line in lines:
  p=Paragraph(line,style); _,ph=p.wrap(w-34*mm,y-18*mm); y-=ph; p.drawOn(c,17*mm,y); y-=4*mm
 c.save()

def package(frame,key,res):
 d=STATS/f"{key}_parameter_analysis" if key in ['beta','scale'] else STATS/f"{key}_analysis"
 figs=d/'individual_figures'; figs.mkdir(parents=True,exist_ok=True)
 primary='strict' if OUTCOMES[key]['strict'] else 'primary'; rr=res[res['sample']==primary].copy()
 for i,row in enumerate(rr.itertuples(),1): plot_one(frame,key,row,figs/f"{i:02d}_{row.variable}.pdf")
 res.to_csv(d/f"{key}_analysis_results.csv",index=False)
 for sample,g in res.groupby('sample'): g.to_csv(d/f"{sample}_univariate_summary.csv",index=False)
 pd.DataFrame([{'requested_workbook':f'{key}_analysis_results.xlsx','status':'deferred - required artifact-tool runtime unavailable','source_csv':f'{key}_analysis_results.csv'}]).to_csv(d/'workbook_handoff_manifest.csv',index=False)
 intro=d/'_intro.pdf'; summary=d/'_summary.pdf'
 text_page(f"{OUTCOMES[key]['label']} one-variable analysis",[
  f"Target: <b>{OUTCOMES[key]['physical']}</b>. Response transformation: log({OUTCOMES[key]['field']}); all eligible values are strictly positive.",
  "Each predictor is fitted separately using OLS with HC3 robust uncertainty. Effects are exact percentage differences for frozen meaningful increments. Raw p-values and Benjamini-Hochberg FDR q-values are both reported.",
  f"Primary sample: {primary}, n range {rr.n.min()}-{rr.n.max()}. Annual NIMROD indices are an independent rainfall FDR family. All associations are unadjusted and non-causal."],intro)
 top=rr.sort_values('FDR_Q').head(12)
 lines=[f"<b>{r.readable_label}</b>: {r.effect_pct:+.1f}% [{r.effect_CI_lower_pct:+.1f}, {r.effect_CI_upper_pct:+.1f}]%; p={r.RAW_REGRESSION_P:.3g}; q={r.FDR_Q:.3g}; {r.evidence_status}." for r in top.itertuples()]
 text_page("Results summary",lines,summary)
 writer=PdfWriter();
 for pth in [intro,*sorted(figs.glob('*.pdf')),summary]: writer.append(pth)
 combined=d/f"{key}_analysis_figures.pdf"; writer.write(combined)
 audit=[f"{OUTCOMES[key]['label'].upper()} ANALYSIS AUDIT",f"response=log({OUTCOMES[key]['field']})",f"primary_sample={primary}",f"predictors={len(PREDICTORS)}",f"individual_vector_pdfs={len(list(figs.glob('*.pdf')))}",f"combined_pages={len(PdfReader(combined).pages)}","HC3=true","FDR=BH within established families plus annual_nimrod","causal_claims=false"]
 (d/f"{key}_analysis_audit.txt").write_text('\n'.join(audit)+'\n',encoding='utf-8')
 return d

def count_sensitivity(frame):
 rows=[]; count=pd.to_numeric(frame.spill_event_count,errors='coerce'); exposure=pd.to_numeric(frame.monitoring_years,errors='coerce')
 overdisp=float(count[count.notna()&exposure.gt(0)].var()/count[count.notna()&exposure.gt(0)].mean())
 for p in PREDICTORS:
  x=transform_x(frame[p['variable']],p); mask=count.ge(0)&exposure.gt(0)&x.notna()&np.isfinite(x)
  X=sm.add_constant(pd.DataFrame({'x':x[mask]}),has_constant='add')
  try:
   m=NegativeBinomial(count[mask],X,offset=np.log(exposure[mask])).fit(disp=False,maxiter=200)
   b=float(m.params['x']); lo,hi=map(float,m.conf_int().loc['x'])
   rows.append({'variable':p['variable'],'n':int(mask.sum()),'model':'Negative Binomial with log(exposure) offset','overdispersion_variance_to_mean':overdisp,'log_rate_coefficient':b,'incidence_rate_ratio':math.exp(b),'IRR_CI_lower':math.exp(lo),'IRR_CI_upper':math.exp(hi),'p':float(m.pvalues['x']),'alpha':float(m.params.get('alpha',np.nan)),'failure':''})
  except Exception as e: rows.append({'variable':p['variable'],'n':int(mask.sum()),'model':'Negative Binomial','failure':str(e)})
 pd.DataFrame(rows).to_csv(STATS/'spill_count_analysis/count_model_sensitivity.csv',index=False)

def reproduce(res,key):
 prior=pd.read_csv(ROOT/f"dynamic_rainfall/statistical_results/{key}_dynamic_rainfall_univariate_tables/strict_univariate_summary.csv")
 new=res[res['sample']=='strict']; old_names=NAMES[:16]; rows=[]
 for metric_new,metric_old in [('coefficient_log_outcome','coefficient_log_parameter'),('HC3_SE','HC3_SE'),('RAW_REGRESSION_P','RAW_REGRESSION_P'),('FDR_Q','FDR_Q')]:
  a=new[new.variable.isin(old_names)].set_index('variable')[metric_new]; b=prior[prior.variable.isin(old_names)].set_index('variable')[metric_old]; diff=(a-b).abs()
  rows.append({'outcome':key,'metric':metric_new,'rows':len(diff),'max_absolute_difference':diff.max(),'passed':bool(diff.max()<1e-10)})
 return rows

def main():
 plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
 master,orig,ml=build_master(); all_results={}; repro=[]
 for key in OUTCOMES:
  res=analyse(master,key); all_results[key]=res; package(master,key,res)
  if key in ['beta','scale']: repro+=reproduce(res,key)
 count_sensitivity(master)
 pd.DataFrame(repro).to_csv(STATS/'old_beta_scale_reproduction.csv',index=False)
 assert pd.DataFrame(repro).passed.all()
 print('MASTER',len(master),'unique',not master.duplicated(['company','uwwCode']).any(),'annual rainfall',master.rain_ann_n_valid_years.notna().sum())
 for k,r in all_results.items():
  s='strict' if OUTCOMES[k]['strict'] else 'primary'; q=r[r['sample']==s]
  print(k,'n range',q.n.min(),q.n.max(),'FDR supported',int((q.FDR_Q<.05).sum()))
 print('BETA/SCALE OLD RESULT REPRODUCTION PASS')

if __name__=='__main__': main()
