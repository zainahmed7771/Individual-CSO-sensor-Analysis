"""Build beta-log(scale), full correlation and cross-outcome products."""
from __future__ import annotations
import io, json, math
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
from pypdf import PdfWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.pdfgen import canvas

ROOT=Path(__file__).resolve().parents[4]; RELEASE=Path(__file__).resolve().parents[2]
STATS=RELEASE/'statistical_analysis'; DATA=RELEASE/'data'; BLUE='#173F5F'; MID='#176D9C'; GREY='#65737E'
ANNUAL=['rain_ann_prcptot_mm','rain_ann_rx1day_mm','rain_ann_rx5day_mm','rain_ann_sdii_mm_per_wet_day','rain_ann_r10_days','rain_ann_r20_days','rain_ann_cwd_days','rain_ann_cdd_days']
ENV=['hydro_high_productivity_pct','hydro_moderate_productivity_pct','hydro_low_productivity_pct','hydro_no_groundwater_pct','hydro_intergranular_flow_pct','hydro_fracture_flow_pct','rain_mean_annual_mm_1991_2020','rain_winter_mean_mm_1991_2020',*ANNUAL,'landcover_urban_pct','landcover_suburban_pct','continuous_land_cover_pct','population_density_km2_2021','building_pre_1973_pct','terrain_mean_slope_deg','capacity_ratio_latest','sewershed_area_km2']

def truthy(s): return pd.to_numeric(s,errors='coerce').eq(1)|s.astype(str).str.lower().isin(['true','yes'])
def beta_scale(frame,sample):
 b=pd.to_numeric(frame.fitted_beta,errors='coerce'); l=pd.to_numeric(frame.fitted_lambda,errors='coerce'); mask=b.gt(0)&l.gt(0)&np.isfinite(b)&np.isfinite(l)
 if sample=='strict': mask&=truthy(frame.eligible_for_beta_analysis)
 x=np.log(l[mask]); y=np.log(b[mask]); m=sm.OLS(y,sm.add_constant(x,has_constant='add')).fit(cov_type='HC3'); lo,hi=m.conf_int().loc[x.name if x.name in m.params.index else 'fitted_lambda'] if False else m.conf_int().iloc[1]
 pr=stats.pearsonr(x,y); sr=stats.spearmanr(b[mask],l[mask])
 return dict(sample=sample,n=int(mask.sum()),slope=float(m.params.iloc[1]),HC3_SE=float(m.bse.iloc[1]),CI_lower=float(lo),CI_upper=float(hi),p=float(m.pvalues.iloc[1]),R_squared=float(m.rsquared),Pearson_log_r=float(pr.statistic),Pearson_log_p=float(pr.pvalue),Spearman_original_rho=float(sr.statistic),Spearman_original_p=float(sr.pvalue)),mask,m

def relationship(frame):
 d=STATS/'beta_scale_relationship'; d.mkdir(exist_ok=True)
 rows=[]; models={}; masks={}
 for s in ['strict','broad']:
  r,mask,m=beta_scale(frame,s); rows.append(r); models[s]=m; masks[s]=mask
 pd.DataFrame(rows).to_csv(d/'beta_scale_relationship_results.csv',index=False)
 pd.DataFrame([{'requested_workbook':'beta_scale_relationship_results.xlsx','status':'deferred - artifact-tool unavailable','source_csv':'beta_scale_relationship_results.csv'}]).to_csv(d/'workbook_handoff_manifest.csv',index=False)
 r=rows[0]; mask=masks['strict']; b=pd.to_numeric(frame.fitted_beta,errors='coerce')[mask]; l=pd.to_numeric(frame.fitted_lambda,errors='coerce')[mask]; x=np.log(l)
 grid=np.linspace(x.min(),x.max(),200); pred=models['strict'].get_prediction(sm.add_constant(grid,has_constant='add')).summary_frame()
 fig,ax=plt.subplots(figsize=(8.27,6)); ax.scatter(x,b,s=18,alpha=.55,color=MID,edgecolors='none'); ax.plot(grid,np.exp(pred['mean']),color=BLUE,lw=2); ax.fill_between(grid,np.exp(pred.mean_ci_lower),np.exp(pred.mean_ci_upper),color='#D7EAF3'); ax.set_yscale('log'); ax.grid(alpha=.18); ax.set_xlabel('log(fitted inverse-timescale lambda)'); ax.set_ylabel('Fitted beta (log axis)'); ax.set_title('Beta versus log(scale): strict joint-fit sample',loc='left',weight='bold',color=BLUE,fontsize=15); fig.text(.08,.035,f"n={r['n']} | slope={r['slope']:.3f} | 95% CI [{r['CI_lower']:.3f}, {r['CI_upper']:.3f}] | p={r['p']:.3g} | R2={r['R_squared']:.3f} | Pearson={r['Pearson_log_r']:.3f} | Spearman={r['Spearman_original_rho']:.3f}",fontsize=8); fig.text(.08,.012,'Parameter-relationship/identifiability diagnostic: beta and lambda are jointly fitted; this is not an environmental mechanism.',fontsize=7.5,color=GREY); fig.tight_layout(rect=[0,.065,1,1]); fig.savefig(d/'beta_vs_log_scale.pdf'); plt.close(fig)
 (d/'beta_scale_relationship_audit.txt').write_text('Primary: log(beta)=alpha+b*log(lambda)+error; HC3.\nStrict joint-fit-quality sample plus broad finite-positive sensitivity.\nJoint bootstrap covariance unavailable: no stored paired bootstrap draws were found.\nBeta and lambda are jointly fitted; association can reflect distribution structure and estimation trade-off.\n',encoding='utf-8')

def correlation(frame):
 d=STATS/'cross_outcome'; d.mkdir(exist_ok=True)
 corr=frame[['fitted_beta','fitted_lambda','spill_mean_duration_minutes','spill_events_per_monitoring_year',*ENV]].copy(); corr['log_scale']=np.log(pd.to_numeric(corr.fitted_lambda,errors='coerce').where(pd.to_numeric(corr.fitted_lambda,errors='coerce')>0)); corr=corr.drop(columns='fitted_lambda'); cols=['fitted_beta','log_scale','spill_mean_duration_minutes','spill_events_per_monitoring_year',*ENV]; corr=corr[cols]
 pear=corr.corr('pearson'); spear=corr.corr('spearman'); n=pd.DataFrame(index=cols,columns=cols,dtype=int); pp=pd.DataFrame(index=cols,columns=cols,dtype=float)
 for a in cols:
  for b in cols:
   if a == b:
    n.loc[a,b]=int(corr[a].notna().sum()); pp.loc[a,b]=0.0
   else:
    q=corr[[a,b]].dropna(); n.loc[a,b]=len(q); pp.loc[a,b]=stats.pearsonr(q[a],q[b]).pvalue if len(q)>=3 and q[a].nunique()>1 and q[b].nunique()>1 else np.nan
 pear.to_csv(d/'pearson_r.csv'); pp.to_csv(d/'pearson_p.csv'); spear.to_csv(d/'spearman_rho.csv'); n.to_csv(d/'pairwise_n.csv')
 pairs=[]
 for i,a in enumerate(cols):
  for b in cols[i+1:]:
   if abs(pear.loc[a,b])>=.70: pairs.append({'variable_1':a,'variable_2':b,'Pearson_r':pear.loc[a,b],'p':pp.loc[a,b],'n':n.loc[a,b]})
 pd.DataFrame(pairs).sort_values('Pearson_r',key=abs,ascending=False).to_csv(d/'high_correlation_pairs.csv',index=False)
 fig,ax=plt.subplots(figsize=(12,10)); im=ax.imshow(pear,vmin=-1,vmax=1,cmap='coolwarm'); labels=[c.replace('_',' ') for c in cols]; ax.set_xticks(range(len(cols)),labels,rotation=55,ha='right',fontsize=6); ax.set_yticks(range(len(cols)),labels,fontsize=6); fig.colorbar(im,ax=ax,shrink=.75); ax.set_title('Four-outcome and predictor Pearson correlation matrix',loc='left',weight='bold',color=BLUE); fig.tight_layout(); fig.savefig(d/'all_variables_correlation_matrix.pdf'); plt.close(fig)
 return pear

def summary():
 d=STATS/'cross_outcome'; tables={}
 for k,path,s in [('beta','beta_parameter_analysis/beta_analysis_results.csv','strict'),('scale','scale_parameter_analysis/scale_analysis_results.csv','strict'),('duration','spill_duration_analysis/spill_duration_analysis_results.csv','primary'),('frequency','spill_count_analysis/spill_count_analysis_results.csv','primary')]: tables[k]=pd.read_csv(STATS/path).query('sample==@s').set_index('variable')
 rows=[]
 for v in ENV:
  row={'variable':v,'label':tables['beta'].loc[v,'readable_label']}
  for k,t in tables.items():
   r=t.loc[v]; row.update({f'{k}_effect_pct':r.effect_pct,f'{k}_CI_lower_pct':r.effect_CI_lower_pct,f'{k}_CI_upper_pct':r.effect_CI_upper_pct,f'{k}_p':r.RAW_REGRESSION_P,f'{k}_q':r.FDR_Q,f'{k}_status':r.evidence_status})
  rows.append(row)
 out=pd.DataFrame(rows); out.to_csv(d/'four_outcome_univariate_summary.csv',index=False)
 pd.DataFrame([{'requested_workbook':'four_outcome_univariate_summary.xlsx','status':'deferred - artifact-tool unavailable','source_csv':'four_outcome_univariate_summary.csv'}]).to_csv(d/'workbook_handoff_manifest.csv',index=False)
 # Two landscape vector pages.
 midpoint=(len(out)+1)//2
 for part,sub in enumerate([out.iloc[:midpoint],out.iloc[midpoint:]],1):
  c=canvas.Canvas(str(d/f'_summary_{part}.pdf'),pagesize=landscape(A4)); w,h=landscape(A4); c.setFillColor(colors.HexColor(BLUE)); c.rect(0,h-18*mm,w,18*mm,fill=1,stroke=0); c.setFillColor(colors.white); c.setFont('Helvetica-Bold',10); c.drawString(12*mm,h-11*mm,f'FOUR-OUTCOME UNIVARIATE SUMMARY ({part}/2)');
  styles=getSampleStyleSheet(); small=ParagraphStyle('s',parent=styles['BodyText'],fontSize=5.3,leading=6.3); head=ParagraphStyle('h',parent=small,textColor=colors.white,fontName='Helvetica-Bold')
  data=[[Paragraph(x,head) for x in ['Predictor','Beta effect/q','Scale effect/q','Duration effect/q','Frequency effect/q']]]
  for r in sub.itertuples(): data.append([Paragraph(r.label,small),Paragraph(f'{r.beta_effect_pct:+.1f}% / {r.beta_q:.3g}',small),Paragraph(f'{r.scale_effect_pct:+.1f}% / {r.scale_q:.3g}',small),Paragraph(f'{r.duration_effect_pct:+.1f}% / {r.duration_q:.3g}',small),Paragraph(f'{r.frequency_effect_pct:+.1f}% / {r.frequency_q:.3g}',small)])
  t=Table(data,colWidths=[63*mm,43*mm,43*mm,43*mm,43*mm]); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor(MID)),('GRID',(0,0),(-1,-1),.25,colors.HexColor('#C3D2DA')),('VALIGN',(0,0),(-1,-1),'TOP'),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F3F7F9')]) ])); _,th=t.wrap(w-24*mm,h-35*mm); t.drawOn(c,12*mm,h-26*mm-th); c.setFillColor(colors.HexColor(GREY)); c.setFont('Helvetica',7); c.drawString(12*mm,9*mm,'Effects are unadjusted percentage associations per frozen meaningful increment; q is within-family BH FDR. Not causal.'); c.save()
 writer=PdfWriter(); writer.append(d/'_summary_1.pdf'); writer.append(d/'_summary_2.pdf'); writer.write(d/'four_outcome_univariate_summary.pdf')

def main():
 plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
 frame=pd.read_csv(DATA/'scientific_master_annual_nimrod_v1.csv',low_memory=False); relationship(frame); correlation(frame); summary(); print('cross-outcome products complete')
if __name__=='__main__': main()
