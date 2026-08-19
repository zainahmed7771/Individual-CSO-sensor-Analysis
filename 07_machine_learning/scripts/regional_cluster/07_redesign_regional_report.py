"""Rebuild the regional report in the compact CSO_NEW redesigned visual language."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[2]
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"01_regional_cluster_model"
ASSETS=BASE/"report"/"assets_redesigned"
ASSETS.mkdir(parents=True,exist_ok=True)
sys.path.append(str(REPO/"PROJECT_RESULTS"/"machine_learning"/"_runtime"))

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import BaseDocTemplate, Flowable, Frame, Image, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle

NAVY=colors.HexColor("#173f5f")
TEAL=colors.HexColor("#2d6f73")
ORANGE=colors.HexColor("#d7902f")
INK=colors.HexColor("#202a31")
MUTED=colors.HexColor("#66737d")
PALE_GREEN=colors.HexColor("#eaf3ef")
PALE_BLUE=colors.HexColor("#edf4f7")
PALE_YELLOW=colors.HexColor("#fbf2dc")
PALE_RED=colors.HexColor("#f7e8e5")
GRID=colors.HexColor("#bdc8ce")

S=getSampleStyleSheet()
S.add(ParagraphStyle(name="Kicker",fontName="Helvetica-Bold",fontSize=7.1,leading=8.4,textColor=ORANGE,spaceAfter=4))
S.add(ParagraphStyle(name="CoverTitle",fontName="Helvetica-Bold",fontSize=22,leading=21.5,textColor=NAVY,spaceAfter=4))
S.add(ParagraphStyle(name="Deck",fontName="Times-Roman",fontSize=10.2,leading=15,textColor=MUTED,spaceAfter=9))
S.add(ParagraphStyle(name="H1",fontName="Helvetica-Bold",fontSize=14.2,leading=15.2,textColor=NAVY,spaceAfter=4))
S.add(ParagraphStyle(name="H2",fontName="Helvetica-Bold",fontSize=9.1,leading=10.5,textColor=NAVY,spaceBefore=3,spaceAfter=2))
S.add(ParagraphStyle(name="Body",fontName="Times-Roman",fontSize=7.35,leading=9.65,textColor=INK,spaceAfter=3))
S.add(ParagraphStyle(name="Small",fontName="Times-Roman",fontSize=6.2,leading=7.6,textColor=INK))
S.add(ParagraphStyle(name="Tiny",fontName="Times-Roman",fontSize=5.5,leading=6.6,textColor=INK))
S.add(ParagraphStyle(name="Box",fontName="Times-Roman",fontSize=6.9,leading=9.0,textColor=INK))
S.add(ParagraphStyle(name="BoxBold",fontName="Helvetica-Bold",fontSize=7.0,leading=9.0,textColor=NAVY,alignment=TA_CENTER))
S.add(ParagraphStyle(name="Caption",fontName="Times-Roman",fontSize=5.8,leading=7.0,textColor=INK,spaceAfter=3))
S.add(ParagraphStyle(name="Footer",fontName="Times-Roman",fontSize=5.1,leading=6,textColor=MUTED,alignment=TA_CENTER))

TARGET_LABEL={"beta":"fitted beta","lambda":"inverse-timescale lambda","duration":"pooled mean duration","frequency":"annualised frequency"}
TARGET_QUESTION={
    "beta":"Can regional environmental conditions predict the mean fitted beta shape parameter?",
    "lambda":"Can regional conditions predict the mean inverse-timescale lambda?",
    "duration":"Can the catchment environment predict pooled mean spill duration?",
    "frequency":"Can the catchment environment predict exposure-adjusted spill frequency?",
}

def P(text,style="Body"): return Paragraph(str(text),S[style])
def fm(x,d=3):
    try: return "-" if pd.isna(x) else f"{float(x):.{d}f}"
    except Exception: return str(x)

class Rule(Flowable):
    def __init__(self,width=180*mm,colour=NAVY,thickness=.7): Flowable.__init__(self); self.width=width; self.height=1.2*mm; self.colour=colour; self.thickness=thickness
    def draw(self): self.canv.setStrokeColor(self.colour); self.canv.setLineWidth(self.thickness); self.canv.line(0,self.height/2,self.width,self.height/2)

class Report(BaseDocTemplate):
    def __init__(self,path):
        super().__init__(str(path),pagesize=A4,leftMargin=15.5*mm,rightMargin=15.5*mm,topMargin=14*mm,bottomMargin=14*mm,title="CSO Regional Cluster Machine Learning Report",author="UCL CSO Spatial Drivers Project")
        frame=Frame(self.leftMargin,self.bottomMargin,self.width,self.height,id="normal",showBoundary=0)
        self.addPageTemplates(PageTemplate(id="main",frames=frame,onPage=self.header_footer))
    def header_footer(self,c,doc):
        c.saveState(); c.setFillColor(MUTED); c.setFont("Helvetica",4.8); c.drawCentredString(A4[0]/2,A4[1]-7.1*mm,"CSO SPATIAL DRIVERS PROJECT  |  REGIONAL CLUSTER MACHINE LEARNING")
        c.setStrokeColor(colors.HexColor("#d7dfe3")); c.setLineWidth(.35); c.line(15.5*mm,10.2*mm,A4[0]-15.5*mm,10.2*mm)
        c.setFont("Times-Roman",5); c.drawCentredString(A4[0]/2,6.5*mm,"UCL CSO research project  |  19 August 2026"); c.drawRightString(A4[0]-15.5*mm,6.5*mm,str(doc.page)); c.restoreState()

def heading(number,title): return [P(f"{number}.  {title}","H1"),Rule(),Spacer(1,1.5*mm)]

def box(title,text,colour=PALE_GREEN):
    content=P(f"<b>{title}</b> {text}","Box")
    return Table([[content]],colWidths=[178.5*mm],style=TableStyle([("BACKGROUND",(0,0),(-1,-1),colour),("BOX",(0,0),(-1,-1),.25,colors.white),("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]))

def compact_table(frame,cols=None,widths=None,font=5.8,header=True):
    cols=list(cols or frame.columns); data=[cols]+frame.loc[:,cols].astype(str).values.tolist()
    cells=[[P(x,"Tiny") for x in row] for row in data]
    t=Table(cells,colWidths=widths,repeatRows=1 if header else 0,hAlign="LEFT")
    style=[("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),3),("RIGHTPADDING",(0,0),(-1,-1),3),("TOPPADDING",(0,0),(-1,-1),2.2),("BOTTOMPADDING",(0,0),(-1,-1),2.2),("LINEBELOW",(0,0),(-1,0),.8,NAVY),("LINEBELOW",(0,-1),(-1,-1),.35,NAVY),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f5f7f8")])]
    t.setStyle(TableStyle(style)); return t

def figure(path,width=178*mm,maxheight=78*mm):
    im=Image(str(path)); scale=min(width/im.imageWidth,maxheight/im.imageHeight); im.drawWidth*=scale; im.drawHeight*=scale; return im

def save_fig(fig,path):
    fig.savefig(path,dpi=320,bbox_inches="tight",facecolor="white"); plt.close(fig)

def plot_assets(summary):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7,"axes.titlesize":8,"axes.labelsize":7,"xtick.labelsize":6,"ytick.labelsize":6,"axes.edgecolor":"#314b5b","axes.linewidth":.6})
    for _,r in summary.iterrows():
        key=r.target; sel=pd.read_csv(BASE/"results"/key/"validation_model_comparison.csv").head(6); imp=pd.read_csv(BASE/"results"/key/"permutation_importance.csv").head(7).sort_values("permutation_importance")
        fig,axes=plt.subplots(1,2,figsize=(6.9,2.35),gridspec_kw={"wspace":.55})
        names=[f"{a} ({x:g})" if a!="OLS" else "OLS" for a,x in zip(sel.algorithm,sel.alpha)]; colours=["#d7902f"]+["#173f5f"]*(len(sel)-1)
        axes[0].barh(np.arange(len(sel))[::-1],sel.validation_rmse,color=colours); axes[0].set_yticks(np.arange(len(sel))[::-1],names); axes[0].set_xlabel("Validation RMSE"); axes[0].set_title("Candidate selection - validation only",fontweight="bold",color="#173f5f")
        axes[1].barh(imp.feature.str.replace("_"," "),imp.permutation_importance,color="#173f5f"); axes[1].axvline(0,color="#777",lw=.6); axes[1].set_xlabel("Permutation importance"); axes[1].set_title("Locked-test predictive importance",fontweight="bold",color="#173f5f")
        for ax in axes: ax.grid(axis="x",color="#e5e9eb",lw=.4); ax.set_axisbelow(True)
        save_fig(fig,ASSETS/f"{key}_selection_importance.png")
        pred=pd.read_csv(BASE/"results"/key/"test_predictions.csv"); y=pred.observed_transformed.to_numpy(); p=pred.predicted_transformed.to_numpy(); res=y-p
        fig,axes=plt.subplots(1,3,figsize=(7.05,2.2),gridspec_kw={"wspace":.48})
        axes[0].scatter(y,p,s=13,color="#2d6f9f",alpha=.78,edgecolor="white",linewidth=.25); lo=min(y.min(),p.min()); hi=max(y.max(),p.max()); axes[0].plot([lo,hi],[lo,hi],ls="--",color="#d7902f",lw=1); axes[0].set(xlabel="Observed",ylabel="Predicted",title="Observed vs predicted")
        axes[1].scatter(p,res,s=13,color="#2d6f9f",alpha=.72,edgecolor="white",linewidth=.2); axes[1].axhline(0,ls="--",color="#d7902f",lw=1); axes[1].set(xlabel="Predicted",ylabel="Residual",title="Residual pattern")
        axes[2].hist(res,bins=min(12,max(6,len(res)//3)),color="#173f5f",alpha=.88,edgecolor="white"); axes[2].axvline(0,color="#d7902f",lw=1); axes[2].set(xlabel="Residual",ylabel="Count",title="Residual distribution")
        for ax in axes: ax.grid(color="#e5e9eb",lw=.35); ax.set_axisbelow(True)
        save_fig(fig,ASSETS/f"{key}_diagnostics.png")
    # Four-panel learning curves.
    fig,axes=plt.subplots(2,2,figsize=(7.0,4.5),gridspec_kw={"hspace":.48,"wspace":.35})
    for ax,(_,r) in zip(axes.flat,summary.iterrows()):
        lc=pd.read_csv(BASE/"results"/r.target/"learning_curve.csv"); q=lc.groupby("n").cv_rmse.agg(["mean","std"]); ax.plot(q.index,q["mean"],marker="o",color="#173f5f",lw=1.2,ms=3); ax.fill_between(q.index,q["mean"]-q["std"],q["mean"]+q["std"],color="#cbdbe7",alpha=.7); ax.set(title=TARGET_LABEL[r.target].title(),xlabel="Rows",ylabel="CV RMSE"); ax.grid(color="#e5e9eb",lw=.35)
    fig.suptitle("Regional learning curves",fontweight="bold",color="#173f5f",y=.99); save_fig(fig,ASSETS/"learning_curves.png")
    # Radius and old/WWTW/regional comparison.
    rad=pd.read_csv(BASE/"results"/"cluster_radius_model_sensitivity.csv"); comp=pd.read_csv(BASE/"results"/"old_wwtw_regional_comparison.csv")
    fig,axes=plt.subplots(1,2,figsize=(7.05,2.45))
    for key,g in rad.groupby("target"): axes[0].plot(g.radius_km,g.rmse_improvement_pct,marker="o",lw=1.2,label=key)
    axes[0].axhline(0,color="#777",lw=.6); axes[0].axvline(25,color="#d7902f",ls="--",lw=.9); axes[0].set(title="Pre-specified radius sensitivity",xlabel="Radius (km)",ylabel="RMSE gain vs Dummy (%)"); axes[0].legend(fontsize=5,ncol=2)
    units=["selected_sensor","multi_sensor_WWTW","regional_25km"]; x=np.arange(4); width=.22
    for j,u in enumerate(units):
        g=comp.loc[comp.analysis_unit.eq(u)].set_index("target").reindex(["beta","lambda","duration","frequency"]); axes[1].bar(x+(j-1)*width,g.rmse_improvement_pct,width,label=u.replace("_"," "))
    axes[1].axhline(0,color="#777",lw=.6); axes[1].set_xticks(x,["beta","lambda","duration","frequency"]); axes[1].set(title="Change with analysis unit",ylabel="RMSE gain vs Dummy (%)"); axes[1].legend(fontsize=4.8)
    for ax in axes: ax.grid(axis="y",color="#e5e9eb",lw=.35); ax.set_axisbelow(True)
    save_fig(fig,ASSETS/"radius_and_unit_comparison.png")

def metrics_block(row):
    d=[
        [P("Data","Small"),P(f"{int(row.eligible_n)} eligible regions; {int(row.train_n)} train, {int(row.validation_n)} validation, {int(row.test_n)} locked test","Small"),P("Selected model","Small"),P(f"{row.selected_algorithm}, alpha={row.selected_alpha:g}","Small")],
        [P("Test RMSE","Small"),P(fm(row.rmse),"Small"),P("Dummy RMSE","Small"),P(fm(row.dummy_rmse),"Small")],
        [P("RMSE gain","Small"),P(f"{row.rmse_improvement_pct:.1f}%","Small"),P("R-squared","Small"),P(fm(row.r2),"Small")],
        [P("Rank correlation","Small"),P(fm(row.spearman),"Small"),P("95% gain interval","Small"),P(f"{row.bootstrap_improvement_low_95:.1f}% to {row.bootstrap_improvement_high_95:.1f}%","Small")],
    ]
    return Table(d,colWidths=[26*mm,58*mm,27*mm,67*mm],style=TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),("LINEABOVE",(0,0),(-1,0),.55,NAVY),("LINEBELOW",(0,-1),(-1,-1),.55,NAVY),("BACKGROUND",(0,0),(0,-1),PALE_BLUE),("BACKGROUND",(2,0),(2,-1),PALE_BLUE),("LEFTPADDING",(0,0),(-1,-1),3),("RIGHTPADDING",(0,0),(-1,-1),3),("TOPPADDING",(0,0),(-1,-1),2.4),("BOTTOMPADDING",(0,0),(-1,-1),2.4)]))

def model_main_page(story,page_no,key,row):
    model_no={"beta":1,"lambda":2,"duration":3,"frequency":4}[key]
    section_no=model_no+2
    story+=heading(section_no,f"Model {model_no}: predicting {TARGET_LABEL[key]}")
    story.append(box("Question.",TARGET_QUESTION[key],PALE_GREEN)); story.append(Spacer(1,2*mm)); story.append(metrics_block(row)); story.append(Spacer(1,2*mm))
    meaning={
        "beta":["The model reduces transformed-scale RMSE by 17.4% and has the largest regional R-squared.","The bootstrap interval still crosses zero because only 34 regions are in the locked test.","Population density dominates permutation importance, but this is conditional predictive evidence."],
        "lambda":["The model improves on Dummy by 9.7%, with modest rank ordering of unseen regions.","The wide target scale and bootstrap interval show that the gain is uncertain.","Beta and lambda are jointly fitted, so parameter instability remains relevant."],
        "duration":["The exact pooled mean uses total cleaned duration divided by total valid events.","The 7.8% point gain is encouraging but its bootstrap interval crosses zero.","Static/annual predictors remain a coarse representation of event hydraulics."],
        "frequency":["The locked-test model is worse than the mean-only Dummy despite good validation RMSE.","Calibration slope is only 0.325, showing severe compression toward the regional mean.","Exposure adjustment is correct; weak skill therefore cannot be blamed on raw count exposure."],
    }
    story.append(P("<b>What this means</b>","H2"));
    for text in meaning[key]: story.append(P(f"- {text}","Body"))
    story.append(figure(ASSETS/f"{key}_selection_importance.png",168*mm,64*mm)); story.append(P(f"Figure. Validation-only candidate selection and locked-test predictive importance for regional {TARGET_LABEL[key]}.","Caption"));
    interp="Promising but uncertain regional signal." if key=="beta" else "Modest, uncertain regional signal." if key in ("lambda","duration") else "No locked-test evidence over the internal baseline."
    story.append(box("Core finding.",interp,PALE_YELLOW if key!="frequency" else PALE_RED)); story.append(PageBreak())

def model_diag_page(story,key,row):
    story.append(P(f"Locked-test diagnostics: where regional {TARGET_LABEL[key]} succeeds and fails","H2")); story.append(Rule()); story.append(Spacer(1,2*mm)); story.append(figure(ASSETS/f"{key}_diagnostics.png",177*mm,65*mm)); story.append(P("Figure. Observed versus predicted values, residual pattern, and residual distribution on the transformed scale.","Caption"))
    if key=="frequency":
        diag="Predictions collapse toward the centre and do not track extremes. The negative test R-squared means the selected model is less accurate than predicting the training-plus-validation mean for every locked-test region."
        verdict="Regional aggregation does not rescue frequency. Annual/static predictors still omit the storm sequence, control state, storage and monitoring heterogeneity that generate spill occurrence."
    elif key=="beta":
        diag="Predictions show useful rank ordering but still compress high and low beta values. Residual spread and the wide bootstrap interval prevent a strong generalization claim."
        verdict="The larger analysis unit appears to remove some selected-sensor noise, but the result is compatible with both better physical alignment and smoothing of extremes."
    else:
        diag="Predictions retain modest ordering but show substantial residual spread and compression toward the mean. There is no evidence of precise regional prediction."
        verdict="The model finds some aggregate signal, but richer hydraulic and event-state information is needed before the result can be treated as operationally useful."
    story.append(box("How to read the diagnostics.",diag,PALE_BLUE)); story.append(Spacer(1,2*mm)); story.append(box("Scientific verdict.",verdict,PALE_YELLOW if key!="frequency" else PALE_RED)); story.append(Spacer(1,3*mm)); story.append(P("<b>Guardrail.</b> RMSE is the headline prediction-error metric. R-squared can be negative on a locked test; this correctly indicates performance worse than Dummy rather than a software failure.","Body")); story.append(PageBreak())

def build():
    summary=pd.read_csv(BASE/"results"/"regional_four_model_summary.csv"); master=pd.read_csv(BASE/"data"/"regional_master_25km.csv"); comp=pd.read_csv(BASE/"results"/"old_wwtw_regional_comparison.csv"); radius=pd.read_csv(BASE/"results"/"cluster_radius_model_sensitivity.csv"); hier=pd.read_csv(ROOT/"03_hierarchical_benchmark"/"results"/"hierarchical_benchmark_summary.csv")
    plot_assets(summary)
    story=[]
    # Page 1 cover
    story+=[Spacer(1,8*mm),P("SUPERVISOR-READY MACHINE-LEARNING REPORT","Kicker"),P("CSO REGIONAL CLUSTER ANALYSIS\nPredicting Collective Spill Behaviour Across WWTW Regions","CoverTitle"),Rule(),P("A compact research report testing whether a larger, physically transparent spatial unit reveals stronger environmental signal than one selected sensor.","Deck"),box("Core conclusion.","Regional aggregation improves beta most, but it does not rescue every target. Beta reaches a 17.4% locked-test RMSE gain and R-squared 0.314; lambda and duration improve modestly; frequency is worse than Dummy. The evidence supports partial spatial-unit mismatch plus target smoothing and missing hydraulic/event-state information.",PALE_GREEN),Spacer(1,7*mm)]
    flow=[[P("MODEL 1\nRegional beta","BoxBold"),P("MODEL 2\nRegional lambda","BoxBold"),P("MODEL 3\nPooled duration","BoxBold"),P("MODEL 4\nExposure frequency","BoxBold")],[P("company x 25 km target-independent WWTW clusters; shared regional catchment information","Small"),"","",""]]
    ft=Table(flow,colWidths=[44.5*mm]*4,rowHeights=[17*mm,12*mm]); ft.setStyle(TableStyle([("SPAN",(0,1),(3,1)),("BOX",(0,0),(-1,0),.7,NAVY),("INNERGRID",(0,0),(-1,0),.55,NAVY),("BACKGROUND",(0,0),(-1,0),PALE_BLUE),("ALIGN",(0,0),(-1,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("LINEABOVE",(0,1),(-1,1),.6,NAVY),("TEXTCOLOR",(0,1),(-1,1),TEAL)])); story+=[ft,Spacer(1,43*mm),P("259 primary regions  |  1,444 WWTWs  |  2,045,832 exact assigned events\n70% train  |  15% validation  |  15% locked test  |  RMSE headline metric","Caption"),PageBreak()]
    # Page 2 executive
    story+=heading(0,"Executive interpretation: what the four regional models actually say")
    st=summary[["target","selected_algorithm","rmse","rmse_improvement_pct","r2","spearman"]].copy(); st.target=st.target.map(TARGET_LABEL); st.columns=["Target","Selected model","Test RMSE","Gain vs Dummy","R-squared","Spearman"]
    st["Test RMSE"]=st["Test RMSE"].map(lambda x:fm(x)); st["Gain vs Dummy"]=st["Gain vs Dummy"].map(lambda x:f"{x:.1f}%"); st["R-squared"]=st["R-squared"].map(lambda x:fm(x)); st["Spearman"]=st["Spearman"].map(lambda x:fm(x))
    story+=[compact_table(st,widths=[31*mm,29*mm,25*mm,29*mm,25*mm,25*mm]),Spacer(1,2*mm),box("The result in plain English.","Changing the spatial unit helps beta most. The other positive gains are smaller and uncertain, while frequency fails on the locked test. A larger region makes targets smoother, but smoothing alone is not the same as uncovering stronger catchment physics.",PALE_GREEN),P("Five findings to tell a supervisor first","H2")]
    findings=["Regional beta is the clearest result: 17.4% lower RMSE than Dummy, but its bootstrap interval (-8.0% to 30.1%) crosses zero.","Lambda and duration show modest point gains of 9.7% and 7.8%; neither is precise enough for a strong claim.","Frequency deteriorates by 2.0% on the locked test despite valid exposure adjustment.","The 50 km sensitivity often performs poorly because only 87-107 eligible regions remain.","The most defensible explanation combines noisy targets, partial unit mismatch, variance shrinkage, and missing hydraulic/event-state predictors."]
    for i,x in enumerate(findings,1): story.append(P(f"<b>{i}.</b> {x}","Body"))
    story+=[box("Critical comparison rule.","Compare normalized gain over each model's own Dummy, R-squared, rank correlation, target variance and uncertainty. Do not compare raw RMSE across different target definitions or spatial units.",PALE_YELLOW),PageBreak()]
    # Page 3 metrics
    story+=heading(1,"How to read the metrics and plots")
    story+=[P("<b>RMSE: the main prediction-error metric</b>","H2"),P("RMSE measures the typical transformed-scale prediction error. Lower is better. Improvement is calculated relative to the internal Dummy fitted to the same target and split.","Body"),box("Why transformed targets?","Beta and lambda are strictly positive and skewed, so log transforms prevent a few extreme regions dominating model selection. Duration and frequency use log1p.",PALE_BLUE),P("<b>R-squared can be negative on a locked test</b>","H2"),P("Negative R-squared means a model is worse than predicting a constant mean. It is an important failure diagnostic, not an invalid number.","Body"),P("<b>Spearman correlation: ranking rather than exact prediction</b>","H2"),P("Spearman rho asks whether high-observed regions tend to receive high predictions. A model can rank moderately while still having poor absolute calibration.","Body"),P("<b>Bootstrap intervals and calibration</b>","H2"),P("Bootstrap intervals describe uncertainty in test-set improvement. Calibration slope near one is desirable; slopes well below one indicate compression toward the mean.","Body"),box("Most important caution.","All predictor importance is conditional and predictive. It does not identify causal environmental drivers.",PALE_YELLOW),P("<b>Residual plots</b>","H2"),P("A useful model should not leave systematic curvature or funnel-shaped spread. The residual histogram reveals skew and extreme errors that an average metric can hide.","Body"),PageBreak()]
    # Page 4 design
    story+=heading(2,"Experimental design: why the locked test and regional definition matter")
    story+=[box("The central methodological strength.","The region definition is fixed before outcomes are inspected. Model selection happens on training CV and validation data only; the locked test is opened once after the model is frozen.",PALE_GREEN)]
    split=Table([[P("70% TRAIN","BoxBold"),P("15% VALIDATION","BoxBold"),P("15% LOCKED TEST","BoxBold")],[P("fit preprocessing and candidates","Small"),P("choose final specification","Small"),P("final evaluation once","Small")]],colWidths=[59.5*mm]*3,style=TableStyle([("BOX",(0,0),(-1,-1),.65,NAVY),("INNERGRID",(0,0),(-1,-1),.4,NAVY),("BACKGROUND",(0,0),(-1,0),PALE_BLUE),("ALIGN",(0,0),(-1,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)])); story+=[Spacer(1,3*mm),split,P("Regional unit","H2"),P("No supplied authoritative EA operational- or management-catchment polygon was suitable. The documented fallback is complete-linkage clustering of WWTW centroids within company in EPSG:27700. The primary maximum separation is 25 km; 10 and 50 km are pre-specified sensitivities.","Body")]
    design=pd.DataFrame({"Component":["Region membership","Beta / lambda","Duration","Frequency","Catchment area","Percentages / rainfall / slope","Population density","Capacity"],"Rule":["company + BNG coordinates only","sensor mean plus median/spread sensitivities","exact pooled total duration / exact event count","exact events / monitored sensor-years","sum","catchment-area weighted","population sum / represented area","sum(load PE) / sum(design PE)"]})
    story+=[compact_table(design,widths=[48*mm,130*mm]),P("Cohort support","H2"),P(f"The primary master contains {len(master)} regions. Median membership is {master.number_of_WWTWs.median():.0f} WWTWs and {master.number_of_assigned_CSO_sensors.median():.0f} assigned sensors; median event count is {master.number_of_valid_events.median():,.0f}. Exact counts, totals and medians were reconstructed from 2,045,832 cleaned, de-duplicated assigned events.","Body"),box("No leakage from geography.","Target values, feature values and model performance never influence the region boundary.",PALE_YELLOW),PageBreak()]
    # Pages 5-12 model sections
    for i,key in enumerate(["beta","lambda","duration","frequency"]):
        row=summary.loc[summary.target.eq(key)].iloc[0]; model_main_page(story,5+i*2,key,row); model_diag_page(story,key,row)
    # Page 13 radius and units
    story+=heading(7,"Spatial-scale sensitivity and old-versus-regional comparison")
    story+=[figure(ASSETS/"radius_and_unit_comparison.png",176*mm,64*mm),P("Figure. Fixed-Ridge radius sensitivity and normalized improvement for selected-sensor, multi-sensor WWTW, and primary regional units.","Caption")]
    rt=radius[["radius_km","target","n_regions","rmse_improvement_pct","r2"]].copy(); rt.columns=["Radius km","Target","Regions","RMSE gain %","R-squared"]; rt["RMSE gain %"]=rt["RMSE gain %"].map(lambda x:fm(x,1)); rt["R-squared"]=rt["R-squared"].map(lambda x:fm(x)); story+=[compact_table(rt,widths=[25*mm,33*mm,28*mm,48*mm,36*mm]),box("How to interpret radius sensitivity.","Ten kilometres can perform better for some targets, while 50 km often loses signal and sample size. These results do not replace the pre-declared 25 km headline; they show that spatial scale itself is a material modelling assumption.",PALE_BLUE),box("Smoothing warning.","A larger cluster can improve RMSE by shrinking extremes even when the predictor-target relationship is not physically stronger. Normalized metrics and variance change must be read together.",PALE_YELLOW),PageBreak()]
    # Page 14 learning curves
    story+=heading(8,"Learning curves: would simply adding more regional rows fix the problem?")
    story+=[figure(ASSETS/"learning_curves.png",172*mm,115*mm),P("Figure. Training-plus-validation sample size against cross-validated transformed-scale RMSE.","Caption")]
    lct=[]
    for _,r in summary.iterrows():
        lc=pd.read_csv(BASE/"results"/r.target/"learning_curve.csv"); q=lc.groupby("n").cv_rmse.mean(); rel=(q.iloc[0]-q.iloc[-1])/q.iloc[0]*100; lct.append({"Target":r.target,"First CV RMSE":fm(q.iloc[0]),"Largest-n RMSE":fm(q.iloc[-1]),"Change":f"{rel:.1f}%","Diagnosis":"still declining" if rel>10 else "near plateau"})
    story+=[compact_table(pd.DataFrame(lct),widths=[32*mm,35*mm,38*mm,30*mm,43*mm]),box("Why a plateau matters.","More similar regions are unlikely to create strong prediction if feature information and target reliability are limiting. The curves argue for richer process-matched inputs before a much larger algorithm search.",PALE_GREEN),PageBreak()]
    # Page 15 robustness and hierarchy
    story+=heading(9,"Spatial/company robustness and the hierarchical benchmark")
    robust=[]
    for key in summary.target:
        x=pd.read_csv(BASE/"results"/key/"spatial_company_robustness.csv"); sp=x.loc[x.analysis.eq("100km_spatial_GroupKFold")]; robust.append({"Target":key,"100 km spatial CV RMSE":fm(sp.rmse_mean.iloc[0]) if len(sp) else "-","LOCO folds":int(x.analysis.str.startswith("leave_").sum()),"Headline test RMSE":fm(summary.loc[summary.target.eq(key),"rmse"].iloc[0])})
    story+=[P("Spatial-block and leave-one-company-out checks","H2"),compact_table(pd.DataFrame(robust),widths=[35*mm,62*mm,34*mm,47*mm]),P("These sensitivities are deliberately harder than random regional prediction. Operator-specific monitoring, network practice and geography can dominate individual held-company folds.","Body"),P("Hierarchical benchmark","H2")]
    hh=hier[["target","method","n","groups","status","held_region_cv_rmse_mean"]].copy(); hh.columns=["Target","Method","Rows","Regions","Status","Held-region RMSE"]; hh["Held-region RMSE"]=hh["Held-region RMSE"].map(lambda x:fm(x)); story+=[compact_table(hh,widths=[20*mm,48*mm,18*mm,20*mm,49*mm,25*mm]),box("Why the mixed model is not a success claim.","The full beta and duration MixedLM fits reached singular random-effect covariance. Some folds fitted, but the full specification is unstable. It is reported as an honest negative result and not promoted above regional averaging.",PALE_RED),P("What would make partial pooling defensible?","H2"),P("Use event- or sensor-level outcomes, a better physical grouping hierarchy, explicit exposure/likelihood structure, and enough repeated observations per group. A random-intercept model should not be forced merely because partial pooling is attractive in principle.","Body"),box("Robustness verdict.","Regional beta remains the most promising signal, but uncertainty and spatial/operator sensitivity mean it requires confirmation rather than immediate operational interpretation.",PALE_YELLOW),PageBreak()]
    # Page 16 beta geometry and limitations
    story+=heading(10,"Why the models remain weak: parameter geometry and missing physics")
    story+=[figure(ROOT/"00_diagnostics"/"beta_logit_lambda"/"beta_logit_vs_loglambda.png",176*mm,84*mm),P("Figure. Original beta and generalized logit log[beta/(2-beta)] against log(lambda). Exact optimizer-boundary values are excluded rather than clipped.","Caption"),box("Parameter-geometry finding.","Strict-sample R-squared rises from 0.527 to 0.763 after the bounded-coordinate transform, but RESET and quadratic tests remain highly significant. Beta and lambda are jointly fitted from the same duration distribution, so this is an identifiability diagnostic, not a causal relationship.",PALE_BLUE)]
    limits=pd.DataFrame({"Limitation":["Target quality","Hydraulic state","Catchment scale","Event dynamics","Operator variation","Ecological aggregation"],"Why it matters":["tail fits and extreme estimates remain uncertain","topology, storage, pumping and control are absent","environmental polygons do not equal sewer systems","annual/static summaries omit storm sequence and saturation","monitoring and operating practice differ","averaging can reduce variance without stronger physics"]})
    story+=[compact_table(limits,widths=[44*mm,134*mm]),box("The wrong conclusion would be:","The regional model has solved the problem because beta R-squared is higher. The correct conclusion is narrower: aggregation appears to remove some noise for beta, while physical information remains insufficient for strong prediction across all outcomes.",PALE_RED),PageBreak()]
    # Page 17 synthesis
    story+=heading(11,"Supervisor-facing synthesis: how to explain the results clearly")
    story+=[box("Thirty-second summary.","We replaced one selected sensor with independently defined within-company WWTW regions and retained a locked test. Regional beta improves most; lambda and duration improve modestly; frequency does not. No result is precise enough to prove a universal regional mechanism. The combined evidence points to partial unit mismatch, smoothing, fitted-parameter noise, and missing hydraulic/event-state information.",PALE_GREEN)]
    qa=[("Did the regional unit work?","Partially. Beta gains materially, but the confidence interval crosses zero and frequency deteriorates."),("Did aggregation just make targets easier?","Partly. Variance and extremes shrink, so normalized gain and locked-test diagnostics are essential."),("Which regional model should be trusted most?","Beta is the strongest candidate for confirmation, not deployment. Duration and lambda are qualified; frequency fails."),("What did the hierarchical model show?","The attempted random-intercept model was singular. Better group structure and event-level likelihoods are required."),("What should we do next?","Acquire topology, pipe/storage capacity, pumping/control, event rainfall, antecedent saturation, groundwater and receiving-water state before increasing algorithm complexity.")]
    for q,a in qa: story+=[P(f"<b>If asked: '{q}'</b>","H2"),P(a,"Body")]
    story+=[P("Final validation status","H2"),P("All 40 scientific release gates pass: exact event aggregation, unique geography, exposure handling, historical split reuse, model reloads, figure pairs and rendered-PDF checks. The workbook packaging limitation is separate from the modelling results.","Body"),box("Final research conclusion.","Changing the spatial unit reveals a stronger beta signal but does not produce a generally strong regional prediction system. The highest-value next step is better physical and event-state data, followed by a carefully validated hierarchical or event-level model.",PALE_GREEN),Spacer(1,3*mm),P("This report matches the compact supervisor-report visual system used by the CSO_NEW redesigned master report: dense evidence, paired diagnostics, decision boxes, and explicit plain-English interpretation.","Caption")]

    target=BASE/"report"/"CSO_REGIONAL_CLUSTER_MACHINE_LEARNING_REPORT.pdf"; temp=target.with_suffix(".tmp.pdf"); Report(temp).build(story); temp.replace(target)
    mirror=ROOT/"output"/"pdf"/target.name; mirror.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(target,mirror)
    print(target)

if __name__=="__main__": build()
