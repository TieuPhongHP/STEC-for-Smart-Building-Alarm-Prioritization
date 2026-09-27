from __future__ import annotations
import math, json, time, os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_fscore_support,
    brier_score_loss, precision_recall_curve, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

RNG = np.random.default_rng(20260728)
SYSTEMS = ["camera", "access", "parking", "bms", "elevator", "intrusion"]
SYSTEM_IDX = {s:i for i,s in enumerate(SYSTEMS)}

# 12 zones: 3 floors x 4 zones; horizontal adjacency plus vertical cores at zone offsets 1 and 2.
N_ZONES = 12
ADJ = {i:set() for i in range(N_ZONES)}
for f in range(3):
    for j in range(4):
        z=f*4+j
        if j>0: ADJ[z].add(z-1)
        if j<3: ADJ[z].add(z+1)
for j in [1,2]:
    for f in range(2):
        a=f*4+j; b=(f+1)*4+j
        ADJ[a].add(b); ADJ[b].add(a)
DIST=np.full((N_ZONES,N_ZONES),99,dtype=int)
for i in range(N_ZONES):
    DIST[i,i]=0
    frontier=[i]
    for d in range(1,10):
        nf=[]
        for u in frontier:
            for v in ADJ[u]:
                if DIST[i,v]>d:
                    DIST[i,v]=d; nf.append(v)
        frontier=nf
        if not frontier: break

TEMPLATES = {
    "unauthorized_entry": [
        ("access","access_denied",-14,0.86), ("access","door_forced",-5,0.91),
        ("camera","person_detected",2,0.78), ("intrusion","motion_alarm",7,0.83)],
    "tailgating": [
        ("access","access_granted",-10,0.76), ("camera","multiple_persons",0,0.84),
        ("intrusion","occupancy_mismatch",8,0.72)],
    "vehicle_mismatch": [
        ("parking","plate_mismatch",-12,0.88), ("parking","barrier_forced",-2,0.90),
        ("camera","vehicle_detected",4,0.79), ("access","access_denied",10,0.71)],
    "after_hours_access": [
        ("access","access_granted",-7,0.78), ("camera","person_detected",0,0.80),
        ("bms","occupancy_after_hours",9,0.74)],
    "forced_door": [
        ("access","door_forced",-5,0.94), ("intrusion","contact_alarm",0,0.89),
        ("camera","person_detected",5,0.77)],
    "sabotage": [
        ("camera","camera_occluded",-6,0.91), ("bms","device_offline",1,0.85),
        ("access","reader_tamper",8,0.87)],
    "elevator_override": [
        ("access","access_denied",-12,0.72), ("elevator","override_active",-2,0.92),
        ("camera","person_detected",5,0.75), ("bms","restricted_floor_occupied",13,0.78)],
    "life_safety": [
        ("bms","smoke_alarm",-8,0.95), ("elevator","fire_recall",0,0.96),
        ("access","doors_released",5,0.90), ("camera","crowd_movement",12,0.74)]
}

NORMAL_SUBTYPES = {
    "camera":["person_detected","vehicle_detected","motion_analytics","camera_tamper"],
    "access":["access_granted","access_denied","door_held","door_forced"],
    "parking":["plate_read","plate_mismatch","barrier_open","barrier_fault"],
    "bms":["temperature_alarm","device_offline","occupancy_change","maintenance_mode"],
    "elevator":["car_call","door_fault","override_active","service_mode"],
    "intrusion":["motion_alarm","contact_alarm","occupancy_mismatch","panel_fault"]
}

@dataclass
class SiteParams:
    site_id:int
    rates:Dict[str,float]
    reliab:Dict[str,float]
    false_burst:float
    clock_bias:float
    incidence:float


def make_sites(n=12):
    sites=[]
    for sid in range(n):
        rates={s:float(RNG.uniform(0.10,0.42)) for s in SYSTEMS}
        reliab={s:float(RNG.uniform(0.72,0.97)) for s in SYSTEMS}
        sites.append(SiteParams(sid,rates,reliab,float(RNG.uniform(0.15,0.28)),
                                float(RNG.normal(0,2.5)),float(RNG.uniform(0.065,0.105))))
    return sites

SITES = make_sites()


def neighbor_zone(z:int, rng:np.random.Generator, p_same=.72):
    if rng.random()<p_same or not ADJ[z]: return z
    return int(rng.choice(list(ADJ[z])))


def add_event(events, system, subtype, zone, t, conf, reliab, rng, miss=0.0):
    # Lower-reliability sources are more likely to be absent and noisier.
    if rng.random() < miss + (1-reliab)*0.10:
        return
    events.append({"system":system,"subtype":subtype,"zone":int(zone),
                   "time":float(t),"confidence":float(np.clip(conf,0.01,0.999)),
                   "reliability":float(reliab)})


def generate_window(site:SiteParams, rng:np.random.Generator, force_label=None,
                    extra_missing=0.11, time_jitter=7.0):
    hour=int(rng.integers(0,24)); weekday=int(rng.integers(0,7))
    after_hours = int(hour<6 or hour>=21 or weekday>=5)
    maintenance = int(rng.random() < (0.06 if weekday<5 else 0.10))
    y=int(rng.random()<site.incidence) if force_label is None else int(force_label)
    zone=int(rng.integers(0,N_ZONES))
    events=[]

    # Benign background events.
    for s in SYSTEMS:
        lam=site.rates[s]*(1.3 if s in ("camera","access") and not after_hours else 0.75 if after_hours else 1.0)
        if maintenance and s in ("bms","elevator"): lam*=2.0
        n=int(rng.poisson(lam))
        for _ in range(n):
            subtype=str(rng.choice(NORMAL_SUBTYPES[s]))
            z=int(rng.integers(0,N_ZONES))
            t=float(rng.uniform(-60,60)+site.clock_bias+rng.normal(0,time_jitter))
            base=float(rng.beta(2.2,4.8))
            # nuisance high confidence on a single source
            if rng.random()<0.06: base=float(rng.beta(6,1.8))
            add_event(events,s,subtype,z,t,base,site.reliab[s],rng,miss=extra_missing*0.35)

    # Normal nuisance burst: high-confidence but poorly corroborated.
    if not y and rng.random()<site.false_burst:
        s=str(rng.choice(SYSTEMS)); subtype=str(rng.choice(NORMAL_SUBTYPES[s]))
        for k in range(int(rng.integers(1,4))):
            add_event(events,s,subtype,neighbor_zone(zone,rng,.55),
                      rng.normal(0,18)+site.clock_bias+rng.normal(0,time_jitter),
                      rng.beta(6,1.8),site.reliab[s],rng,miss=extra_missing*0.2)

    # Hard negatives: alarm sets with incident-like marginal counts/confidence but
    # inconsistent locations, timing, or sequence (e.g., unrelated device faults).
    if not y and rng.random()<0.115:
        pseudo=str(rng.choice(list(TEMPLATES.keys())))
        for s,subtype,dt,base_conf in TEMPLATES[pseudo]:
            z=int(rng.integers(0,N_ZONES))
            t=float(rng.uniform(-60,60)+site.clock_bias+rng.normal(0,time_jitter*1.7))
            conf=float(np.clip(rng.normal(base_conf*site.reliab[s],0.13),0.12,0.98))
            add_event(events,s,subtype,z,t,conf,site.reliab[s],rng,miss=extra_missing*.55)

    # Legitimate coordinated operational workflow, spatially coherent but semantically benign.
    if not y and maintenance and rng.random()<0.33:
        for s,subtype,dt,cf in [("bms","maintenance_mode",-10,.78),("elevator","service_mode",0,.82),("camera","person_detected",12,.62)]:
            add_event(events,s,subtype,neighbor_zone(zone,rng,.83),dt+rng.normal(0,time_jitter),
                      np.clip(rng.normal(cf,0.09),.1,.98),site.reliab[s],rng,miss=extra_missing*.4)

    incident_type="normal"
    if y:
        incident_type=str(rng.choice(list(TEMPLATES.keys()), p=np.array([.18,.14,.13,.13,.14,.10,.10,.08])))
        # after_hours_access is more likely to occur after hours; other incidents can happen anytime
        if incident_type=="after_hours_access" and not after_hours and rng.random()<0.76:
            hour=int(rng.choice([0,1,2,3,4,5,21,22,23])); after_hours=1
        # life safety is less affected by source missingness; sabotage more affected
        for s,subtype,dt,base_conf in TEMPLATES[incident_type]:
            rel=site.reliab[s]
            conf=float(np.clip(rng.normal(base_conf*rel+0.015,0.12),0.10,0.995))
            z=neighbor_zone(zone,rng,.78)
            miss=extra_missing*(0.75 if incident_type=="life_safety" else 1.15)
            add_event(events,s,subtype,z,dt+site.clock_bias+rng.normal(0,time_jitter),conf,rel,rng,miss=miss)
        # occasional distractor event elsewhere
        if rng.random()<0.30:
            s=str(rng.choice(SYSTEMS)); subtype=str(rng.choice(NORMAL_SUBTYPES[s]))
            add_event(events,s,subtype,int(rng.integers(0,N_ZONES)),rng.uniform(-60,60),
                      rng.beta(2,4),site.reliab[s],rng,miss=extra_missing*.2)

    return {"site":site.site_id,"hour":hour,"weekday":weekday,"after_hours":after_hours,
            "maintenance":maintenance,"label":y,"incident_type":incident_type,"events":events}

PAIR_WEIGHTS = {
    frozenset(["camera","access"]):1.2,
    frozenset(["camera","parking"]):1.0,
    frozenset(["access","intrusion"]):1.35,
    frozenset(["bms","elevator"]):1.1,
    frozenset(["camera","intrusion"]):1.2,
    frozenset(["access","elevator"]):0.9,
    frozenset(["camera","bms"]):0.75,
}
TAU = 30.0     # temporal decay (s) in Eq. (1)
LAMBDA = 1.6   # topological decay (zone hops) in Eq. (1)
SEQUENCES = [
    ("access_denied","door_forced"), ("door_forced","person_detected"),
    ("plate_mismatch","barrier_forced"), ("barrier_forced","vehicle_detected"),
    ("camera_occluded","device_offline"), ("access_denied","override_active"),
    ("smoke_alarm","fire_recall"), ("fire_recall","doors_released")
]
HIGH_RISK_SUBTYPES = {
    "door_forced":1.0,"reader_tamper":1.0,"camera_occluded":.95,"barrier_forced":.95,
    "smoke_alarm":1.0,"fire_recall":.95,"override_active":.8,"plate_mismatch":.65,
    "occupancy_mismatch":.7,"restricted_floor_occupied":.75,"access_denied":.45
}


def shannon_entropy(vals):
    arr=np.array(vals,dtype=float)
    if arr.sum()==0: return 0.0
    p=arr/arr.sum(); p=p[p>0]
    return float(-(p*np.log(p)).sum()/math.log(len(vals))) if len(vals)>1 else 0.0


def extract_features(w):
    ev=w["events"]
    f={"site":w["site"],"label":w["label"],"incident_type":w["incident_type"],
       "after_hours":w["after_hours"],"maintenance":w["maintenance"],
       "hour_sin":math.sin(2*math.pi*w["hour"]/24),"hour_cos":math.cos(2*math.pi*w["hour"]/24),
       "weekend":int(w["weekday"]>=5)}
    confs=[]; times=[]; zones=[]; rels=[]; sys_counts=[]
    for s in SYSTEMS:
        ee=[e for e in ev if e["system"]==s]
        c=[e["confidence"] for e in ee]
        sys_counts.append(len(ee)); confs+=c; times += [e["time"] for e in ee]
        zones += [e["zone"] for e in ee]; rels += [e["reliability"] for e in ee]
        f[f"count_{s}"]=len(ee)
        f[f"max_{s}"]=max(c) if c else 0.0
        f[f"mean_{s}"]=float(np.mean(c)) if c else 0.0
    n=len(ev)
    f["total_events"]=n; f["overall_max_conf"]=max(confs) if confs else 0.0
    f["overall_mean_conf"]=float(np.mean(confs)) if confs else 0.0
    f["active_systems"]=sum(x>0 for x in sys_counts)
    f["source_entropy"]=shannon_entropy(sys_counts)
    f["mean_reliability"]=float(np.mean(rels)) if rels else 0.0
    f["weighted_confidence"]=float(sum(e["confidence"]*e["reliability"] for e in ev)/(n or 1))

    # Events in the same (system-grouped) order as times/zones/confs above. The submitted
    # version indexed ev[i] here (arrival order), which mis-paired systems/confidences with
    # times/zones in most incident windows; corrected for the camera-ready revision.
    evs=[e for s in SYSTEMS for e in ev if e["system"]==s]
    if n>=2:
        span=float(max(times)-min(times)); std=float(np.std(times))
        f["temporal_span"]=span; f["temporal_cohesion"]=math.exp(-std/28.0)
        ds=[]; close=0; pairs=0; pair_score=0.0; cross_pairs=0
        for i in range(n):
            for j in range(i+1,n):
                d=int(DIST[zones[i],zones[j]]); ds.append(d); pairs+=1
                if d<=1 and abs(times[i]-times[j])<=35: close+=1
                if evs[i]["system"] != evs[j]["system"]:
                    cross_pairs+=1
                    key=frozenset([evs[i]["system"],evs[j]["system"]])
                    weight=PAIR_WEIGHTS.get(key,0.28)
                    pair_score += weight*math.exp(-abs(times[i]-times[j])/TAU)*math.exp(-d/LAMBDA)*min(evs[i]["confidence"],evs[j]["confidence"])
        f["spatial_mean_distance"]=float(np.mean(ds))
        f["spatiotemporal_cohesion"]=close/pairs
        f["semantic_pair_score"]=pair_score
        f["cross_system_pairs"]=cross_pairs
    else:
        f.update(temporal_span=120.0,temporal_cohesion=0.0,spatial_mean_distance=6.0,
                 spatiotemporal_cohesion=0.0,semantic_pair_score=0.0,cross_system_pairs=0)

    # Ordered subtype sequences.
    seq=0.0
    for a,b in SEQUENCES:
        aa=[e for e in ev if e["subtype"]==a]
        bb=[e for e in ev if e["subtype"]==b]
        for x in aa:
            for y in bb:
                if 0 <= y["time"]-x["time"] <= 35 and DIST[x["zone"],y["zone"]] <= 1:
                    seq += min(x["confidence"],y["confidence"])
    f["sequence_support"]=seq
    f["high_risk_evidence"]=sum(HIGH_RISK_SUBTYPES.get(e["subtype"],0)*e["confidence"]*e["reliability"] for e in ev)

    # Interpretable rules; maintenance suppresses operational co-alarms but not access/camera/intrusion evidence.
    active={s for s,c in zip(SYSTEMS,sys_counts) if c>0}
    cross = len(active)
    rule = 0.28*f["high_risk_evidence"] + 0.18*f["semantic_pair_score"] + 0.32*f["sequence_support"]
    if w["after_hours"] and ("access" in active or "camera" in active): rule += 0.42
    if cross>=3: rule += 0.28*(cross-2)
    contradiction=0.0
    if w["maintenance"] and active.issubset({"bms","elevator","camera"}): contradiction=0.8
    if w["maintenance"] and ("bms" in active or "elevator" in active): contradiction += 0.25
    f["contradiction_score"]=contradiction
    f["rule_score"]=rule-contradiction
    return f


def make_dataset(site_ids, n_per_site, seed, missing=0.06, jitter=4.0):
    rng=np.random.default_rng(seed)
    rows=[]
    for sid in site_ids:
        site=SITES[sid]
        for _ in range(n_per_site):
            rows.append(extract_features(generate_window(site,rng,extra_missing=missing,time_jitter=jitter)))
    return pd.DataFrame(rows)

META=["site","label","incident_type"]
RAW_FEATURES=["after_hours","maintenance","hour_sin","hour_cos","weekend","total_events","overall_max_conf","overall_mean_conf"]
for s in SYSTEMS: RAW_FEATURES += [f"count_{s}",f"max_{s}",f"mean_{s}"]
SEMANTIC_FEATURES=["active_systems","source_entropy","mean_reliability","weighted_confidence",
                   "temporal_span","temporal_cohesion","spatial_mean_distance","spatiotemporal_cohesion",
                   "semantic_pair_score","cross_system_pairs","sequence_support","high_risk_evidence",
                   "contradiction_score","rule_score"]
ALL_FEATURES=RAW_FEATURES+SEMANTIC_FEATURES


def best_threshold(y,p):
    pr,rc,th=precision_recall_curve(y,p)
    f=2*pr*rc/(pr+rc+1e-12)
    idx=int(np.nanargmax(f[:-1])) if len(th) else 0
    return float(th[idx]) if len(th) else .5


def metrics(y,p,thr):
    pred=(p>=thr).astype(int)
    prec,rec,f1,_=precision_recall_fscore_support(y,pred,average="binary",zero_division=0)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {"AUROC":roc_auc_score(y,p),"AUPRC":average_precision_score(y,p),
            "Precision":prec,"Recall":rec,"F1":f1,"False alarms / 1000 normal":1000*fp/max(tn+fp,1),
            "Brier":brier_score_loss(y,p),"Threshold":thr,"TP":int(tp),"FP":int(fp),"TN":int(tn),"FN":int(fn)}


def bootstrap_ci(y,p,thr,n=400,seed=99):
    rng=np.random.default_rng(seed); vals={k:[] for k in ["AUPRC","F1","False alarms / 1000 normal"]}
    m=len(y)
    for _ in range(n):
        idx=rng.integers(0,m,m); yy=y[idx]; pp=p[idx]
        if yy.min()==yy.max(): continue
        mm=metrics(yy,pp,thr)
        for k in vals: vals[k].append(mm[k])
    return {k:(float(np.percentile(v,2.5)),float(np.percentile(v,97.5))) for k,v in vals.items()}


def fit_hgb(X,y):
    return HistGradientBoostingClassifier(max_iter=220,learning_rate=.06,max_leaf_nodes=15,
                                          l2_regularization=.8,min_samples_leaf=35,random_state=42).fit(X,y)

def run():
    out=Path(os.environ.get('STEC_OUTPUT_DIR', str(Path(__file__).resolve().parent / 'results'))); out.mkdir(parents=True, exist_ok=True)
    t0=time.time()
    train=make_dataset(range(0,7),2800,1001,missing=.11,jitter=7.0)
    val=make_dataset([7],4800,1002,missing=.11,jitter=7.0)
    test=make_dataset(range(8,12),2800,1003,missing=.11,jitter=7.0)
    train.to_csv(out/'train_features.csv',index=False)
    val.to_csv(out/'validation_features.csv',index=False)
    test.to_csv(out/'test_features.csv',index=False)
    ytr=train.label.to_numpy(); yv=val.label.to_numpy(); yt=test.label.to_numpy()

    results=[]; models={}; probs={}; thresholds={}
    # Max-alarm baseline
    pv=val.overall_max_conf.to_numpy(); pt=test.overall_max_conf.to_numpy(); thr=best_threshold(yv,pv)
    results.append({"Model":"Max-confidence alarm",**metrics(yt,pt,thr)}); probs["Max-confidence alarm"]=pt; thresholds["Max-confidence alarm"]=thr
    # Rule-only
    pv=1/(1+np.exp(-val.rule_score.to_numpy())); pt=1/(1+np.exp(-test.rule_score.to_numpy())); thr=best_threshold(yv,pv)
    results.append({"Model":"Rule-only correlation",**metrics(yt,pt,thr)}); probs["Rule-only correlation"]=pt; thresholds["Rule-only correlation"]=thr
    # Flat ML
    flat=fit_hgb(train[RAW_FEATURES],ytr); pv=flat.predict_proba(val[RAW_FEATURES])[:,1]; pt=flat.predict_proba(test[RAW_FEATURES])[:,1]; thr=best_threshold(yv,pv)
    results.append({"Model":"Flat HGB (raw alarms)",**metrics(yt,pt,thr)}); models['flat']=flat; probs["Flat HGB (raw alarms)"]=pt; thresholds["Flat HGB (raw alarms)"]=thr
    # Proposed
    prop=fit_hgb(train[ALL_FEATURES],ytr); pv=prop.predict_proba(val[ALL_FEATURES])[:,1]; pt=prop.predict_proba(test[ALL_FEATURES])[:,1]; thr=best_threshold(yv,pv)
    results.append({"Model":"Proposed STEC-HGB",**metrics(yt,pt,thr)}); models['proposed']=prop; probs["Proposed STEC-HGB"]=pt; thresholds["Proposed STEC-HGB"]=thr
    res=pd.DataFrame(results)
    res.to_csv(out/'main_results.csv',index=False)

    # Bootstrap CIs
    cis={}
    for name in ["Flat HGB (raw alarms)","Proposed STEC-HGB"]:
        cis[name]=bootstrap_ci(yt,probs[name],thresholds[name])
    (out/'bootstrap_ci.json').write_text(json.dumps(cis,indent=2),encoding='utf-8')

    # Ablations
    groups={
        "Full STEC-HGB":ALL_FEATURES,
        "w/o topology": [x for x in ALL_FEATURES if x not in ["spatial_mean_distance","spatiotemporal_cohesion","semantic_pair_score","cross_system_pairs"]],
        "w/o temporal": [x for x in ALL_FEATURES if x not in ["temporal_span","temporal_cohesion","spatiotemporal_cohesion","sequence_support"]],
        "w/o semantic rules": [x for x in ALL_FEATURES if x not in ["semantic_pair_score","sequence_support","high_risk_evidence","contradiction_score","rule_score"]],
        "w/o reliability": [x for x in ALL_FEATURES if x not in ["mean_reliability","weighted_confidence"]],
    }
    ab=[]
    for name,features in groups.items():
        m=fit_hgb(train[features],ytr); pval=m.predict_proba(val[features])[:,1]; ptest=m.predict_proba(test[features])[:,1]; th=best_threshold(yv,pval)
        ab.append({"Variant":name,**metrics(yt,ptest,th)})
    pd.DataFrame(ab).to_csv(out/'ablation_results.csv',index=False)

    # Robustness to missing events; keep trained models and original thresholds.
    rob=[]
    for miss in [0.0,0.08,0.16,0.24,0.32]:
        ds=make_dataset(range(8,12),900,2000+int(miss*1000),missing=miss,jitter=4.0)
        yy=ds.label.to_numpy()
        for name,model,features in [("Flat HGB",flat,RAW_FEATURES),("STEC-HGB",prop,ALL_FEATURES)]:
            pp=model.predict_proba(ds[features])[:,1]
            mm=metrics(yy,pp,thresholds["Flat HGB (raw alarms)" if name=="Flat HGB" else "Proposed STEC-HGB"])
            rob.append({"Condition":"event missingness","Level":miss,"Model":name,"AUPRC":mm["AUPRC"],"F1":mm["F1"],"FAR":mm["False alarms / 1000 normal"]})
    for jit in [0,8,20,40,60]:
        ds=make_dataset(range(8,12),900,3000+jit,missing=.06,jitter=float(jit))
        yy=ds.label.to_numpy()
        for name,model,features in [("Flat HGB",flat,RAW_FEATURES),("STEC-HGB",prop,ALL_FEATURES)]:
            pp=model.predict_proba(ds[features])[:,1]
            mm=metrics(yy,pp,thresholds["Flat HGB (raw alarms)" if name=="Flat HGB" else "Proposed STEC-HGB"])
            rob.append({"Condition":"timestamp jitter (s)","Level":jit,"Model":name,"AUPRC":mm["AUPRC"],"F1":mm["F1"],"FAR":mm["False alarms / 1000 normal"]})
    robdf=pd.DataFrame(rob); robdf.to_csv(out/'robustness_results.csv',index=False)

    # Incident-type recall for proposed
    pred=(probs["Proposed STEC-HGB"]>=thresholds["Proposed STEC-HGB"]).astype(int)
    tmp=test[["incident_type","label"]].copy(); tmp['pred']=pred
    it=[]
    for typ,g in tmp[tmp.label==1].groupby('incident_type'):
        it.append({"Incident type":typ,"n":len(g),"Recall":float(g.pred.mean())})
    pd.DataFrame(it).sort_values('Recall',ascending=False).to_csv(out/'incident_recall.csv',index=False)

    # Feature importance via permutation on a subsample, using AUPRC drop.
    rng=np.random.default_rng(123); idx=rng.choice(len(test),size=min(4500,len(test)),replace=False)
    base=average_precision_score(yt[idx],prop.predict_proba(test.iloc[idx][ALL_FEATURES])[:,1])
    imp=[]
    Xs=test.iloc[idx][ALL_FEATURES].copy(); ys=yt[idx]
    for feat in ALL_FEATURES:
        drops=[]
        for r in range(4):
            xp=Xs.copy(); xp[feat]=rng.permutation(xp[feat].to_numpy())
            drops.append(base-average_precision_score(ys,prop.predict_proba(xp)[:,1]))
        imp.append({"Feature":feat,"AUPRC decrease":float(np.mean(drops)),"Std":float(np.std(drops))})
    impdf=pd.DataFrame(imp).sort_values('AUPRC decrease',ascending=False)
    impdf.to_csv(out/'permutation_importance.csv',index=False)

    # PR curve
    plt.figure(figsize=(6.2,4.2))
    for name in ["Max-confidence alarm","Rule-only correlation","Flat HGB (raw alarms)","Proposed STEC-HGB"]:
        pr,rc,_=precision_recall_curve(yt,probs[name])
        plt.plot(rc,pr,label=f"{name} (AP={average_precision_score(yt,probs[name]):.3f})")
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.xlim(0,1); plt.ylim(0,1.02); plt.grid(True,alpha=.25); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(out/'precision_recall_curve.png',dpi=220); plt.close()

    # Missingness robustness plot
    mdf=robdf[robdf.Condition=='event missingness']
    plt.figure(figsize=(6.2,4.0))
    for name,g in mdf.groupby('Model'):
        plt.plot(g.Level*100,g.AUPRC,marker='o',label=name)
    plt.xlabel('Event missingness (%)'); plt.ylabel('AUPRC'); plt.grid(True,alpha=.25); plt.legend(); plt.tight_layout()
    plt.savefig(out/'missingness_robustness.png',dpi=220); plt.close()

    # Feature importance plot top 10
    top=impdf.head(10).sort_values('AUPRC decrease')
    plt.figure(figsize=(6.2,4.5)); plt.barh(top.Feature,top['AUPRC decrease']); plt.xlabel('Permutation AUPRC decrease'); plt.tight_layout()
    plt.savefig(out/'feature_importance.png',dpi=220); plt.close()

    # Save summary
    summary={"runtime_sec":time.time()-t0,"n_train":len(train),"n_val":len(val),"n_test":len(test),
             "positive_rate_train":float(train.label.mean()),"positive_rate_test":float(test.label.mean()),
             "features_raw":len(RAW_FEATURES),"features_all":len(ALL_FEATURES)}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(res.to_string(index=False))
    print('\nAblation')
    print(pd.DataFrame(ab)[['Variant','AUPRC','F1','False alarms / 1000 normal']].to_string(index=False))
    print('\nSummary',summary)

if __name__=='__main__': run()
