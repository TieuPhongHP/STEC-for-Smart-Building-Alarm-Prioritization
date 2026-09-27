"""Supplementary experiments for the camera-ready revision of the STEC-HGB paper
(ICDSAIA 2026, paper 253). The main results in run_experiment.py are NOT changed.

Experiments (all use the unchanged generator, features, classifier and thresholds unless stated):
  E1  site-level evaluation: per-site results on the fixed test sites and a 12-fold
      leave-one-site-out comparison with a Wilcoxon signed-rank test.
  E2  sensitivity of STEC to its hand-set parameters: tau, lambda, pair weights, sequence templates.
  E3  generator-decoupling stress tests: dispersed/slow incidents, unseen incident families,
      leave-one-family-out, correlated upstream faults, and site drift.
  E4  learned-graph baseline: an attention GNN over event graphs (with and without zone distance).
  E5  evidence-path fidelity: deletion test on the strongest evidence pair.

Usage:  python revision_experiments.py            (writes to revision_results/)
Requires: numpy, pandas, scikit-learn, scipy, torch (CPU).
"""
from __future__ import annotations
import copy, json, math, os, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score
import run_experiment as R

OUT = Path(os.environ.get("STEC_REV_DIR", "revision_results")); OUT.mkdir(parents=True, exist_ok=True)
MISS, JIT = 0.11, 7.0

# ---------------------------------------------------------------- raw windows and parameterised features
def gen_windows(site_ids, n, seed, missing=MISS, jitter=JIT, gen=None, sites=None):
    """Same RNG usage as R.make_dataset, but keeps the raw windows."""
    gen = gen or R.generate_window; sites = sites or R.SITES
    rng = np.random.default_rng(seed); out = []
    for sid in site_ids:
        for _ in range(n):
            out.append(gen(sites[sid], rng, extra_missing=missing, time_jitter=jitter))
    return out

DEFAULT = dict(tau=30.0, lam=1.6, weights=R.PAIR_WEIGHTS, w0=0.28, seqs=R.SEQUENCES)

def features(w, tau=30.0, lam=1.6, weights=None, w0=0.28, seqs=None):
    """Copy of R.extract_features (corrected version) with the hand-set STEC parameters exposed."""
    weights = R.PAIR_WEIGHTS if weights is None else weights
    seqs = R.SEQUENCES if seqs is None else seqs
    ev = w["events"]; S = R.SYSTEMS
    f = {"site": w["site"], "label": w["label"], "incident_type": w["incident_type"],
         "after_hours": w["after_hours"], "maintenance": w["maintenance"],
         "hour_sin": math.sin(2*math.pi*w["hour"]/24), "hour_cos": math.cos(2*math.pi*w["hour"]/24),
         "weekend": int(w["weekday"] >= 5)}
    confs=[]; times=[]; zones=[]; rels=[]; sys_counts=[]
    for s in S:
        ee=[e for e in ev if e["system"]==s]; c=[e["confidence"] for e in ee]
        sys_counts.append(len(ee)); confs+=c; times+=[e["time"] for e in ee]
        zones+=[e["zone"] for e in ee]; rels+=[e["reliability"] for e in ee]
        f[f"count_{s}"]=len(ee); f[f"max_{s}"]=max(c) if c else 0.0; f[f"mean_{s}"]=float(np.mean(c)) if c else 0.0
    evs=[e for s in S for e in ev if e["system"]==s]   # same order as times/zones
    P = evs
    n=len(ev)
    f["total_events"]=n; f["overall_max_conf"]=max(confs) if confs else 0.0
    f["overall_mean_conf"]=float(np.mean(confs)) if confs else 0.0
    f["active_systems"]=sum(x>0 for x in sys_counts); f["source_entropy"]=R.shannon_entropy(sys_counts)
    f["mean_reliability"]=float(np.mean(rels)) if rels else 0.0
    f["weighted_confidence"]=float(sum(e["confidence"]*e["reliability"] for e in ev)/(n or 1))
    best=(0.0,None)
    if n>=2:
        span=float(max(times)-min(times)); std=float(np.std(times))
        f["temporal_span"]=span; f["temporal_cohesion"]=math.exp(-std/28.0)
        ds=[]; close=0; pairs=0; ps=0.0; cross=0
        for i in range(n):
            for j in range(i+1,n):
                d=int(R.DIST[zones[i],zones[j]]); ds.append(d); pairs+=1
                if d<=1 and abs(times[i]-times[j])<=35: close+=1
                if P[i]["system"]!=P[j]["system"]:
                    cross+=1; key=frozenset([P[i]["system"],P[j]["system"]])
                    phi=weights.get(key,w0)*math.exp(-abs(times[i]-times[j])/tau)*math.exp(-d/lam)*min(P[i]["confidence"],P[j]["confidence"])
                    ps+=phi
                    if phi>best[0]: best=(phi,(i,j))
        f["spatial_mean_distance"]=float(np.mean(ds)); f["spatiotemporal_cohesion"]=close/pairs
        f["semantic_pair_score"]=ps; f["cross_system_pairs"]=cross
    else:
        f.update(temporal_span=120.0,temporal_cohesion=0.0,spatial_mean_distance=6.0,
                 spatiotemporal_cohesion=0.0,semantic_pair_score=0.0,cross_system_pairs=0)
    seq=0.0
    for a,b in seqs:
        aa=[e for e in ev if e["subtype"]==a]; bb=[e for e in ev if e["subtype"]==b]
        for x in aa:
            for y in bb:
                if 0<=y["time"]-x["time"]<=35 and R.DIST[x["zone"],y["zone"]]<=1: seq+=min(x["confidence"],y["confidence"])
    f["sequence_support"]=seq
    f["high_risk_evidence"]=sum(R.HIGH_RISK_SUBTYPES.get(e["subtype"],0)*e["confidence"]*e["reliability"] for e in ev)
    active={s for s,c in zip(S,sys_counts) if c>0}; k=len(active)
    rule=0.28*f["high_risk_evidence"]+0.18*f["semantic_pair_score"]+0.32*f["sequence_support"]
    if w["after_hours"] and ("access" in active or "camera" in active): rule+=0.42
    if k>=3: rule+=0.28*(k-2)
    con=0.0
    if w["maintenance"] and active.issubset({"bms","elevator","camera"}): con=0.8
    if w["maintenance"] and ("bms" in active or "elevator" in active): con+=0.25
    f["contradiction_score"]=con; f["rule_score"]=rule-con
    f["_best_pair"]=None if best[1] is None else (evs[best[1][0]],evs[best[1][1]])
    return f

def table(ws, **kw):
    return pd.DataFrame([features(w, **kw) for w in ws])

def fit_eval(tr, va, te, feats):
    m = R.fit_hgb(tr[feats], tr.label.to_numpy())
    thr = R.best_threshold(va.label.to_numpy(), m.predict_proba(va[feats])[:,1])
    p = m.predict_proba(te[feats])[:,1]
    return m, thr, p, R.metrics(te.label.to_numpy(), p, thr)

def short(m): return {"AUPRC":m["AUPRC"],"F1":m["F1"],"FAR":m["False alarms / 1000 normal"],"Recall":m["Recall"],"Precision":m["Precision"]}

# ---------------------------------------------------------------- base split (identical to the paper)
t0=time.time()
W_tr=gen_windows(range(0,7),2800,1001); W_va=gen_windows([7],4800,1002); W_te=gen_windows(range(8,12),2800,1003)
TR,VA,TE=table(W_tr),table(W_va),table(W_te)
ref=pd.read_csv(Path(__file__).resolve().parent/"results"/"test_features.csv") if (Path(__file__).resolve().parent/"results"/"test_features.csv").exists() else None
if ref is not None:
    assert np.allclose(ref[R.ALL_FEATURES].to_numpy(), TE[R.ALL_FEATURES].to_numpy()), "feature copy diverges from run_experiment.py"
flat,thr_f,p_f,m_f=fit_eval(TR,VA,TE,R.RAW_FEATURES)
stec,thr_s,p_s,m_s=fit_eval(TR,VA,TE,R.ALL_FEATURES)
base={"Flat HGB":short(m_f),"STEC-HGB":short(m_s)}
print("base",json.dumps(base,indent=1)); results={"base":base}

# ---------------------------------------------------------------- E1 site-level
rows=[]
for sid in range(8,12):
    mk=TE.site.to_numpy()==sid; y=TE.label.to_numpy()[mk]
    for name,p,thr in [("Flat HGB",p_f,thr_f),("STEC-HGB",p_s,thr_s)]:
        rows.append({"split":"fixed","site":sid,"model":name,**short(R.metrics(y,p[mk],thr))})
ALL=[gen_windows([s],2800,5000+s) for s in range(12)]
ALLT=[table(w) for w in ALL]
for k in range(12):
    v=(k+1)%12; tr=pd.concat([ALLT[i] for i in range(12) if i not in (k,v)])
    for name,feats in [("Flat HGB",R.RAW_FEATURES),("STEC-HGB",R.ALL_FEATURES)]:
        _,_,_,m=fit_eval(tr,ALLT[v],ALLT[k],feats); rows.append({"split":"LOSO","site":k,"model":name,**short(m)})
site=pd.DataFrame(rows); site.to_csv(OUT/"E1_site_level.csv",index=False)
lo=site[site.split=="LOSO"].pivot(index="site",columns="model")
e1={}
for met in ["AUPRC","F1","FAR"]:
    a=lo[(met,"STEC-HGB")].to_numpy(); b=lo[(met,"Flat HGB")].to_numpy()
    better=(a<b) if met=="FAR" else (a>b)
    e1[met]={"STEC_mean":float(a.mean()),"STEC_sd":float(a.std(ddof=1)),"Flat_mean":float(b.mean()),"Flat_sd":float(b.std(ddof=1)),
             "STEC_min":float(a.min()),"STEC_max":float(a.max()),"sites_better":int(better.sum()),"wilcoxon_p":float(wilcoxon(a,b).pvalue)}
results["E1"]=e1; print("E1",json.dumps(e1,indent=1),time.time()-t0)

# ---------------------------------------------------------------- E2 parameter sensitivity (retrain STEC-HGB per setting)
def run_setting(**kw):
    tr,va,te=table(W_tr,**kw),table(W_va,**kw),table(W_te,**kw)
    return short(fit_eval(tr,va,te,R.ALL_FEATURES)[3])
rng=np.random.default_rng(77); rows=[]
for tau in [10.0,60.0,90.0]: rows.append({"setting":f"tau={tau:g} s","group":"tau",**run_setting(tau=tau)})
for lam in [0.8,3.2]: rows.append({"setting":f"lambda={lam:g}","group":"lambda",**run_setting(lam=lam)})
pairs=[frozenset([a,b]) for i,a in enumerate(R.SYSTEMS) for b in R.SYSTEMS[i+1:]]
rows.append({"setting":"uniform weights (all 1.0)","group":"weights",**run_setting(weights={p:1.0 for p in pairs},w0=1.0)})
vals=[R.PAIR_WEIGHTS.get(p,0.28) for p in pairs]
for r in range(5):
    perm=rng.permutation(vals); rows.append({"setting":f"shuffled weights #{r+1}","group":"weights_shuffled",**run_setting(weights=dict(zip(pairs,perm)),w0=0.28)})
for r in range(5):
    noisy={p:float(v*rng.lognormal(0,0.5)) for p,v in zip(pairs,vals)}
    rows.append({"setting":f"noisy weights (x lognormal 0.5) #{r+1}","group":"weights_noisy",**run_setting(weights=noisy,w0=0.28)})
rows.append({"setting":"no sequence templates","group":"sequences",**run_setting(seqs=[])})
subs=sorted({x for v in R.NORMAL_SUBTYPES.values() for x in v}|{x[1] for t in R.TEMPLATES.values() for x in t})
for r in range(5):
    rs=[tuple(rng.choice(subs,2,replace=False)) for _ in range(8)]
    rows.append({"setting":f"random sequence templates #{r+1}","group":"sequences_random",**run_setting(seqs=rs)})
sens=pd.DataFrame(rows); sens.to_csv(OUT/"E2_sensitivity.csv",index=False)
results["E2"]=sens.groupby("group")[["AUPRC","F1","FAR"]].agg(["mean","min","max"]).round(4).to_dict()
print(sens.round(3).to_string(),time.time()-t0)

# ---------------------------------------------------------------- E3 generator decoupling
def variant_generator(slow=1.0,p_same=None,templates=None,cascade=0.0,drift=False):
    def gen(site,rng,force_label=None,extra_missing=0.11,time_jitter=7.0):
        saved=(R.TEMPLATES,R.neighbor_zone)
        if templates is not None: R.TEMPLATES=templates
        if p_same is not None:
            nz=saved[1]; R.neighbor_zone=lambda z,r,p=.72: nz(z,r,p_same if p in (.78,) else p)
        try:
            if slow!=1.0:
                T=R.TEMPLATES; R.TEMPLATES={k:[(s,q,dt*slow,c) for s,q,dt,c in v] for k,v in T.items()}
            w=R.generate_window(site,rng,force_label,extra_missing,time_jitter)
        finally:
            R.TEMPLATES,R.neighbor_zone=saved
        if cascade and not w["label"] and rng.random()<cascade:
            # correlated upstream fault (e.g. PoE/network switch outage): several subsystems report
            # tamper/offline events in the same area within seconds -> incident-like but benign.
            z=int(rng.integers(0,R.N_ZONES))
            for s,q in [("camera","camera_tamper"),("access","reader_tamper"),("bms","device_offline")]:
                R.add_event(w["events"],s,q,R.neighbor_zone(z,rng,.8),float(rng.normal(0,4)),float(rng.beta(6,1.8)),site.reliab[s],rng,miss=0.05)
        return w
    return gen

def drift_sites():
    S=copy.deepcopy(R.SITES)
    for s in S:
        s.rates={k:v*1.6 for k,v in s.rates.items()}; s.reliab={k:max(.5,v-.10) for k,v in s.reliab.items()}
        s.false_burst=min(.5,s.false_burst*1.5); s.clock_bias=s.clock_bias*3
    return S

NOVEL={
 "vehicle_to_restricted_floor":[("parking","plate_read",-12,0.80),("elevator","car_call",-1,0.78),("intrusion","motion_alarm",8,0.80),("bms","occupancy_change",15,0.74)],
 "door_held_intrusion":[("access","door_held",-6,0.82),("intrusion","contact_alarm",2,0.84),("camera","motion_analytics",9,0.77)],
 "plant_room_intrusion":[("intrusion","motion_alarm",-6,0.83),("bms","device_offline",2,0.80),("elevator","service_mode",10,0.76)],
 "panel_tamper":[("intrusion","panel_fault",-4,0.86),("camera","camera_tamper",3,0.82),("access","door_held",11,0.78)],
 "loading_dock_breach":[("parking","barrier_open",-10,0.80),("intrusion","contact_alarm",-1,0.84),("camera","person_detected",6,0.78)],
 "stairwell_bypass":[("access","door_held",-8,0.80),("elevator","door_fault",0,0.74),("intrusion","occupancy_mismatch",7,0.78)],
 "hvac_room_tamper":[("bms","temperature_alarm",-5,0.80),("intrusion","panel_fault",2,0.82),("camera","motion_analytics",8,0.76)],
 "service_lift_misuse":[("elevator","service_mode",-9,0.82),("access","access_granted",-2,0.76),("bms","occupancy_change",9,0.74)],
}  # 8 families so that the generator's family probabilities still apply
def evaluate_on(ws,label):
    te=table(ws); y=te.label.to_numpy(); out={}
    for name,m,thr,feats in [("Flat HGB",flat,thr_f,R.RAW_FEATURES),("STEC-HGB",stec,thr_s,R.ALL_FEATURES)]:
        out[name]=short(R.metrics(y,m.predict_proba(te[feats])[:,1],thr))
    return {"condition":label,**{f"{k}_{m}":v for k,d in out.items() for m,v in d.items()}}
rows=[evaluate_on(W_te,"reference test sites")]
rows.append(evaluate_on(gen_windows(range(8,12),2800,1003,gen=variant_generator(slow=3.0,p_same=0.40)),"incidents 3x slower and spatially dispersed"))
rows.append(evaluate_on(gen_windows(range(8,12),2800,1003,gen=variant_generator(templates=NOVEL)),"unseen incident families only"))
rows.append(evaluate_on(gen_windows(range(8,12),2800,1003,gen=variant_generator(cascade=0.10)),"correlated upstream faults in 10% of normal windows"))
rows.append(evaluate_on(gen_windows(range(8,12),2800,1003,sites=drift_sites()),"site drift (rates x1.6, reliability -0.1, clock bias x3)"))
shift=pd.DataFrame(rows); shift.to_csv(OUT/"E3_generator_shift.csv",index=False)
print(shift.round(3).T.to_string(),time.time()-t0)

# leave-one-family-out
rows=[]
for fam in R.TEMPLATES:
    keep=~((TR.label==1)&(TR.incident_type==fam))
    kv=~((VA.label==1)&(VA.incident_type==fam))
    for name,feats in [("Flat HGB",R.RAW_FEATURES),("STEC-HGB",R.ALL_FEATURES)]:
        m,thr,p,_=fit_eval(TR[keep],VA[kv],TE,feats)
        mk=(TE.incident_type==fam).to_numpy()
        full_p = (p_f if name=="Flat HGB" else p_s); full_thr=(thr_f if name=="Flat HGB" else thr_s)
        rows.append({"family":fam,"model":name,"recall_seen":float((full_p[mk]>=full_thr).mean()),"recall_unseen":float((p[mk]>=thr).mean())})
lofo=pd.DataFrame(rows); lofo.to_csv(OUT/"E3_leave_one_family_out.csv",index=False)
print(lofo.round(3).to_string(),time.time()-t0)

# ---------------------------------------------------------------- E4 learned-graph baseline (attention GNN)
import torch, torch.nn as nn
torch.set_num_threads(2)
SUBS={q:i for i,q in enumerate(subs)}; NMAX=16
def tensorise(ws):
    B=len(ws); X=np.zeros((B,NMAX,6+len(SUBS)+12+3),np.float32); M=np.zeros((B,NMAX),bool)
    T=np.zeros((B,NMAX),np.float32); Z=np.zeros((B,NMAX),np.int64); C=np.zeros((B,5),np.float32); y=np.zeros(B,np.float32)
    for b,w in enumerate(ws):
        C[b]=[w["after_hours"],w["maintenance"],math.sin(2*math.pi*w["hour"]/24),math.cos(2*math.pi*w["hour"]/24),int(w["weekday"]>=5)]
        y[b]=w["label"]
        for i,e in enumerate(w["events"][:NMAX]):
            X[b,i,R.SYSTEM_IDX[e["system"]]]=1; X[b,i,6+SUBS[e["subtype"]]]=1; X[b,i,6+len(SUBS)+e["zone"]]=1
            X[b,i,-3:]=[e["confidence"],e["reliability"],e["time"]/60.0]; M[b,i]=True; T[b,i]=e["time"]; Z[b,i]=e["zone"]
    dt=np.abs(T[:,:,None]-T[:,None,:])/60.0; dz=R.DIST[Z[:,:,None],Z[:,None,:]].astype(np.float32)/5.0
    return [torch.tensor(a) for a in (X,M,dt,dz,C,y)]
class EventGNN(nn.Module):
    def __init__(s,din,topo,h=64,layers=2):
        super().__init__(); s.topo=topo; s.inp=nn.Linear(din,h); ne=2 if topo else 1
        s.att=nn.ModuleList([nn.Sequential(nn.Linear(2*h+ne,h),nn.ReLU(),nn.Linear(h,1)) for _ in range(layers)])
        s.msg=nn.ModuleList([nn.Linear(h+ne,h) for _ in range(layers)]); s.upd=nn.ModuleList([nn.GRUCell(h,h) for _ in range(layers)])
        s.out=nn.Sequential(nn.Linear(2*h+5,h),nn.ReLU(),nn.Dropout(0.1),nn.Linear(h,1))
    def forward(s,X,M,dt,dz,C):
        h=torch.relu(s.inp(X)); B,N,H=h.shape
        e=dt.unsqueeze(-1) if not s.topo else torch.stack([dt,dz],-1)
        pair=M.unsqueeze(1)&M.unsqueeze(2)&~torch.eye(N,dtype=torch.bool).unsqueeze(0)
        for att,msg,upd in zip(s.att,s.msg,s.upd):
            hi=h.unsqueeze(2).expand(B,N,N,H); hj=h.unsqueeze(1).expand(B,N,N,H)
            a=att(torch.cat([hi,hj,e],-1)).squeeze(-1).masked_fill(~pair,-1e9)
            a=torch.softmax(a,-1)*pair.any(-1,keepdim=True)
            m=(a.unsqueeze(-1)*msg(torch.cat([hj,e],-1))).sum(2)
            h=upd(m.reshape(-1,H),h.reshape(-1,H)).reshape(B,N,H)*M.unsqueeze(-1)
        cnt=M.sum(1,keepdim=True).clamp(min=1)
        g=torch.cat([h.sum(1)/cnt,h.masked_fill(~M.unsqueeze(-1),-1e9).max(1).values.clamp(min=0),C],-1)
        return s.out(g).squeeze(-1)
Ttr,Tva,Tte=tensorise(W_tr),tensorise(W_va),tensorise(W_te)
def predict(model,T):
    model.eval(); ps=[]
    with torch.no_grad():
        for i in range(0,len(T[0]),1024): ps.append(torch.sigmoid(model(*[t[i:i+1024] for t in T[:5]])))
    return torch.cat(ps).numpy()
def train_gnn(topo,seed,epochs=25):
    torch.manual_seed(seed); np.random.seed(seed)
    model=EventGNN(Ttr[0].shape[-1],topo); opt=torch.optim.Adam(model.parameters(),lr=2e-3,weight_decay=1e-5)
    lossf=nn.BCEWithLogitsLoss(); n=len(Ttr[0]); best=(-1,None)
    for ep in range(epochs):
        model.train(); perm=torch.randperm(n)
        for i in range(0,n,256):
            idx=perm[i:i+256]; opt.zero_grad(); loss=lossf(model(*[t[idx] for t in Ttr[:5]]),Ttr[5][idx]); loss.backward(); opt.step()
        ap=average_precision_score(Tva[5].numpy(),predict(model,Tva))
        if ap>best[0]: best=(ap,copy.deepcopy(model.state_dict()))
    model.load_state_dict(best[1]); pv=predict(model,Tva); thr=R.best_threshold(Tva[5].numpy(),pv)
    return R.metrics(Tte[5].numpy(),predict(model,Tte),thr)
rows=[]
for topo in [False,True]:
    for seed in [0,1,2]:
        m=train_gnn(topo,seed); rows.append({"model":"Event-GNN + zone distance" if topo else "Event-GNN (learned relations)","seed":seed,**short(m),"AUROC":m["AUROC"],"Brier":m["Brier"]})
        print(rows[-1],time.time()-t0)
gnn=pd.DataFrame(rows); gnn.to_csv(OUT/"E4_gnn_baseline.csv",index=False)
results["E4"]=gnn.groupby("model")[["AUROC","AUPRC","F1","FAR","Brier"]].agg(["mean","std"]).round(4).to_dict()

# ---------------------------------------------------------------- E5 evidence-path fidelity (deletion test)
# For correctly detected incident windows with >=3 events, delete the two events of the strongest
# evidence pair (max phi_ij) and compare the probability drop with deleting two random events.
rng=np.random.default_rng(5); de=[]; dr=[]; flip_e=[]; flip_r=[]
def prob(w,evs):
    ww=dict(w); ww["events"]=evs; f=pd.DataFrame([features(ww)]); return stec.predict_proba(f[R.ALL_FEATURES])[:,1][0]
for i in np.where((TE.label.to_numpy()==1)&(p_s>=thr_s))[0]:
    w=W_te[i]; bp=TE["_best_pair"].iloc[i]; evs=w["events"]
    if bp is None or len(evs)<3: continue
    pe=prob(w,[e for e in evs if e is not bp[0] and e is not bp[1]])
    k=rng.choice(len(evs),2,replace=False); prn=prob(w,[e for q,e in enumerate(evs) if q not in k])
    de.append(p_s[i]-pe); dr.append(p_s[i]-prn); flip_e.append(pe<thr_s); flip_r.append(prn<thr_s)
e5={"n_windows":len(de),"mean_drop_evidence_pair":float(np.mean(de)),"mean_drop_random_pair":float(np.mean(dr)),
    "flip_rate_evidence_pair":float(np.mean(flip_e)),"flip_rate_random_pair":float(np.mean(flip_r)),"wilcoxon_p":float(wilcoxon(de,dr).pvalue)}
results["E5"]=e5; print("E5",e5)
results["runtime_sec"]=time.time()-t0
def sk(d): return {("|".join(map(str,k)) if isinstance(k,tuple) else k):(sk(v) if isinstance(v,dict) else v) for k,v in d.items()}
(OUT/"revision_summary.json").write_text(json.dumps(sk(results),indent=2,default=str))
print("done",time.time()-t0)
