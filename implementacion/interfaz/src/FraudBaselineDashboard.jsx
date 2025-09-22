import React, { useEffect, useMemo, useState } from "react";
import "./fraud-dashboard.css";
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, ReferenceLine, CartesianGrid, Legend,
  BarChart, Bar
} from "recharts";

/**
  FraudBAFDashboard
  - Lee JSON (metrics_baf_base.json) con roc.points + confusion
  - diseño 
  
 
  Fuentes de datos (en este orden):
   1) props.data
   2) window.METRICS_BAF_DATA o window.FRAUD_DASH_DATA
   3) fetch(src)  (por defecto "/metrics_baf_base.json")
 */



//UI elementos


//cartitas
function MetricCard({ label, value, hint, format = "auto" }) {
  const fmt = (v) => {
    if (v == null || Number.isNaN(v)) return "—";
    if (format === "pct1") return (v * 100).toFixed(1) + "%";
    if (format === "pct0") return (v * 100).toFixed(0) + "%";
    if (typeof v === "number") return Math.abs(v) <= 1 ? v.toFixed(3) : v.toLocaleString();
    return String(v);
  };
  return (
    <div className="card metric-card" role="group" aria-label={label}>
      <div className="metric-label">{label}</div>
      <div className="metric-value tabnums">{fmt(value)}</div>
      {hint && <div className="metric-hint subtle" style={{fontSize:12}}>{hint}</div>}
    </div>
  );
}

//normaliza
function normalize(input) {
  if (!input) return null;

  //Formato BAF (metrics_baf_base.json)
  const isBAF = typeof input["val_auc"] === "number" && typeof input["test_auc"] === "number";
  if (isBAF) {
    const baf = {
      val_auc: input["val_auc"],
      val_ap: input["val_ap"],
      thr: input["threshold_at_fpr=0.05(val)"] ?? null,
      val_fpr: input["val_fpr"],
      val_recall: input["val_tpr(recall)"],
      test_auc: input["test_auc"],
      test_ap: input["test_ap"],
      test_fpr_at_val_thr: input["test_fpr@val_thr"],
      test_recall_at_val_thr: input["test_recall@val_thr"],
      test_recall_at_exact5_curve: input["test_recall_at_exact5_curve"],
      fpr_age_ge50: input["fpr_age_ge50"],
      fpr_age_lt50: input["fpr_age_lt50"],
      fpr_ratio_age: input["fpr_ratio_age"],
    };

    // ROC (TEST)
    let rocPoints = [];
    if (input.roc && Array.isArray(input.roc.points)) {
      rocPoints = input.roc.points.map(p => ({ x: Number(p.fpr) || 0, y: Number(p.tpr) || 0 }));
    } else if (Array.isArray(input.roc)) {
      rocPoints = input.roc.map(p => ({ x: Number(p.fpr) || 0, y: Number(p.tpr) || 0 }));
    }

    const roc = {
      points: rocPoints,
      auc: typeof input.roc?.auc === "number" ? input.roc.auc : baf.test_auc,
      modelName: input.roc?.modelName || "LightGBM (operando @ 5% FPR val)"
    };

    //confusion
    let confusion = { labels: ["Pred 0","Pred 1"], rows: [{name:"Actual 0", values:[0,0]}, {name:"Actual 1", values:[0,0]}] };
    if (input.confusion && input.confusion.labels && input.confusion.rows) {
      confusion = input.confusion;
    } else if (typeof input.tn === "number") {
      confusion = {
        labels: ["Pred 0","Pred 1"],
        rows: [
          { name: "Actual 0", values: [input.tn|0, input.fp|0] },
          { name: "Actual 1", values: [input.fn|0, input.tp|0] },
        ],
      };
    }

    return {
      mode: "BAF",
      project: { name: "Detección de fraudes", version: "BAF · LGBM", date: new Date().toISOString().slice(0,10) },
      summary: { auc: baf.test_auc, accuracy: null, precision: null, recall: baf.test_recall_at_val_thr, threshold: baf.thr },
      roc,
      confusion,
      baf,
    };
  }

  // formato dashboard
  if (input.summary && input.roc && input.confusion) {
    return {
      mode: "BASE",
      project: input.project || { name: "Detección de fraudes", version: "v0.1-baseline", date: new Date().toISOString().slice(0,10) },
      summary: input.summary,
      roc: {
        points: (input.roc.points || []).map(p => ({ x: p.fpr ?? p.x ?? 0, y: p.tpr ?? p.y ?? 0 })),
        auc: input.roc.auc,
        modelName: input.roc.modelName || "Modelo",
      },
      confusion: input.confusion,
      leaderboard: input.leaderboard || [],
      nextSteps: input.nextSteps || [],
    };
  }

  // ---- Formato resultados.json plano ----
  if (typeof input.auc === "number") {
    return {
      mode: "PLAIN",
      project: { name: "Detección de fraudes", version: "v0.1-baseline", date: new Date().toISOString().slice(0,10) },
      summary: { auc: input.auc, accuracy: input.accuracy, precision: input.precision, recall: input.recall, threshold: 0.5 },
      roc: { points: (input.roc || []).map(p => ({ x: p.fpr ?? 0, y: p.tpr ?? 0 })), auc: input.auc, modelName: "LogReg (baseline)" },
      confusion: input.confusion || { labels:["Pred 0","Pred 1"], rows:[{name:"Actual 0", values:[0,0]}, {name:"Actual 1", values:[0,0]}] },
    };
  }

  return null;
}

export default function FraudBAFDashboard({ src = "/metrics_baf_base.json", data: dataProp, scores = null, yTrue = null }) {
  //hooks
  const pickWindow = () => (typeof window !== "undefined" ? (window.METRICS_BAF_DATA || window.FRAUD_DASH_DATA) : null);
  const initialNorm = normalize(dataProp || pickWindow());

  const [hydrated, setHydrated] = useState(() => initialNorm);
  const [loading, setLoading] = useState(!initialNorm);
  const [error, setError] = useState(null);
  const [threshold, setThreshold] = useState(() => initialNorm?.summary?.threshold ?? 0.5);

  // fetch
  useEffect(() => {
    if (hydrated) { setLoading(false); return; }
    let cancelled = false;
    async function pull(){
      try{
        const res = await fetch(src, { cache: "no-store" });
        if(!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        const norm = normalize(json);
        if(!norm) throw new Error("Formato JSON invalido");
        if(!cancelled) {
          setHydrated(norm);
          setThreshold(norm?.summary?.threshold ?? 0.5);
        }
      }catch(e){ if(!cancelled) setError(e.message || String(e)); }
      finally{ if(!cancelled) setLoading(false); }
    }
    pull();
    return () => { cancelled = true; };
  }, [src, hydrated, dataProp]);

  // Derivados (null-safe)
  const project   = hydrated?.project ?? { name:"Detección de fraudes", version:"—", date:new Date().toISOString().slice(0,10) };
  const summary   = hydrated?.summary ?? { auc:null, accuracy:null, precision:null, recall:null, threshold:threshold };
  const roc       = hydrated?.roc ?? { points:[], auc: summary.auc ?? null, modelName:"" };
  const confusion = hydrated?.confusion ?? { labels:["Pred 0","Pred 1"], rows:[{name:"Actual 0", values:[0,0]},{name:"Actual 1", values:[0,0]}] };
  const baf       = hydrated?.baf;

  //matriz
  const liveConfusion = useMemo(() => {
    if (!scores || !yTrue || scores.length !== yTrue.length) return null;
    let tp=0, tn=0, fp=0, fn=0;
    for (let i=0; i<scores.length; i++){
      const pred = scores[i] >= threshold ? 1 : 0;
      const y = yTrue[i] ? 1 : 0;
      if (pred===1 && y===1) tp++; else if (pred===1 && y===0) fp++; else if (pred===0 && y===1) fn++; else tn++;
    }
    return { labels:["Pred 0","Pred 1"], rows:[{name:"Actual 0", values:[tn, fp]}, {name:"Actual 1", values:[fn, tp]}] };
  }, [scores, yTrue, threshold]);

  const cm = liveConfusion || confusion;

  const tnr = useMemo(() => {
    const [tn, fp] = cm.rows?.[0]?.values ?? [0,0];
    const denom = (tn||0)+(fp||0); return denom ? (tn||0)/denom : 0;
  }, [cm]);
  const tpr = useMemo(() => {
    const [fn, tp] = cm.rows?.[1]?.values ?? [0,0];
    const denom = (tp||0)+(fn||0); return denom ? (tp||0)/denom : 0;
  }, [cm]);

  const fairnessData = useMemo(() => (
    baf ? [
      { group: "age<50", fpr: baf.fpr_age_lt50 ?? 0 },
      { group: "age≥50", fpr: baf.fpr_age_ge50 ?? 0 },
    ] : []
  ), [baf]);

  // Render
  return (
    <div className="dash">
 

      {/* header */}
      <header className="dash-header">
        <div>
          <h2 className="dash-title">{project.name}</h2>
        </div>
      
      </header>

      {/*eatado*/}
      {loading && <div className="card subtle" style={{marginBottom:12}}>Cargando métricas…</div>}
      {error && <div className="card" style={{marginBottom:12, color:'#b91c1c'}}>Error: {error}</div>}

      {/*metricas*/}
      <section className="grid metrics">
        <MetricCard label="AUC (test)" value={summary.auc} hint={roc.modelName} />
        <MetricCard label="Recall@test-thr (test)" value={baf?.test_recall_at_val_thr} format="pct1" />
        <MetricCard label="FPR@test-thr (test)" value={baf?.test_fpr_at_val_thr} format="pct1" />
        <MetricCard label="AP (test)" value={baf?.test_ap} />
      </section>

      {/* ROC y Confusion */}
      <section className="grid" style={{gridTemplateColumns: "2fr 1fr", gap: 16, marginBottom: 24}}>
        <div className="card">
          <div className="card-title">ROC Curve</div>
          <p className="subtle" style={{margin:"6px 0 12px"}}>AUC = {summary.auc != null ? Number(roc.auc ?? summary.auc).toFixed(3) : '—'}</p>
          <div style={{height: 260}}>
            {roc.points && roc.points.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={roc.points} margin={{top:8,right:16,bottom:8,left:8}}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.08)"/>
                  <XAxis type="number" dataKey="x" domain={[0,1]} ticks={[0,0.25,0.5,0.75,1]} tick={{fill:"#475569"}} label={{value:"FPR", position:"insideBottomRight", offset:-2, fill:"#475569"}}/>
                  <YAxis type="number" dataKey="y" domain={[0,1]} ticks={[0,0.25,0.5,0.75,1]} tick={{fill:"#475569"}} label={{value:"TPR", angle:-90, position:"insideLeft", fill:"#475569"}}/>
                  <Tooltip formatter={(v)=> (typeof v==="number"? v.toFixed(3): v)} contentStyle={{background:"#ffffff", border:"1px solid #e7eaf2", color:"#0b0b0c"}}/>
                  <Legend />
                  <ReferenceLine segment={[{x:0,y:0},{x:1,y:1}]} stroke="#94a3b8" strokeDasharray="5 5"/>
                  <Line type="monotone" dataKey="y" name={roc.modelName} dot={false} stroke="#0ea5e9" strokeWidth={2}/>
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="subtle">Este JSON no trae puntos de ROC; se muestra solo el AUC.</div>
            )}
          </div>
        </div>

        <div className="card">

         

        </div>
      </section>

      {/*fairness y otras metricas*/}
      <section className="grid" style={{gridTemplateColumns: "1fr 1fr", gap: 16}}>
        <div className="card">
          <div className="card-title">Punto operativo (validación) & métricas</div>
          <div className="grid metrics" style={{gridTemplateColumns: "repeat(auto-fit, minmax(160px,1fr))"}}>
            <MetricCard label="Umbral @ FPR 5% (val)" value={baf?.thr} />
            <MetricCard label="FPR (val)" value={baf?.val_fpr} format="pct1" />
            <MetricCard label="Recall (val)" value={baf?.val_recall} format="pct1" />
            <MetricCard label="AUC (val)" value={baf?.val_auc} />
            <MetricCard label="AP (val)" value={baf?.val_ap} />
            <MetricCard label="Recall@test FPR=5% (curve)" value={baf?.test_recall_at_exact5_curve} format="pct1" />
          </div>
        </div>

        <div className="card">
          <div className="card-title">Fairness (FPR por edad)</div>
          <div style={{height: 220}}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={fairnessData} margin={{top:8,right:16,bottom:8,left:8}}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.08)"/>
                <XAxis dataKey="group" tick={{fill:"#475569"}}/>
                <YAxis domain={[0, (dataMax) => Math.max(0.05, dataMax)]} tickFormatter={(v)=> (v*100).toFixed(0)+"%"} tick={{fill:"#475569"}}/>
                <Tooltip formatter={(v)=> (typeof v==="number"? (v*100).toFixed(2)+"%" : v)} contentStyle={{background:"#ffffff", border:"1px solid #e7eaf2", color:"#0b0b0c"}}/>
                <Bar dataKey="fpr" name="FPR" fill="#0ea5e9"/>
                <ReferenceLine y={0.05} stroke="#94a3b8" strokeDasharray="5 5" label={{ value: "5%", position: "insideTopRight" }} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          {typeof baf?.fpr_ratio_age === 'number' && (
            <div className="subtle" style={{marginTop:8, fontSize:12}}>
              FPR ratio (min/max): <span className="tabnums" style={{color:'#0b0b0c'}}>{baf.fpr_ratio_age.toFixed(3)}</span>
            </div>
          )}
        </div>
      </section>

      <footer className="subtle" style={{marginTop: 24, fontSize: 12}}>
        Equipo Minions
      </footer>
    </div>
  );
}

// contenedor json
export function Page() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    fetch("/metrics_baf_base.json")
      .then((res) => { if(!res.ok) throw new Error(`HTTP ${res.status}`); return res.json(); })
      .then(setData)
      .catch(() => {
        //resultados antiguos (primeras versiones)
        fetch("/resultados.json")
          .then((res) => { if(!res.ok) throw new Error(`HTTP ${res.status}`); return res.json(); })
          .then(setData)
          .catch((e) => setErr(e.message));
      });
  }, []);

  if (err) return <pre style={{ padding: 16, color: "#b91c1c" }}>Error cargando datos: {err}</pre>;
  if (!data) return <p style={{ padding: 16 }}>Cargando...</p>;
  return <FraudBAFDashboard data={data} />;
}
