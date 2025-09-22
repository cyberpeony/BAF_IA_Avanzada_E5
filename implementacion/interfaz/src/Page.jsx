import React, { useEffect, useState } from "react";
import FraudBaselineDashboard from "./FraudBaselineDashboard";

function normalizeROC(rocRaw) {
  if (!rocRaw) return [];
 
  if (Array.isArray(rocRaw) && Array.isArray(rocRaw[0])) {
    return rocRaw.map(([fpr, tpr]) => ({ fpr: Number(fpr) || 0, tpr: Number(tpr) || 0 }));
  }

  if (Array.isArray(rocRaw) && typeof rocRaw[0] === "object") {
    return rocRaw.map(p => ({ fpr: Number(p.fpr) || 0, tpr: Number(p.tpr) || 0 }));
  }

  if (rocRaw.points) return normalizeROC(rocRaw.points);
  return [];
}

function normalizeConfusion(c) {
  if (!c) return { labels: ["Pred 0", "Pred 1"], rows: [{ name: "Actual 0", values: [0, 0] }, { name: "Actual 1", values: [0, 0] }] };

  // Formato con labels
  if (c.labels && c.rows) return c;

  //formato de la matriz
  if (typeof c.tn === "number") {
    return {
      labels: ["Pred 0", "Pred 1"],
      rows: [
        { name: "Actual 0", values: [c.tn, c.fp] },
        { name: "Actual 1", values: [c.fn, c.tp] },
      ],
    };
  }
  return { labels: ["Pred 0", "Pred 1"], rows: [{ name: "Actual 0", values: [0, 0] }, { name: "Actual 1", values: [0, 0] }] };
}

export default function Page() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    fetch("/resultados.json")
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then(setData)
      .catch(e => setErr(e.message));
  }, []);

  if (err) return <pre style={{ padding: 16, color: "#b91c1c" }}>Error cargando datos: {err}</pre>;
  if (!data) return <p style={{ padding: 16 }}>Cargando...</p>;

  const rocPoints = normalizeROC(data.roc);
  const confusion = normalizeConfusion(data.confusion);

  return (
    <FraudBaselineDashboard
      summary={{
        auc: data.auc,
        accuracy: data.accuracy,
        precision: data.precision,
        recall: data.recall,
        threshold: 0.5
      }}
      roc={{
        points: rocPoints,            
        auc: data.auc,
        modelName: "LogReg (baseline)"
      }}
      confusion={confusion}
    />
  );
}