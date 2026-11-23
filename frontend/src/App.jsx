import { useEffect, useRef, useState } from "react";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";
const MAX_MB = 8;

const LABEL_ID = {
  cardboard: "Kardus",
  glass: "Kaca",
  metal: "Logam",
  paper: "Kertas",
  plastic: "Plastik",
  trash: "Residu",
};

export default function App() {
  const [models, setModels] = useState([]);
  const [model, setModel] = useState("");
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [drag, setDrag] = useState(false);
  const inputRef = useRef(null);

  useEffect(() => {
    fetch(`${API}/models`)
      .then((r) => r.json())
      .then((d) => {
        setModels(d.models);
        setModel(d.default);
      })
      .catch(() => setError("Tidak dapat terhubung ke API. Pastikan backend berjalan."));
  }, []);

  useEffect(() => () => preview && URL.revokeObjectURL(preview), [preview]);

  function pick(f) {
    setError("");
    setResult(null);
    if (!f) return;
    if (!["image/jpeg", "image/png", "image/webp"].includes(f.type)) {
      setError("Format harus JPG, PNG, atau WEBP.");
      return;
    }
    if (f.size > MAX_MB * 1024 * 1024) {
      setError(`Ukuran file maksimal ${MAX_MB} MB.`);
      return;
    }
    setFile(f);
    setPreview(URL.createObjectURL(f));
  }

  async function classify() {
    if (!file) return;
    setLoading(true);
    setError("");
    const t0 = performance.now();
    try {
      const fd = new FormData();
      fd.append("file", file);
      const r = await fetch(`${API}/predict?model=${model}`, { method: "POST", body: fd });
      const body = await r.json();
      if (!r.ok) throw new Error(body.detail || "Terjadi kesalahan.");
      body.roundtrip_ms = Math.round(performance.now() - t0);
      setResult(body);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const rows = result ? Object.entries(result.probabilities).slice(0, 3) : [];

  return (
    <main className="wrap">
      <header>
        <h1>Smartwaste-AI</h1>
        <p>Unggah foto sampah tunggal untuk mengetahui jenisnya.</p>
      </header>

      <section className="card">
        <div
          className={`drop ${drag ? "drag" : ""}`}
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files[0]); }}
        >
          {preview ? <img src={preview} alt="pratinjau" /> : <span>Klik atau seret gambar ke sini</span>}
          <input ref={inputRef} type="file" accept="image/jpeg,image/png,image/webp" hidden
                 onChange={(e) => pick(e.target.files[0])} />
        </div>

        <div className="row">
          <label>
            Model
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              {models.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
            </select>
          </label>
          <button onClick={classify} disabled={!file || loading}>
            {loading ? "Memproses..." : "Klasifikasikan"}
          </button>
        </div>

        {error && <p className="error">{error}</p>}
      </section>

      {result && (
        <section className="card">
          <h2>
            {LABEL_ID[result.prediction] || result.prediction}{" "}
            <small>({result.prediction})</small>
          </h2>
          <p className="muted">
            Keyakinan {(result.confidence * 100).toFixed(1)}% · inferensi {result.inference_ms} ms ·
            total {result.roundtrip_ms} ms
          </p>
          {rows.map(([name, p]) => (
            <div className="bar" key={name}>
              <span>{LABEL_ID[name] || name}</span>
              <div><i style={{ width: `${p * 100}%` }} /></div>
              <b>{(p * 100).toFixed(1)}%</b>
            </div>
          ))}
        </section>
      )}
    </main>
  );
}
