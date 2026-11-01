import { useCallback, useEffect, useState } from "react";
import {
  getHealth,
  getProducts,
  getStats,
  recommend,
  reloadModel,
  sendFeedback,
  setAdminToken,
} from "./api.js";

const EVENTS = ["black_friday", "christmas", "seasonal_sale", "flash_sale", "generic"];
const pct = (value) => `${(value * 100).toFixed(1)}%`;
const discountOf = (p) =>
  p.original_price && p.original_price > p.current_price
    ? Math.round((1 - p.current_price / p.original_price) * 100)
    : 0;

function Stat({ label, value }) {
  return (
    <div className="stat">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Overview({ token }) {
  const [health, setHealth] = useState(null);
  const [stats, setStats] = useState(null);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    setError("");
    try {
      setHealth((await getHealth()).data);
      setStats((await getStats()).data);
    } catch (e) {
      setError(e.message);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load, token]);

  const reload = async () => {
    setError("");
    try {
      setHealth((await reloadModel()).data);
    } catch (e) {
      setError(e.message);
    }
  };

  return (
    <section>
      {error && <p className="error">{error}</p>}
      <div className="stats">
        <Stat label="Model" value={health?.model_version ?? "—"} />
        <Stat label="Neural loaded" value={health ? String(health.neural_loaded) : "—"} />
        <Stat label="Products" value={stats?.products ?? "—"} />
        <Stat label="Requests" value={stats?.requests ?? "—"} />
        <Stat label="Impressions" value={stats?.impressions ?? "—"} />
        <Stat label="Clicks" value={stats?.clicks ?? "—"} />
        <Stat label="Purchases" value={stats?.purchases ?? "—"} />
        <Stat label="CTR" value={stats ? pct(stats.ctr) : "—"} />
      </div>
      <div className="row">
        <button onClick={load}>Refresh</button>
        <button onClick={reload}>Reload model</button>
      </div>
      {stats?.by_model?.length > 0 && (
        <table>
          <thead>
            <tr><th>Model version</th><th>Impressions</th><th>Clicks</th><th>CTR</th></tr>
          </thead>
          <tbody>
            {stats.by_model.map((row) => (
              <tr key={row.model_version}>
                <td>{row.model_version}</td>
                <td>{row.impressions}</td>
                <td>{row.clicks}</td>
                <td>{pct(row.ctr)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}

function Products({ token }) {
  const [items, setItems] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    getProducts(200)
      .then(({ data }) => {
        setItems(data);
        setError("");
      })
      .catch((e) => setError(e.message));
  }, [token]);

  return (
    <section>
      {error && <p className="error">{error}</p>}
      <div className="scroll">
        <table>
          <thead>
            <tr>
              <th>Title</th><th>Category</th><th>Price</th><th>Discount</th>
              <th>Rating</th><th>Reviews</th><th>Stock</th><th>Source</th>
            </tr>
          </thead>
          <tbody>
            {items.map((p) => (
              <tr key={`${p.source}:${p.external_id}`}>
                <td>{p.title}</td>
                <td>{p.category}</td>
                <td>{p.current_price.toFixed(2)} {p.currency}</td>
                <td>{discountOf(p)}%</td>
                <td>{p.rating ?? "—"}</td>
                <td>{p.review_count}</td>
                <td>{p.availability}</td>
                <td>{p.source}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Playground() {
  const [form, setForm] = useState({ event: "black_friday", category: "electronics", budget: 150, urgency: 0.9 });
  const [result, setResult] = useState({ requestId: null, items: [] });
  const [message, setMessage] = useState("");
  const update = (key) => (e) => setForm({ ...form, [key]: e.target.value });

  const submit = async (e) => {
    e.preventDefault();
    setMessage("");
    const body = { event: form.event, urgency: Number(form.urgency) };
    if (form.category.trim()) body.category = form.category.trim();
    if (Number(form.budget) > 0) body.budget = Number(form.budget);
    try {
      const { data, requestId } = await recommend(body);
      setResult({ requestId, items: data });
    } catch (err) {
      setMessage(err.message);
    }
  };

  const feedback = async (item, action) => {
    try {
      await sendFeedback({
        request_id: result.requestId,
        source: item.product.source,
        external_id: item.product.external_id,
        action,
      });
      setMessage(`Recorded ${action} for "${item.product.title}"`);
    } catch (err) {
      setMessage(err.message);
    }
  };

  return (
    <section className="playground">
      <form onSubmit={submit} className="panel">
        <label>Event
          <select value={form.event} onChange={update("event")}>
            {EVENTS.map((ev) => <option key={ev} value={ev}>{ev}</option>)}
          </select>
        </label>
        <label>Category
          <input value={form.category} onChange={update("category")} placeholder="electronics" />
        </label>
        <label>Budget
          <input type="number" min="0" value={form.budget} onChange={update("budget")} />
        </label>
        <label>Urgency: {Math.round(form.urgency * 100)}%
          <input type="range" min="0" max="1" step="0.05" value={form.urgency} onChange={update("urgency")} />
        </label>
        <button type="submit">Get recommendations</button>
      </form>
      <div className="panel">
        {message && <p className="note">{message}</p>}
        {result.items.length === 0 && <p className="muted">No results yet.</p>}
        {result.items.map((item) => (
          <article key={item.product.external_id} className="card">
            <div>
              <h3>{item.product.title}</h3>
              <p className="muted">
                {item.product.current_price.toFixed(2)} {item.product.currency} · {item.model_version}
              </p>
              <p>{item.reasons.join(" · ")}</p>
            </div>
            <div className="actions">
              <strong>{Math.round(item.score * 100)}</strong>
              <button type="button" onClick={() => feedback(item, "click")}>Simulate click</button>
              <button type="button" onClick={() => feedback(item, "purchase")}>Simulate purchase</button>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

export default function App() {
  const [tab, setTab] = useState("overview");
  const [draft, setDraft] = useState("");
  const [token, setToken] = useState("");

  const applyToken = () => {
    setAdminToken(draft);
    setToken(draft);
  };

  return (
    <main>
      <header>
        <h1>Recommender Admin</h1>
        <div className="row">
          <input
            type="password"
            placeholder="Admin token"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
          />
          <button onClick={applyToken}>Use token</button>
        </div>
      </header>
      <nav>
        {["overview", "products", "playground"].map((name) => (
          <button key={name} className={tab === name ? "active" : ""} onClick={() => setTab(name)}>
            {name}
          </button>
        ))}
      </nav>
      {tab === "overview" && <Overview token={token} />}
      {tab === "products" && <Products token={token} />}
      {tab === "playground" && <Playground />}
    </main>
  );
}