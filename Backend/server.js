import "dotenv/config";
import crypto from "node:crypto";
import cors from "cors";
import express from "express";
import rateLimit from "express-rate-limit";

const PORT = Number(process.env.PORT ?? 4000);
const RECOMMENDER_URL = (process.env.RECOMMENDER_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
const ADMIN_TOKEN = process.env.ADMIN_TOKEN ?? "";
const ORIGINS = (process.env.CORS_ORIGINS ?? "http://localhost:5173").split(",").map((s) => s.trim());

const app = express();
app.disable("x-powered-by");
app.use(cors({ origin: ORIGINS, exposedHeaders: ["X-Request-ID"] }));
app.use(express.json({ limit: "100kb" }));
app.use("/api/", rateLimit({ windowMs: 60_000, limit: 120, standardHeaders: true, legacyHeaders: false }));

async function forward(res, path, { method = "GET", body, admin = false } = {}) {
  try {
    const headers = { "Content-Type": "application/json" };
    if (admin && ADMIN_TOKEN) headers["X-Admin-Token"] = ADMIN_TOKEN;
    const upstream = await fetch(`${RECOMMENDER_URL}${path}`, {
      method,
      headers,
      body: body ? JSON.stringify(body) : undefined,
      signal: AbortSignal.timeout(15_000),
    });
    const requestId = upstream.headers.get("x-request-id");
    if (requestId) res.set("X-Request-ID", requestId);
    res
      .status(upstream.status)
      .type(upstream.headers.get("content-type") ?? "application/json")
      .send(await upstream.text());
  } catch (error) {
    res.status(502).json({ error: "Recommender service unavailable", detail: String(error?.message ?? error) });
  }
}

function requireAdmin(req, res, next) {
  if (!ADMIN_TOKEN) return res.status(503).json({ error: "ADMIN_TOKEN is not configured on the server" });
  const provided = Buffer.from(String(req.get("x-admin-token") ?? ""));
  const expected = Buffer.from(ADMIN_TOKEN);
  if (provided.length !== expected.length || !crypto.timingSafeEqual(provided, expected)) {
    return res.status(401).json({ error: "Invalid admin token" });
  }
  next();
}

// ---- public ----
app.get("/api/health", (_req, res) => forward(res, "/health"));

app.post("/api/recommendations", (req, res) => {
  const limit = Math.min(Math.max(parseInt(req.query.limit, 10) || 10, 1), 50);
  forward(res, `/recommendations?limit=${limit}`, { method: "POST", body: req.body });
});

app.post("/api/feedback", (req, res) => forward(res, "/feedback", { method: "POST", body: req.body }));

// ---- admin ----
app.get("/api/admin/stats", requireAdmin, (_req, res) => forward(res, "/admin/stats", { admin: true }));

app.get("/api/admin/products", requireAdmin, (req, res) => {
  const limit = Math.min(Math.max(parseInt(req.query.limit, 10) || 100, 1), 1000);
  forward(res, `/admin/products?limit=${limit}`, { admin: true });
});

app.post("/api/admin/reload-model", requireAdmin, (_req, res) =>
  forward(res, "/admin/reload-model", { method: "POST", admin: true }),
);

app.use((_req, res) => res.status(404).json({ error: "Not found" }));

app.listen(PORT, () => console.log(`Gateway listening on http://localhost:${PORT} -> ${RECOMMENDER_URL}`));