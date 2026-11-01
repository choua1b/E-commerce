const BASE = "/api";
let adminToken = "";

export const setAdminToken = (token) => {
  adminToken = token;
};

async function request(path, { method = "GET", body, admin = false } = {}) {
  const headers = { "Content-Type": "application/json" };
  if (admin) headers["X-Admin-Token"] = adminToken;
  const response = await fetch(`${BASE}${path}`, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await response.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = text;
  }
  if (!response.ok) {
    const detail = data?.detail ?? data?.error ?? response.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return { data, requestId: response.headers.get("X-Request-ID") };
}

export const getHealth = () => request("/health");
export const getStats = () => request("/admin/stats", { admin: true });
export const getProducts = (limit = 100) => request(`/admin/products?limit=${limit}`, { admin: true });
export const reloadModel = () => request("/admin/reload-model", { method: "POST", admin: true });
export const recommend = (context, limit = 10) =>
  request(`/recommendations?limit=${limit}`, { method: "POST", body: context });
export const sendFeedback = (payload) => request("/feedback", { method: "POST", body: payload });