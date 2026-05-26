const input = document.getElementById("input");
const output = document.getElementById("output");
const meta = document.getElementById("meta");
const modeSelect = document.getElementById("mode");

const API_URL = (window.HINDI_JINNIE_API || "http://localhost:8000").replace(/\/$/, "");
const DEBOUNCE_MS = 450;

let timer = null;
let requestId = 0;

function setOutput(text, { loading = false, error = false } = {}) {
  output.classList.toggle("loading", loading);
  output.classList.toggle("error", error);
  output.textContent = text;
}

async function translate(text) {
  const id = ++requestId;
  setOutput("....translating....", { loading: true });
  meta.hidden = true;

  try {
    const res = await fetch(`${API_URL}/translate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, mode: modeSelect.value }),
    });

    if (id !== requestId) return;

    if (!res.ok) {
      const err = await res.text();
      throw new Error(err || `HTTP ${res.status}`);
    }

    const data = await res.json();
    setOutput(data.translation || "—");
    if (data.latency_ms != null) {
      const label = data.mode === "accurate" ? "Accurate" : "Fast";
      const seconds = (data.latency_ms / 1000).toFixed(data.latency_ms < 1000 ? 2 : 1);
      meta.textContent = `${label} · Translated in ${seconds} seconds`;
      meta.hidden = false;
    }
  } catch (e) {
    if (id !== requestId) return;
    setOutput("Could not reach Hindi Jinnie. Is the API running?", { error: true });
    console.error(e);
  }
}

input.addEventListener("input", () => {
  clearTimeout(timer);
  const text = input.value.trim();
  if (!text) {
    requestId++;
    setOutput("....translating....", { loading: true });
    meta.hidden = true;
    return;
  }
  timer = setTimeout(() => translate(text), DEBOUNCE_MS);
});

modeSelect.addEventListener("change", () => {
  const text = input.value.trim();
  if (text) translate(text);
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    clearTimeout(timer);
    const text = input.value.trim();
    if (text) translate(text);
  }
});
