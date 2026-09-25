"use strict";
/* Fine-Tuning Platform dashboard logic. Vanilla JS, no build step.
   Talks to the FastAPI backend under /api/v1 and streams training logs via SSE. */

// ---- tiny helpers ---------------------------------------------------------
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (_) {}
    throw new Error(msg);
  }
  return res.status === 204 ? null : res.json();
}
const apiGet = (p) => api(p);
const apiPost = (p, body) => api(p, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });

function toast(msg, isErr = false) {
  const t = el("div", "toast" + (isErr ? " err" : ""), msg);
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 4200);
}
const badge = (status) => `<span class="badge ${status}">${status}</span>`;
const fmt = (n, d = 4) => (n == null ? "–" : Number(n).toFixed(d));

// ---- navigation -----------------------------------------------------------
function showView(name) {
  $$(".view").forEach(v => v.classList.toggle("active", v.id === "view-" + name));
  $$("nav button").forEach(b => b.classList.toggle("active", b.dataset.view === name));
  if (name === "datasets") loadDatasets();
  if (name === "jobs") loadJobs();
  if (name === "models") loadModels();
  if (name === "playground") loadPlaygroundModels();
}

// ==========================================================================
//  DATASETS
// ==========================================================================
async function uploadDataset() {
  const fileInput = $("#ds-file");
  if (!fileInput.files.length) return toast("Pick a JSONL file first.", true);
  const fd = new FormData();
  fd.append("file", fileInput.files[0]);
  if ($("#ds-name").value) fd.append("name", $("#ds-name").value);
  $("#ds-upload-btn").disabled = true;
  try {
    const ds = await api("/api/v1/datasets", { method: "POST", body: fd });
    toast(`Uploaded ${ds.name}: ${ds.status}`);
    renderReport(ds);
    loadDatasets();
  } catch (e) { toast("Upload failed: " + e.message, true); }
  finally { $("#ds-upload-btn").disabled = false; }
}

function renderReport(ds) {
  const box = $("#ds-report");
  box.classList.remove("hidden");
  const r = ds.quality_report;
  if (!r) { box.innerHTML = "<p class='muted'>No report.</p>"; return; }
  const okRec = r.recommendation.startsWith("Dataset looks good");
  box.innerHTML = `
    <h2>Quality report — ${ds.name} ${badge(ds.status)}</h2>
    <div class="stat-row">
      ${stat("examples", r.total_examples)}
      ${stat("unique", r.unique_examples)}
      ${stat("exact dupes", r.exact_duplicates)}
      ${stat("near dupes", r.near_duplicates)}
      ${stat("low quality", (r.low_quality_examples || []).length)}
      ${stat("train / val", `${r.train_size} / ${r.val_size}`)}
    </div>
    <h3>Token distribution (${r.tokenizer})</h3>
    <table><tr><th></th><th>mean</th><th>median</th><th>p95</th><th>max</th></tr>
      ${tokRow("prompt", r.prompt_token_stats)}${tokRow("completion", r.completion_token_stats)}
    </table>
    <h3>Cost estimate (${r.cost.total_tokens.toLocaleString()} tokens)</h3>
    <div class="stat-row">
      ${stat("OpenAI API", "$" + r.cost.openai_usd)}
      ${stat("local GPU", "$" + r.cost.local_gpu_usd)}
      ${stat("~train time", r.cost.training_time_minutes + " min")}
      ${stat("quality via", r.quality_method)}
    </div>
    <h3>Recommendation</h3>
    <div class="rec ${okRec ? "ok" : ""}">${r.recommendation}</div>
    ${(r.format_errors || []).length ? `<h3>Format errors</h3><pre class="logs">${r.format_errors.join("\n")}</pre>` : ""}
  `;
}
const stat = (k, v) => `<div class="stat"><div class="v">${v}</div><div class="k">${k}</div></div>`;
const tokRow = (name, s) => `<tr><td>${name}</td><td class="mono">${s.mean}</td><td class="mono">${s.median}</td><td class="mono">${s.p95}</td><td class="mono">${s.max}</td></tr>`;

async function loadDatasets() {
  const rows = await apiGet("/api/v1/datasets");
  const tb = $("#ds-table tbody");
  tb.innerHTML = "";
  fillDatasetSelect(rows);
  rows.forEach(d => {
    const tr = el("tr");
    tr.innerHTML = `<td class="mono">${d.id}</td><td>${d.name}</td><td class="mono">${d.num_examples}</td><td>${badge(d.status)}</td>
      <td><button class="btn secondary small" data-ds="${d.id}">report</button></td>`;
    $("button", tr).onclick = async () => renderReport(await apiGet("/api/v1/datasets/" + d.id));
    tb.appendChild(tr);
  });
}
function fillDatasetSelect(rows) {
  const sel = $("#job-dataset");
  const cur = sel.value;
  sel.innerHTML = "";
  rows.filter(d => d.status === "ready").forEach(d => {
    const o = el("option"); o.value = d.id; o.textContent = `${d.name} (${d.num_examples} ex)`; sel.appendChild(o);
  });
  if (cur) sel.value = cur;
}

// ==========================================================================
//  JOBS
// ==========================================================================
async function createJob() {
  const dataset_id = $("#job-dataset").value;
  if (!dataset_id) return toast("Upload & select a ready dataset first.", true);
  let hyperparameters;
  try { hyperparameters = JSON.parse($("#job-hp").value || "{}"); }
  catch (e) { return toast("Hyperparameters must be valid JSON.", true); }
  try {
    const job = await apiPost("/api/v1/fine-tune", {
      dataset_id, base_model: $("#job-model").value || "demo", hyperparameters,
    });
    toast("Job " + job.id + " queued.");
    showView("jobs");
    openJob(job.id);
  } catch (e) { toast("Could not start job: " + e.message, true); }
}

async function loadJobs() {
  const rows = await apiGet("/api/v1/jobs");
  const tb = $("#jobs-table tbody");
  tb.innerHTML = "";
  rows.forEach(j => {
    const tr = el("tr");
    const pct = Math.round((j.progress || 0) * 100);
    tr.innerHTML = `<td class="mono">${j.id}</td><td>${j.base_model}</td><td>${badge(j.status)}</td>
      <td style="width:120px"><div class="progress"><div style="width:${pct}%"></div></div></td>
      <td class="mono">${pct}%</td>`;
    tr.onclick = () => openJob(j.id);
    tr.style.cursor = "pointer";
    tb.appendChild(tr);
  });
}

let activeStream = null; // EventSource
async function openJob(jobId) {
  if (activeStream) { activeStream.close(); activeStream = null; }
  const job = await apiGet("/api/v1/jobs/" + jobId);
  const panel = $("#job-detail");
  panel.classList.remove("hidden");
  panel.dataset.jobId = jobId;
  renderJobHeader(job);

  const chart = $("#loss-chart");
  const history = (job.metrics && job.metrics.history) || [];
  drawLoss(chart, history);

  const logsPre = $("#job-logs");
  logsPre.innerHTML = "";

  const terminal = ["completed", "completed_below_threshold", "failed"].includes(job.status);
  if (terminal) {
    (job.logs || []).forEach(l => appendLog(logsPre, l));
    return; // no live stream needed
  }
  // Live: stream logs + metrics via SSE.
  const es = new EventSource("/api/v1/jobs/" + jobId + "/logs");
  activeStream = es;
  const liveHistory = [...history];
  es.onmessage = (ev) => {
    const d = JSON.parse(ev.data);
    if (d.type === "log") appendLog(logsPre, d);
    else if (d.type === "metric") {
      liveHistory.push({ step: d.step, loss: d.loss, eval_loss: d.eval_loss });
      drawLoss(chart, liveHistory);
      updateProgress(d.progress, d.eta_seconds);
    } else if (d.type === "status") {
      if (d.final) { es.close(); activeStream = null; refreshJob(jobId); loadJobs(); loadModels(); }
    }
  };
  es.onerror = () => { es.close(); activeStream = null; };
}

async function refreshJob(jobId) {
  const job = await apiGet("/api/v1/jobs/" + jobId);
  if ($("#job-detail").dataset.jobId === jobId) renderJobHeader(job);
}

function renderJobHeader(job) {
  const final = (job.metrics && job.metrics.final) || {};
  $("#job-head").innerHTML = `
    <h2>Job ${job.id} ${badge(job.status)}
      <span class="small muted">backend: ${job.trainer_backend}</span></h2>
    <div class="stat-row">
      ${stat("base model", job.base_model)}
      ${stat("train loss", fmt(final.train_loss))}
      ${stat("eval loss", fmt(final.eval_loss))}
      ${stat("perplexity", fmt(final.perplexity, 2))}
      ${stat("model", job.model_id || "–")}
    </div>
    <div class="progress" style="margin-top:10px"><div id="job-prog" style="width:${Math.round((job.progress||0)*100)}%"></div></div>
    <div class="small muted" id="job-eta"></div>
    ${job.error ? `<div class="rec" style="border-left-color:var(--danger)">${job.error}</div>` : ""}`;
}
function updateProgress(progress, eta) {
  const bar = $("#job-prog"); if (bar) bar.style.width = Math.round((progress || 0) * 100) + "%";
  const e = $("#job-eta"); if (e && eta != null) e.textContent = `ETA ~${eta}s`;
}
function appendLog(pre, l) {
  const line = el("div", "log-line " + (l.level || "info"), `${(l.ts || "").slice(11, 19)}  ${l.message}`);
  pre.appendChild(line);
  pre.scrollTop = pre.scrollHeight;
}

// ---- hand-rolled loss chart (canvas) -------------------------------------
function drawLoss(canvas, history) {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.clientHeight;
  canvas.width = W * dpr; canvas.height = H * dpr; ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);
  const pad = { l: 44, r: 12, t: 14, b: 24 };
  const pts = history.filter(p => p.loss != null);
  if (!pts.length) { ctx.fillStyle = "#8b949e"; ctx.fillText("waiting for training data…", pad.l, H / 2); return; }

  const steps = pts.map(p => p.step);
  const losses = pts.map(p => p.loss);
  const evals = pts.filter(p => p.eval_loss != null);
  const allY = losses.concat(evals.map(p => p.eval_loss));
  const xMin = Math.min(...steps), xMax = Math.max(...steps) || 1;
  const yMin = Math.min(...allY) * 0.95, yMax = Math.max(...allY) * 1.05;
  const X = (s) => pad.l + (s - xMin) / (xMax - xMin || 1) * (W - pad.l - pad.r);
  const Y = (v) => pad.t + (1 - (v - yMin) / (yMax - yMin || 1)) * (H - pad.t - pad.b);

  // axes + gridlines
  ctx.strokeStyle = "#2d3644"; ctx.fillStyle = "#8b949e"; ctx.font = "10px monospace"; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const v = yMin + (yMax - yMin) * i / 4, y = Y(v);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    ctx.fillText(v.toFixed(2), 4, y + 3);
  }
  const drawSeries = (data, key, color) => {
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
    data.forEach((p, i) => { const x = X(p.step), y = Y(p[key]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke();
  };
  drawSeries(pts, "loss", "#58a6ff");
  if (evals.length) drawSeries(evals, "eval_loss", "#d29922");
  // legend
  ctx.fillStyle = "#58a6ff"; ctx.fillText("train loss", W - 150, 14);
  ctx.fillStyle = "#d29922"; ctx.fillText("eval loss", W - 70, 14);
  ctx.fillStyle = "#8b949e"; ctx.fillText("step", W / 2, H - 6);
}

// ==========================================================================
//  MODELS / LEADERBOARD
// ==========================================================================
async function loadModels() {
  const rows = await apiGet("/api/v1/models");
  const tb = $("#models-table tbody");
  tb.innerHTML = "";
  rows.forEach((m, i) => {
    const cmp = m.comparison_vs_base || {};
    const tr = el("tr");
    tr.innerHTML = `<td class="mono">${i + 1}</td><td>${m.name} v${m.version}</td>
      <td>${badge(m.status)}</td>
      <td class="mono">${fmt((m.metrics||{}).eval_loss)}</td>
      <td class="mono">${cmp.relative_improvement != null ? (cmp.relative_improvement*100).toFixed(1)+"%" : "–"}</td>
      <td><button class="btn small" data-deploy="${m.id}">deploy</button></td>`;
    $("button", tr).onclick = () => deployModel(m.id);
    tb.appendChild(tr);
  });
}

async function deployModel(modelId) {
  try {
    const dep = await apiPost(`/api/v1/models/${modelId}/deploy`, { canary_percent: 100 });
    toast(`Deployed: ${dep.status} @ ${dep.endpoint_url || dep.detail.manifest_path}`);
    loadPlaygroundModels();
    showView("playground");
  } catch (e) { toast("Deploy failed: " + e.message, true); }
}

// ==========================================================================
//  PLAYGROUND (base vs fine-tuned)
// ==========================================================================
async function loadPlaygroundModels() {
  const models = await apiGet("/api/v1/models");
  const sel = $("#pg-model"); sel.innerHTML = "";
  for (const m of models) {
    const full = await apiGet("/api/v1/models/" + m.id);
    (full.deployments || []).filter(d => d.backend === "local" && d.status === "live").forEach(d => {
      const o = el("option"); o.value = d.id; o.textContent = `${m.name} v${m.version} · ${d.id}`; sel.appendChild(o);
    });
  }
  if (!sel.options.length) { const o = el("option"); o.textContent = "— deploy a model (local backend) first —"; o.value = ""; sel.appendChild(o); }
}

async function runCompare() {
  const depId = $("#pg-model").value;
  if (!depId) return toast("Deploy a model with the local backend first.", true);
  const prompt = $("#pg-prompt").value.trim();
  if (!prompt) return toast("Enter a prompt.", true);
  try {
    const r = await apiPost(`/api/v1/endpoints/${depId}/compare`, { prompt });
    $("#pg-result").classList.remove("hidden");
    $("#pg-base").innerHTML = `<h4>Base · ${r.base.model}</h4><div class="small">${r.base.completion}</div>`;
    $("#pg-ft").innerHTML = `<h4>Fine-tuned · ${r.finetuned.model}</h4><div class="small">${r.finetuned.completion}</div>`;
    $("#pg-note").textContent = r.note;
  } catch (e) { toast("Compare failed: " + e.message, true); }
}

// ==========================================================================
//  bootstrap
// ==========================================================================
async function init() {
  $$("nav button").forEach(b => b.onclick = () => showView(b.dataset.view));
  $("#ds-upload-btn").onclick = uploadDataset;
  $("#job-create-btn").onclick = createJob;
  $("#pg-run-btn").onclick = runCompare;
  $("#jobs-refresh").onclick = loadJobs;
  try {
    const cfg = await apiGet("/api/v1/config");
    $("#cfg").textContent = `trainer=${cfg.trainer_backend} · deploy=${cfg.deploy_backend} · v${cfg.version}`;
  } catch (_) {}
  showView("datasets");
  // Light polling so job/leaderboard lists stay fresh.
  setInterval(() => { const active = $("nav button.active").dataset.view; if (active === "jobs") loadJobs(); }, 3000);
}
document.addEventListener("DOMContentLoaded", init);
