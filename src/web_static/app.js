"use strict";

const $ = (id) => document.getElementById(id);
const SIGNAL_COLORS = { BUY: "#2ee6a6", SELL: "#ff5c7a", WATCH: "#ffb648", HOLD: "#7c8aa5" };

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}
const pct = (x) => (x * 100).toFixed(2) + "%";
const signedPct = (x) => (x >= 0 ? "+" : "") + (x * 100).toFixed(2) + "%";
const money = (x) => "$" + Number(x).toLocaleString(undefined, { maximumFractionDigits: 2 });
const num = (x, d = 2) => Number(x).toFixed(d);

function payload() {
  return {
    tickers: $("tickers").value,
    period: $("period").value,
    portfolio: parseFloat($("portfolio").value) || 10000,
    demo: $("demo").checked,
  };
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error("Request failed (" + res.status + ")");
  return res.json();
}

function setStatus(msg, isError = false) {
  const el = $("status");
  el.textContent = msg || "";
  el.classList.toggle("error", isError);
}

/* ----------------------------------------------------------------- render */
function ring(conf, signal) {
  const r = 32, c = 2 * Math.PI * r, off = c * (1 - conf / 100);
  const color = SIGNAL_COLORS[signal] || "#7c8aa5";
  return `<svg class="ring" viewBox="0 0 78 78">
    <circle cx="39" cy="39" r="${r}" fill="none" stroke="#0c111c" stroke-width="7"/>
    <circle cx="39" cy="39" r="${r}" fill="none" stroke="${color}" stroke-width="7"
      stroke-linecap="round" stroke-dasharray="${c}" stroke-dashoffset="${off}"
      transform="rotate(-90 39 39)"/>
    <text x="39" y="41" text-anchor="middle">${conf}</text>
    <text x="39" y="53" text-anchor="middle" class="label">CONF</text>
  </svg>`;
}

function factorBars(factors) {
  const order = ["trend", "momentum", "participation", "quality"];
  return `<div class="factors">${order
    .map((name) => {
      const v = Math.max(-1, Math.min(1, factors[name] ?? 0));
      const w = Math.abs(v) * 50;
      const style = v >= 0 ? `left:50%;width:${w}%` : `left:${50 - w}%;width:${w}%`;
      const cls = v >= 0 ? "pos" : "neg";
      return `<div class="factor">
        <span class="fname">${name}</span>
        <span class="bar"><span class="mid"></span><span class="fill ${cls}" style="${style}"></span></span>
        <span class="fval">${v >= 0 ? "+" : ""}${v.toFixed(2)}</span>
      </div>`;
    })
    .join("")}</div>`;
}

function lists(reasons, risks) {
  const li = (items, cls) =>
    items.map((t) => `<li class="${cls}">${escapeHtml(t)}</li>`).join("");
  return `<div class="lists">
    <ul>${li(reasons || [], "reason")}</ul>
    <ul>${li(risks || [], "risk")}</ul>
  </div>`;
}

function levels(r) {
  const chips = [];
  if (r.entry_zone)
    chips.push(`<span class="chip"><b>Entry</b>${num(r.entry_zone[0])} – ${num(r.entry_zone[1])}</span>`);
  if (r.stop_loss != null) chips.push(`<span class="chip"><b>Stop</b>${num(r.stop_loss)}</span>`);
  return chips.length ? `<div class="levels">${chips.join("")}</div>` : "";
}

function sizing(pos) {
  if (!pos) return "";
  const warns = (pos.warnings || []).map((w) => `<span class="warn">! ${escapeHtml(w)}</span>`).join("");
  return `<div class="sizing">Suggested: <b>${pos.shares}</b> shares
    (~${pct(pos.allocation_pct)} of portfolio, risking ${pct(pos.risk_pct)})${warns}</div>`;
}

function renderCard(r) {
  if (!r.ok) {
    return `<div class="card"><div class="card-head"><div class="ticker">${escapeHtml(r.ticker)}</div>
      <span class="badge HOLD">N/A</span></div>
      <p class="narrative" style="margin-top:14px">Could not analyse: ${escapeHtml(r.error)}</p></div>`;
  }
  return `<div class="card" data-ticker="${escapeHtml(r.ticker)}">
    <div class="card-head">
      <div><div class="ticker">${escapeHtml(r.ticker)}</div><div class="price">Close ${num(r.latest_close)}</div></div>
      <span class="badge ${r.signal}">${r.signal}</span>
    </div>
    <div class="gauge-row">${ring(r.confidence, r.signal)}<p class="narrative">${escapeHtml(r.narrative)}</p></div>
    ${factorBars(r.factors)}
    ${lists(r.reasons, r.risks)}
    ${levels(r)}
    ${sizing(r.position)}
    <div class="bt-block">
      <button class="btn ghost small bt-btn">Run backtest</button>
      <div class="bt-result"></div>
    </div>
  </div>`;
}

/* ----------------------------------------------------------------- charts */
function scale(values, W, H, pad) {
  const min = Math.min(...values), max = Math.max(...values), range = max - min || 1;
  return {
    x: (i, n) => pad + (i / (n - 1 || 1)) * (W - 2 * pad),
    y: (v) => H - pad - ((v - min) / range) * (H - 2 * pad),
  };
}
function pathFor(values, s, n) {
  return values.map((v, i) => (i ? "L" : "M") + s.x(i, n).toFixed(1) + "," + s.y(v).toFixed(1)).join(" ");
}

function sparkline(points) {
  if (!points.length) return "<p class='price'>No equity data.</p>";
  const W = 300, H = 90, pad = 6, vals = points.map((p) => p.v), n = vals.length;
  const s = scale(vals, W, H, pad);
  const line = pathFor(vals, s, n);
  const area = `M${s.x(0, n).toFixed(1)},${H - pad} ` +
    points.map((p, i) => "L" + s.x(i, n).toFixed(1) + "," + s.y(p.v).toFixed(1)).join(" ") +
    ` L${s.x(n - 1, n).toFixed(1)},${H - pad} Z`;
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
    <defs><linearGradient id="sparkfill" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#4f8cff" stop-opacity="0.35"/>
      <stop offset="100%" stop-color="#4f8cff" stop-opacity="0"/></linearGradient></defs>
    <path class="spark-area" d="${area}"/><path class="spark-path" d="${line}"/></svg>`;
}

function dualChart(equity, benchmark) {
  const W = 760, H = 220, pad = 10;
  const all = equity.map((p) => p.v).concat(benchmark.map((p) => p.v));
  if (!all.length) return "";
  const s = scale(all, W, H, pad);
  const e = pathFor(equity.map((p) => p.v), s, equity.length);
  const b = pathFor(benchmark.map((p) => p.v), s, benchmark.length);
  return `<svg class="chart" style="height:220px" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
    <path class="bench-path" d="${b}"/><path class="spark-path" d="${e}"/></svg>`;
}

/* --------------------------------------------------------------- actions */
async function analyze() {
  const btn = $("analyze");
  btn.disabled = true;
  setStatus("Analysing…");
  const grid = $("results");
  const body = payload();
  const tickerCount = body.tickers.split(/[\s,]+/).filter(Boolean).length || 5;
  grid.innerHTML = Array.from({ length: tickerCount }, () => '<div class="skeleton"></div>').join("");
  try {
    const data = await postJSON("/api/analyze", body);
    grid.innerHTML = data.results.map(renderCard).join("");
    wireBacktestButtons();
    const ok = data.results.filter((r) => r.ok).length;
    setStatus(`Analysed ${ok}/${data.results.length} ticker(s).` + (body.demo ? " (demo data)" : ""));
  } catch (err) {
    grid.innerHTML = "";
    setStatus(err.message + (body.demo ? "" : " — enable Demo data to preview offline."), true);
  } finally {
    btn.disabled = false;
  }
}

function wireBacktestButtons() {
  document.querySelectorAll(".bt-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const card = btn.closest(".card");
      const ticker = card.getAttribute("data-ticker");
      const out = card.querySelector(".bt-result");
      btn.disabled = true;
      out.innerHTML = "<p class='price'>Running backtest…</p>";
      try {
        const body = Object.assign(payload(), { tickers: [ticker] });
        const d = await postJSON("/api/backtest", body);
        if (!d.ok) { out.innerHTML = `<p class='price'>${escapeHtml(d.error)}</p>`; return; }
        const m = d.metrics;
        out.innerHTML = sparkline(d.equity_curve) + `<div class="bt-metrics">
          <span>Return <b>${signedPct(m.total_return)}</b></span>
          <span>CAGR <b>${signedPct(m.cagr)}</b></span>
          <span>Sharpe <b>${num(m.sharpe)}</b></span>
          <span>MaxDD <b>${pct(m.max_drawdown)}</b></span>
          <span>Win <b>${(m.win_rate * 100).toFixed(0)}%</b></span>
          <span>Trades <b>${m.num_trades}</b></span></div>`;
      } catch (err) {
        out.innerHTML = `<p class='price'>${escapeHtml(err.message)}</p>`;
      } finally {
        btn.disabled = false;
      }
    });
  });
}

function metricCard(k, v, signed = false) {
  const cls = signed ? (v >= 0 ? "pos" : "neg") : "";
  const val = signed ? signedPct(v) : v;
  return `<div class="metric"><div class="k">${k}</div><div class="v ${cls}">${val}</div></div>`;
}

async function runPortfolio() {
  const btn = $("run-portfolio");
  const panel = $("portfolio-panel");
  btn.disabled = true;
  panel.classList.remove("hidden");
  panel.innerHTML = '<div class="skeleton" style="height:260px"></div>';
  const body = payload();
  try {
    const d = await postJSON("/api/portfolio", body);
    if (!d.ok) { panel.innerHTML = `<h2>Portfolio Backtest</h2><p class="sub">${escapeHtml(d.error)}</p>`; return; }
    const m = d.metrics;
    const edge = m.total_return - m.benchmark_return;
    panel.innerHTML = `<h2>Portfolio Backtest <span class="price">${escapeHtml(d.tickers.join(", "))}</span></h2>
      <p class="sub">Risk-managed (2% / trade · 20% / stock) vs equal-weight buy &amp; hold.</p>
      <div class="pf-metrics">
        ${metricCard("Total Return", m.total_return, true)}
        ${metricCard("CAGR", m.cagr, true)}
        ${metricCard("Sharpe", num(m.sharpe))}
        ${metricCard("Sortino", num(m.sortino))}
        ${metricCard("Max Drawdown", pct(m.max_drawdown))}
        ${metricCard("Win Rate", (m.win_rate * 100).toFixed(0) + "%")}
        ${metricCard("Final Value", money(m.final_value))}
        ${metricCard("Edge vs B&H", edge, true)}
      </div>
      ${dualChart(d.equity_curve, d.benchmark_curve)}
      <div class="legend"><span><span class="dot" style="background:#4f8cff"></span>Strategy</span>
        <span><span class="dot" style="background:#8b9bb4"></span>Buy &amp; Hold (${signedPct(m.benchmark_return)})</span></div>`;
  } catch (err) {
    panel.innerHTML = `<h2>Portfolio Backtest</h2><p class="sub">${escapeHtml(err.message)}</p>`;
  } finally {
    btn.disabled = false;
  }
}

$("analyze").addEventListener("click", analyze);
$("run-portfolio").addEventListener("click", runPortfolio);
$("tickers").addEventListener("keydown", (e) => { if (e.key === "Enter") analyze(); });
window.addEventListener("DOMContentLoaded", analyze);
