// Interfaz de prueba de las tres redes neuronales.
const $ = (id) => document.getElementById(id);
const fmt = (v, d = 2) => Number(v).toLocaleString("es-MX", { minimumFractionDigits: d, maximumFractionDigits: d });
const color = (nombre) => getComputedStyle(document.documentElement).getPropertyValue(nombre).trim();
const azar = (n) => Math.floor(Math.random() * n);
let M = {};

// ---------- utilidades de interfaz ----------
function barras(contenedor, etiquetas, probs) {
  const imax = probs.indexOf(Math.max(...probs));
  contenedor.innerHTML = etiquetas.map((e, i) => `
    <div class="fila-barra${i === imax ? " max" : ""}">
      <span>${e}</span>
      <div class="pista"><div class="relleno" style="width:${(100 * probs[i]).toFixed(1)}%"></div></div>
      <span>${fmt(100 * probs[i], 1)} %</span>
    </div>`).join("");
  return imax;
}

function insignia(acierto) {
  return `<span class="insignia ${acierto ? "ok" : "mal"}">${acierto ? "acierto" : "error"}</span>`;
}

function lienzo(canvas) {
  const r = window.devicePixelRatio || 1, w = canvas.clientWidth, h = Number(canvas.getAttribute("height"));
  canvas.width = w * r; canvas.height = h * r; canvas.style.height = h + "px";
  const ctx = canvas.getContext("2d"); ctx.setTransform(r, 0, 0, r, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function ejes(ctx, w, h, xr, yr, xt, yt, m = { i: 48, d: 12, s: 10, b: 34 }) {
  const X = (v) => m.i + (v - xr[0]) / (xr[1] - xr[0]) * (w - m.i - m.d);
  const Y = (v) => h - m.b - (v - yr[0]) / (yr[1] - yr[0]) * (h - m.s - m.b);
  ctx.font = "11px system-ui, sans-serif"; ctx.fillStyle = color("--suave"); ctx.strokeStyle = color("--borde"); ctx.lineWidth = 1;
  for (let k = 0; k <= 4; k++) {
    const vy = yr[0] + k * (yr[1] - yr[0]) / 4, py = Y(vy);
    ctx.beginPath(); ctx.moveTo(m.i, py); ctx.lineTo(w - m.d, py); ctx.stroke();
    ctx.textAlign = "right"; ctx.fillText(fmt(vy, Math.abs(yr[1] - yr[0]) < 10 ? 1 : 0), m.i - 6, py + 4);
    const vx = xr[0] + k * (xr[1] - xr[0]) / 4;
    ctx.textAlign = "center"; ctx.fillText(fmt(vx, 0), X(vx), h - m.b + 15);
  }
  ctx.textAlign = "center"; ctx.fillText(xt, (m.i + w - m.d) / 2, h - 4);
  ctx.save(); ctx.translate(11, (m.s + h - m.b) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(yt, 0, 0); ctx.restore();
  return { X, Y };
}

// ---------- pestañas ----------
document.querySelectorAll(".pestanas button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".pestanas button").forEach((o) => o.setAttribute("aria-selected", o === b));
  document.querySelectorAll(".panel").forEach((p) => (p.hidden = p.id !== b.dataset.tab));
  history.replaceState(null, "", "#" + b.dataset.tab);
  if (b.dataset.tab === "reg") dibujarRegresion();
  if (b.dataset.tab === "ser") dibujarSerie();
}));

// ================= 1. CLASIFICACIÓN =================
const NOMBRES_FRIJOL = {
  Area: "Área (px²)", Perimeter: "Perímetro (px)", MajorAxisLength: "Eje mayor (px)", MinorAxisLength: "Eje menor (px)",
  AspectRation: "Relación de aspecto", Eccentricity: "Excentricidad", ConvexArea: "Área convexa (px²)",
  EquivDiameter: "Diámetro equivalente (px)", Extent: "Extensión", Solidity: "Solidez", roundness: "Redondez",
  Compactness: "Compacidad", ShapeFactor1: "Factor de forma 1", ShapeFactor2: "Factor de forma 2",
  ShapeFactor3: "Factor de forma 3", ShapeFactor4: "Factor de forma 4",
};
const decimales = (v) => (Math.abs(v) >= 100 ? 2 : Math.abs(v) >= 1 ? 4 : 6);
let clasReal = null;

function iniciarClasificacion() {
  const C = M.clas;
  $("clas-campos").innerHTML = C.atributos.map((a, i) =>
    `<label>${NOMBRES_FRIJOL[a] || a}<input type="number" step="any" id="clas-x${i}" inputmode="decimal"></label>`).join("");
  $("clas-tipico").innerHTML = `<option value="">elegir variedad…</option>` + C.clases.map((c) => `<option>${c}</option>`).join("");
  C.atributos.forEach((_, i) => $("clas-x" + i).addEventListener("input", () => {
    clasReal = null; $("clas-origen").textContent = "Valores modificados manualmente."; clasificar();
  }));
  $("clas-aleatorio").onclick = () => {
    const e = C.ejemplos[azar(C.ejemplos.length)];
    ponerGrano(e.x); clasReal = e.y;
    $("clas-origen").textContent = "Grano real del conjunto de prueba (no usado en el entrenamiento).";
    $("clas-tipico").value = ""; clasificar();
  };
  $("clas-tipico").onchange = (ev) => {
    if (!ev.target.value) return;
    ponerGrano(C.mediana_clase[ev.target.value]); clasReal = null;
    $("clas-origen").textContent = `Mediana de las medidas de la variedad ${ev.target.value}.`; clasificar();
  };
  $("clas-aleatorio").onclick();
}

function ponerGrano(x) { x.forEach((v, i) => ($("clas-x" + i).value = Number(v).toFixed(decimales(v)))); }

function clasificar() {
  const C = M.clas;
  const x = C.atributos.map((_, i) => parseFloat($("clas-x" + i).value));
  if (x.some(isNaN)) { $("clas-pred").textContent = "—"; return; }
  const p = Red.predecir(C.capas, x.map((v, i) => (v - C.media[i]) / C.desv[i]));
  const k = barras($("clas-barras"), C.clases, p);
  $("clas-pred").textContent = C.clases[k];
  $("clas-real").innerHTML = clasReal === null ? "&nbsp;" : `Variedad real: <b>${C.clases[clasReal]}</b> ${insignia(clasReal === k)}`;
}

// ================= 2. REGRESIÓN =================
const VARS_REG = [
  { n: "Temperatura ambiente (AT)", u: "°C", paso: 0.1 },
  { n: "Vacío de escape (V)", u: "cm Hg", paso: 0.1 },
  { n: "Presión ambiente (AP)", u: "mbar", paso: 0.1 },
  { n: "Humedad relativa (RH)", u: "%", paso: 0.1 },
];
let regSel = null;

function iniciarRegresion() {
  const R = M.reg;
  $("reg-controles").innerHTML = VARS_REG.map((v, i) => `
    <div class="control">
      <label for="reg-x${i}">${v.n} <output id="reg-v${i}"></output></label>
      <input type="range" id="reg-x${i}" min="${R.min[i]}" max="${R.max[i]}" step="${v.paso}">
      <small>${fmt(R.min[i], 1)} – ${fmt(R.max[i], 1)} ${v.u}</small>
    </div>`).join("");
  VARS_REG.forEach((_, i) => $("reg-x" + i).addEventListener("input", () => {
    regSel = null; $("reg-origen").textContent = "Condiciones elegidas manualmente."; predecirRegresion();
  }));
  $("reg-aleatorio").onclick = () => {
    regSel = azar(R.prueba.real.length);
    R.prueba.x[regSel].forEach((v, i) => ($("reg-x" + i).value = v));
    $("reg-origen").textContent = "Registro horario real del conjunto de prueba (no usado en el entrenamiento).";
    predecirRegresion();
  };
  $("reg-reiniciar").onclick = () => {
    regSel = null; R.media_x.forEach((v, i) => ($("reg-x" + i).value = v));
    $("reg-origen").textContent = "Valores promedio de los datos de entrenamiento."; predecirRegresion();
  };
  $("reg-aleatorio").onclick();
}

function potencia(x) {
  const R = M.reg;
  return Red.predecir(R.capas, x.map((v, i) => (v - R.media_x[i]) / R.desv_x[i]))[0] * R.desv_y + R.media_y;
}

function leerReg() { return VARS_REG.map((_, i) => parseFloat($("reg-x" + i).value)); }

function predecirRegresion() {
  const R = M.reg, x = leerReg();
  x.forEach((v, i) => ($("reg-v" + i).textContent = `${fmt(v, 1)} ${VARS_REG[i].u}`));
  const pe = potencia(x);
  $("reg-pred").textContent = fmt(pe, 2);
  $("reg-medidor").style.width = Math.min(100, Math.max(0, (pe - 420) / 76 * 100)) + "%";
  if (regSel === null) $("reg-real").innerHTML = "&nbsp;";
  else {
    const real = R.prueba.real[regSel], err = pe - real;
    $("reg-real").innerHTML = `Potencia real medida: <b>${fmt(real, 2)} MW</b> · error ${err >= 0 ? "+" : ""}${fmt(err, 2)} MW` +
      ` <span class="insignia ${Math.abs(err) < 7.8 ? "ok" : "mal"}">${fmt(Math.abs(err) / real * 100, 2)} %</span>`;
  }
  dibujarRegresion();
}

function dibujarRegresion() {
  const R = M.reg; if (!R || $("reg").hidden) return;
  const x = leerReg(), acento = color("--acento");
  // Curva de sensibilidad respecto a la temperatura
  let { ctx, w, h } = lienzo($("reg-curva"));
  const ts = Array.from({ length: 60 }, (_, k) => R.min[0] + k * (R.max[0] - R.min[0]) / 59);
  const ps = ts.map((t) => potencia([t, x[1], x[2], x[3]]));
  const lo = Math.floor(Math.min(...ps, 420) / 10) * 10;
  const yr = [lo, lo + Math.ceil((Math.max(...ps, 500) - lo) / 40) * 40];
  let e = ejes(ctx, w, h, [R.min[0], R.max[0]], yr, "Temperatura AT (°C)", "PE (MW)");
  ctx.strokeStyle = acento; ctx.lineWidth = 2.2; ctx.beginPath();
  ts.forEach((t, k) => (k ? ctx.lineTo(e.X(t), e.Y(ps[k])) : ctx.moveTo(e.X(t), e.Y(ps[k])))); ctx.stroke();
  const pe = potencia(x);
  ctx.fillStyle = acento; ctx.beginPath(); ctx.arc(e.X(x[0]), e.Y(pe), 5.5, 0, 7); ctx.fill();
  // Dispersión real contra predicho
  ({ ctx, w, h } = lienzo($("reg-dispersion")));
  e = ejes(ctx, w, h, [420, 500], [420, 500], "PE real (MW)", "PE predicha (MW)");
  ctx.strokeStyle = color("--mal"); ctx.setLineDash([5, 4]); ctx.lineWidth = 1.2;
  ctx.beginPath(); ctx.moveTo(e.X(420), e.Y(420)); ctx.lineTo(e.X(500), e.Y(500)); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = color("--barra");
  R.prueba.real.forEach((r, k) => { ctx.beginPath(); ctx.arc(e.X(r), e.Y(R.prueba.pred[k]), 2.4, 0, 7); ctx.fill(); });
  if (regSel !== null) {
    ctx.fillStyle = acento; ctx.strokeStyle = color("--superficie"); ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(e.X(R.prueba.real[regSel]), e.Y(pe), 6.5, 0, 7); ctx.fill(); ctx.stroke();
  }
}

// ================= 3. SERIES DE TIEMPO =================
let serieBase = null, serieReal = null, serieRuido = [];

function iniciarSeries() {
  const S = M.ser;
  $("ser-lista").innerHTML = S.ejemplos.map((e, k) => `<option value="${k}">#${e.id} · ${S.clases[e.y].toLowerCase()}</option>`).join("");
  $("ser-lista").onchange = (ev) => elegirSerie(Number(ev.target.value));
  $("ser-aleatorio").onclick = () => { const k = azar(S.ejemplos.length); $("ser-lista").value = k; elegirSerie(k); };
  $("ser-ruido").oninput = () => { $("ser-ruido-val").textContent = fmt($("ser-ruido").value); diagnosticar(); };
  $("ser-archivo").onchange = async (ev) => {
    const f = ev.target.files[0]; if (!f) return;
    const v = (await f.text()).split(/[\s,;]+/).map(Number).filter((n) => !isNaN(n));
    if (v.length < 500) { alert(`El archivo tiene ${v.length} valores; se necesitan 500.`); return; }
    const s = v.slice(v.length - 500), media = s.reduce((a, b) => a + b, 0) / 500;
    const desv = Math.sqrt(s.reduce((a, b) => a + (b - media) ** 2, 0) / 500) || 1;
    serieBase = s.map((x) => (x - media) / desv); serieReal = null; nuevoRuido(); diagnosticar();
    $("ser-lista").value = "";
  };
  $("ser-evaluar").onclick = () => {
    $("ser-eval").textContent = "Evaluando…";
    setTimeout(() => {
      let ok = 0; const t0 = performance.now();
      S.ejemplos.forEach((e) => { const p = Red.predecir(S.capas, e.x.map((v) => [v])); if ((p[1] > p[0] ? 1 : 0) === e.y) ok++; });
      $("ser-eval").innerHTML = `<b>${ok} de ${S.ejemplos.length}</b> correctas (${fmt(100 * ok / S.ejemplos.length, 1)} %) en ${fmt(performance.now() - t0, 0)} ms.`;
    }, 20);
  };
  $("ser-aleatorio").onclick();
}

function nuevoRuido() {
  serieRuido = Array.from({ length: 500 }, () => {
    const u = 1 - Math.random(), v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  });
}

function elegirSerie(k) {
  const e = M.ser.ejemplos[k]; serieBase = e.x; serieReal = e.y; nuevoRuido(); diagnosticar();
}

function serieActual() {
  const s = parseFloat($("ser-ruido").value);
  return serieBase.map((v, i) => v + s * serieRuido[i]);
}

function diagnosticar() {
  const S = M.ser, x = serieActual();
  const p = Red.predecir(S.capas, x.map((v) => [v]));
  const k = barras($("ser-barras"), S.clases, p);
  $("ser-pred").textContent = S.clases[k];
  $("ser-pred").style.color = k === 1 ? color("--mal") : color("--ok");
  $("ser-real").innerHTML = serieReal === null ? "Serie cargada desde archivo (normalizada)."
    : `Estado real del motor: <b>${S.clases[serieReal]}</b> ${insignia(serieReal === k)}`;
  dibujarSerie();
}

function dibujarSerie() {
  if (!serieBase || $("ser").hidden) return;
  const x = serieActual(), { ctx, w, h } = lienzo($("ser-grafica"));
  const lim = Math.max(3, Math.ceil(Math.max(...x.map(Math.abs))));
  const e = ejes(ctx, w, h, [0, 500], [-lim, lim], "muestra del sensor", "amplitud");
  ctx.strokeStyle = color("--acento"); ctx.lineWidth = 1.2; ctx.beginPath();
  x.forEach((v, i) => (i ? ctx.lineTo(e.X(i), e.Y(v)) : ctx.moveTo(e.X(i), e.Y(v)))); ctx.stroke();
}

// ---------- arranque ----------
(async function () {
  try {
    const [clas, reg, ser] = await Promise.all(["clasificacion", "regresion", "series"].map((n) =>
      fetch(`modelos/${n}.json`).then((r) => { if (!r.ok) throw new Error(n); return r.json(); })));
    M = { clas, reg, ser };
    $("cargando").remove();
    iniciarClasificacion(); iniciarRegresion(); iniciarSeries();
    const tab = location.hash.slice(1);
    if (["clas", "reg", "ser"].includes(tab)) document.querySelector(`[data-tab="${tab}"]`).click();
  } catch (err) {
    $("cargando").textContent = "No se pudieron cargar los modelos. Abre la página desde un servidor web (GitHub Pages) y no como archivo local.";
  }
})();

let espera;
window.addEventListener("resize", () => { clearTimeout(espera); espera = setTimeout(() => { dibujarRegresion(); dibujarSerie(); }, 120); });
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { dibujarRegresion(); dibujarSerie(); });
