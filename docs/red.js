// Inferencia de las redes entrenadas en Keras, en JavaScript puro.
// Cada capa se exporta con src/exportar_web.py. Una entrada densa es un vector [n];
// una entrada convolucional es una matriz [tiempo][canales].
(function (raiz) {
  const activar = {
    linear: (v) => v,
    relu: (v) => Math.max(0, v),
    tanh: Math.tanh,
  };

  function dense(x, c) {
    const nOut = c.b.length, f = activar[c.act] || activar.linear;
    const y = new Array(nOut);
    for (let j = 0; j < nOut; j++) {
      let s = c.b[j];
      for (let i = 0; i < x.length; i++) s += x[i] * c.W[i][j];
      y[j] = f(s);
    }
    if (c.act === "softmax") return softmax(y);
    return y;
  }

  function softmax(z) {
    const m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  }

  // Conv1D con padding "same" (stride 1): W[k][canalEntrada][filtro]
  function conv1d(x, c) {
    const T = x.length, K = c.W.length, cin = c.W[0].length, cout = c.b.length;
    const izq = Math.floor((K - 1) / 2), f = activar[c.act] || activar.linear;
    const y = new Array(T);
    for (let t = 0; t < T; t++) {
      const fila = new Float64Array(cout);
      for (let o = 0; o < cout; o++) fila[o] = c.b[o];
      for (let k = 0; k < K; k++) {
        const tt = t + k - izq;
        if (tt < 0 || tt >= T) continue;
        const xt = x[tt], Wk = c.W[k];
        for (let i = 0; i < cin; i++) {
          const xi = xt[i], Wki = Wk[i];
          if (xi === 0) continue;
          for (let o = 0; o < cout; o++) fila[o] += xi * Wki[o];
        }
      }
      for (let o = 0; o < cout; o++) fila[o] = f(fila[o]);
      y[t] = fila;
    }
    return y;
  }

  function bn(x, c) {
    if (typeof x[0] === "number") return x.map((v, i) => v * c.escala[i] + c.desp[i]);
    return x.map((fila) => fila.map((v, i) => v * c.escala[i] + c.desp[i]));
  }

  function relu(x) {
    if (typeof x[0] === "number") return x.map((v) => Math.max(0, v));
    return x.map((fila) => fila.map((v) => Math.max(0, v)));
  }

  function gap(x) {
    const n = x[0].length, s = new Array(n).fill(0);
    for (const fila of x) for (let i = 0; i < n; i++) s[i] += fila[i];
    return s.map((v) => v / x.length);
  }

  function predecir(capas, x) {
    let h = x;
    for (const c of capas) {
      if (c.tipo === "dense") h = dense(h, c);
      else if (c.tipo === "conv1d") h = conv1d(h, c);
      else if (c.tipo === "bn") h = bn(h, c);
      else if (c.tipo === "relu") h = relu(h);
      else if (c.tipo === "gap") h = gap(h);
    }
    return h;
  }

  const Red = { predecir, softmax };
  if (typeof module !== "undefined") module.exports = Red;
  else raiz.Red = Red;
})(this);
