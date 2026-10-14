/* LUT chat — frontend.
 *
 * The one idea worth stating up front: these checkpoints are next-token models, so a
 * "conversation" is really one growing piece of text. `state.context` holds that text
 * literally — the user's messages and the model's continuations concatenated in order —
 * and it, not the bubble list, is what gets tokenized and sent. The bubbles are only how
 * it is displayed.
 */

const $ = (id) => document.getElementById(id);

const state = {
  runs: [],
  runId: null,
  kind: 'best',
  context: '',        // the literal text the model is continuing
  lastPrompt: null,   // prompt behind the most recent reply, for Regenerate
  lastContext: null,  // state.context as it stood before that reply
  lastTurn: null,     // the DOM node that reply lives in, so Regenerate can replace it
  generating: false,
  controller: null,
};

/* ------------------------------------------------------------------ utils */

function fmt(value, digits = 4) {
  return (value === null || value === undefined) ? '—' : Number(value).toFixed(digits);
}

function fmtPct(value) {
  return (value === null || value === undefined) ? '—' : (value * 100).toFixed(1) + '%';
}

function fmtSize(bytes) {
  if (!bytes) return '—';
  const mb = bytes / 1e6;
  return mb >= 1000 ? (mb / 1000).toFixed(2) + ' GB' : mb.toFixed(0) + ' MB';
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

async function postJSON(url, payload) {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || ('HTTP ' + response.status));
  return data;
}

/* ------------------------------------------------------------- model list */

async function loadRuns() {
  const list = $('run-list');
  list.textContent = '';
  list.append(el('p', 'hint', 'scanning…'));
  let data;
  try {
    data = await (await fetch('/api/runs')).json();
  } catch (err) {
    list.textContent = '';
    list.append(el('p', 'hint', 'could not reach the server: ' + err.message));
    return;
  }
  state.runs = data.runs || [];
  $('env-line').textContent = `${state.runs.length} model${state.runs.length === 1 ? '' : 's'} · ${data.device}`;
  $('results-root').textContent = data.results_root;
  renderRuns();
}

function renderRuns() {
  const list = $('run-list');
  list.textContent = '';
  if (!state.runs.length) {
    list.append(el('p', 'hint',
      'No LM runs with a checkpoint on disk. A run appears here once it has written ' +
      'model.pt or model_best.pt.'));
    return;
  }
  let sweep = null;
  for (const run of state.runs) {
    if (run.sweep !== sweep) {
      sweep = run.sweep;
      list.append(el('div', 'sweep-label', sweep));
    }
    list.append(runCard(run));
  }
}

function runCard(run) {
  const card = el('div', 'run-card');
  card.dataset.runId = run.run_id;
  if (run.run_id === state.runId) card.classList.add('selected');

  const head = el('div', 'arm');
  const name = el('span', null, run.arm || run.name);
  // Which network, not which sweep. Two engines share this list because they share a
  // corpus, a tokenizer and a validation window set, which is exactly what makes their
  // perplexities comparable — but you should never be unsure which one just replied.
  if (run.engine) {
    const badge = el('span', 'engine-badge ' + run.engine,
      run.engine === 'nanogpt' ? 'GPT' : 'LUT');
    badge.title = run.engine === 'nanogpt'
      ? 'nanoGPT transformer baseline (gpt_baseline_train.py)'
      : 'LUT model (lut_torch.train)';
    name.append(document.createTextNode(' '));
    name.append(badge);
  }
  head.append(name);
  head.append(el('span', 'seed', run.seed ? 'seed ' + run.seed : run.timestamp));
  card.append(head);

  const m = run.metrics || {};
  const metrics = el('div', 'metrics');
  const add = (label, value) => {
    const span = el('span', null, label + ' ');
    span.append(el('b', null, value));
    metrics.append(span);
  };
  add('val loss', fmt(m.best_loss));
  add('ppl', m.best_perplexity === null || m.best_perplexity === undefined
    ? '—' : Number(m.best_perplexity).toFixed(1));
  add('acc', fmtPct(m.best_accuracy));
  if (m.best_epoch !== null && m.best_epoch !== undefined) {
    add('epoch', Number(m.best_epoch).toFixed(2));
  }
  card.append(metrics);

  if (m.best_perplexity_no_unk !== null && m.best_perplexity_no_unk !== undefined) {
    const nounk = el('div', 'metrics');
    const span = el('span', null, 'without <unk>: ppl ');
    span.append(el('b', null, Number(m.best_perplexity_no_unk).toFixed(1)));
    span.append(document.createTextNode(' · acc '));
    span.append(el('b', null, fmtPct(m.best_accuracy_no_unk)));
    nounk.append(span);
    card.append(nounk);
  }

  const row = el('div', 'ckpt-row');
  for (const ckpt of run.checkpoints) {
    const option = el('div', 'ckpt-option',
      `${ckpt.kind} · ${fmtSize(ckpt.size_bytes)}`);
    option.title = ckpt.label + '\n' + ckpt.path;
    if (run.run_id === state.runId && ckpt.kind === state.kind) {
      option.classList.add('active');
    }
    option.addEventListener('click', (event) => {
      event.stopPropagation();
      selectRun(run.run_id, ckpt.kind);
    });
    row.append(option);
  }
  card.append(row);

  card.addEventListener('click', () => {
    const preferred = run.checkpoints.some((c) => c.kind === 'best') ? 'best' : 'last';
    selectRun(run.run_id, preferred);
  });
  return card;
}

async function selectRun(runId, kind) {
  state.runId = runId;
  state.kind = kind;
  renderRuns();
  const run = state.runs.find((r) => r.run_id === runId);
  $('active-name').textContent = (run.arm || run.name) + '  ·  ' + kind;
  $('active-meta').textContent = 'loading checkpoint…';
  $('send').disabled = true;
  refreshTokens();
  try {
    const info = await postJSON('/api/model', { run_id: runId, kind });
    const m = info.model;
    const parts = [
      m.engine === 'nanogpt' ? 'nanoGPT transformer' : 'LUT',
      `${m.num_layers}L × ${m.num_heads}H`,
      `context ${m.context_size}`,
      `vocab ${m.vocab_size.toLocaleString()}`,
    ];
    if (m.engine === 'nanogpt') {
      parts.push(`width ${m.n_embd}`, `${(m.parameters / 1e6).toFixed(2)}M params`);
    } else {
      parts.push(`${(m.table_entries / 1e6).toFixed(1)}M table entries`,
        m.output_code ? `code ${m.output_code}` : 'uncoded head');
    }
    parts.push(`step ${m.step.toLocaleString()}`, m.device);
    if (info.checkpoint_kind !== kind) {
      state.kind = info.checkpoint_kind;
      parts.unshift(`fell back to the ${info.checkpoint_kind} checkpoint`);
      renderRuns();
    }
    $('active-meta').textContent = parts.join(' · ');
    $('send').disabled = false;
  } catch (err) {
    $('active-meta').textContent = 'could not load: ' + err.message;
  }
  refreshTokens();
}

/* ------------------------------------------------------------- tokenizing */

let tokenTimer = null;

function effectivePrompt(userText) {
  const whole = $('whole-transcript').checked;
  if (!whole) return userText;
  if (!state.context) return userText;
  if (!userText) return state.context;
  return state.context + '\n' + userText;
}

function refreshTokens() {
  clearTimeout(tokenTimer);
  tokenTimer = setTimeout(showTokens, 160);
}

async function showTokens() {
  const strip = $('token-strip');
  const text = effectivePrompt($('input').value);
  if (!state.runId || !text.trim()) {
    strip.classList.add('hidden');
    $('token-count').textContent = ' ';
    return;
  }
  let data;
  try {
    data = await postJSON('/api/tokenize', { run_id: state.runId, text });
  } catch (err) {
    strip.classList.add('hidden');
    $('token-count').textContent = 'tokenizer unavailable: ' + err.message;
    return;
  }
  strip.textContent = '';
  data.tokens.forEach((token, index) => {
    const chip = el('span', 'tok-chip', token === '<eol>' ? '' : token);
    if (data.unk[index]) chip.classList.add('unk');
    if (token === '<eol>') chip.classList.add('eol');
    if (index < data.dropped) chip.classList.add('dropped');
    chip.title = `id ${data.ids[index]}`
      + (data.unk[index] ? ' — not in the vocabulary' : '')
      + (index < data.dropped ? ' — outside the context window' : '');
    strip.append(chip);
  });
  strip.classList.remove('hidden');

  const bits = [`${data.count} tokens`];
  if (data.unk_count) {
    bits.push(`${data.unk_count} became <unk> — not in the vocabulary`);
  }
  if (data.dropped) {
    bits.push(`the first ${data.dropped} fall outside the ${data.context_size}-token window`);
  }
  $('token-count').textContent = bits.join(' · ');
}

/* ------------------------------------------------------------- transcript */

function addTurn(role, text) {
  $('placeholder').classList.add('hidden');
  const turn = el('div', 'turn ' + role);
  const bubble = el('div', 'bubble');
  bubble.textContent = text || '';
  turn.append(bubble);
  $('transcript').append(turn);
  scrollDown();
  return { turn, bubble };
}

function scrollDown() {
  const transcript = $('transcript');
  transcript.scrollTop = transcript.scrollHeight;
}

function renderInspector(turn, start) {
  if (!start) return;
  const box = el('div', 'inspect');
  box.append(el('h4', null, `Context fed to the model (${start.context.length} tokens)`));
  const strip = el('div', 'token-strip');
  strip.style.maxHeight = '84px';
  for (const item of start.context) {
    const chip = el('span', 'tok-chip', item.token === '<eol>' ? '' : item.token);
    if (item.id === 0) chip.classList.add('unk');
    if (item.token === '<eol>') chip.classList.add('eol');
    strip.append(chip);
  }
  box.append(strip);
  turn.append(box);

  const probs = el('div', 'inspect');
  probs.append(el('h4', null, 'Most likely next token at the first step'));
  const top = start.top_next[0] ? start.top_next[0].prob : 1;
  for (const item of start.top_next) {
    const row = el('div', 'prob-row');
    row.append(el('span', 'tok', item.token));
    const bar = el('span', 'bar');
    bar.style.width = Math.max(2, (item.prob / top) * 150) + 'px';
    row.append(bar);
    row.append(el('span', 'pct', (item.prob * 100).toFixed(1) + '%'));
    probs.append(row);
  }
  turn.append(probs);
}

/* ------------------------------------------------------------- generation */

function options(prompt) {
  const seed = $('seed').value.trim();
  return {
    run_id: state.runId,
    kind: state.kind,
    prompt,
    max_new_tokens: Number($('max-tokens').value),
    temperature: Number($('temperature').value),
    top_k: Number($('top-k').value),
    top_p: Number($('top-p').value),
    suppress_unk: $('suppress-unk').checked,
    stop_at_eol: $('stop-at-eol').checked,
    seed: seed === '' ? null : Number(seed),
  };
}

async function send() {
  if (state.generating || !state.runId) return;
  const input = $('input');
  const userText = input.value.trim();
  if (!userText) return;

  addTurn('user', userText);
  state.context = state.context ? state.context + '\n' + userText : userText;
  input.value = '';
  input.style.height = 'auto';
  refreshTokens();

  const whole = $('whole-transcript').checked;
  await run(whole ? state.context : userText);
}

async function regenerate() {
  if (state.generating || state.lastPrompt === null) return;
  state.context = state.lastContext;
  if (state.lastTurn) state.lastTurn.remove();
  await run(state.lastPrompt);
}

async function run(prompt) {
  state.lastPrompt = prompt;
  state.lastContext = state.context;

  const { turn, bubble } = addTurn('model', '');
  const cursor = el('span', 'cursor', ' ');
  bubble.append(cursor);

  setGenerating(true);
  state.controller = new AbortController();
  const began = performance.now();
  let tokens = 0;
  let startEvent = null;

  try {
    const response = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(options(prompt)),
      signal: state.controller.signal,
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.error || ('HTTP ' + response.status));
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let newline;
      while ((newline = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, newline).trim();
        buffer = buffer.slice(newline + 1);
        if (!line) continue;
        const event = JSON.parse(line);
        if (event.type === 'start') {
          startEvent = event;
        } else if (event.type === 'token') {
          cursor.remove();
          bubble.append(document.createTextNode(event.text));
          bubble.append(cursor);
          state.context += event.text;
          tokens += 1;
          scrollDown();
        }
      }
    }
  } catch (err) {
    if (err.name !== 'AbortError') {
      cursor.remove();
      turn.classList.remove('model');
      turn.classList.add('error');
      bubble.textContent = err.message;
      setGenerating(false);
      return;
    }
  }

  cursor.remove();
  if (!bubble.textContent.trim()) bubble.textContent = '(nothing generated)';

  const elapsed = (performance.now() - began) / 1000;
  const foot = el('div', 'turn-foot');
  foot.append(el('span', null,
    `${tokens} token${tokens === 1 ? '' : 's'} · ${elapsed.toFixed(1)}s · ` +
    `${(tokens / Math.max(elapsed, 1e-6)).toFixed(1)} tok/s`));
  const again = el('button', null, 'Regenerate');
  again.addEventListener('click', regenerate);
  foot.append(again);
  if (startEvent) {
    const inspect = el('button', null, 'What the model saw');
    let shown = false;
    inspect.addEventListener('click', () => {
      if (shown) {
        turn.querySelectorAll('.inspect').forEach((node) => node.remove());
        inspect.textContent = 'What the model saw';
      } else {
        renderInspector(turn, startEvent);
        inspect.textContent = 'Hide what the model saw';
      }
      shown = !shown;
      scrollDown();
    });
    foot.append(inspect);
  }
  turn.append(foot);

  setGenerating(false);
  refreshTokens();
  scrollDown();
}

function setGenerating(active) {
  state.generating = active;
  $('send').classList.toggle('hidden', active);
  $('stop').classList.toggle('hidden', !active);
  $('input').disabled = active;
}

/* ------------------------------------------------------------------ wiring */

function bindSlider(id, format) {
  const input = $(id);
  const label = $(id + '-value');
  const update = () => { label.textContent = format(input.value); };
  input.addEventListener('input', update);
  update();
}

function init() {
  bindSlider('temperature', (v) => Number(v).toFixed(2));
  bindSlider('top-k', (v) => (Number(v) === 0 ? 'off' : v));
  bindSlider('top-p', (v) => Number(v).toFixed(2));
  bindSlider('max-tokens', (v) => v);

  $('refresh').addEventListener('click', loadRuns);
  $('toggle-settings').addEventListener('click',
    () => $('settings').classList.toggle('hidden'));
  $('clear-chat').addEventListener('click', () => {
    state.context = '';
    state.lastPrompt = null;
    state.lastTurn = null;
    $('transcript').querySelectorAll('.turn').forEach((node) => node.remove());
    $('placeholder').classList.remove('hidden');
    refreshTokens();
  });
  $('send').addEventListener('click', send);
  $('stop').addEventListener('click', () => state.controller && state.controller.abort());
  $('whole-transcript').addEventListener('change', refreshTokens);

  const input = $('input');
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 180) + 'px';
    refreshTokens();
  });
  input.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      send();
    }
  });

  loadRuns();
}

init();
