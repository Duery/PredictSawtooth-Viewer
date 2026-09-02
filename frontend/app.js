const FAMILY_META = {
  timexer: {
    label: "TimeXer",
    mark: "TX",
    description: "仅接入最新一批已完成的三种损失函数模型。",
  },
  wpmixer: {
    label: "WPMixer",
    mark: "WP",
    description: "多尺度窗口混合预测，当前接入 gap 0 → 10 ms。",
  },
  timesfm: {
    label: "TimesFM 2.5",
    mark: "TF",
    description: "LoRA 微调模型，按 1 kHz 输出未来 20 ms 波形。",
  },
  timesfm3: {
    label: "TimesFM 3.0",
    mark: "T3",
    description: "官方 Zero-shot 与原生多变量版本，复用 2.5 的 1 kHz 时间轴。",
  },
};

const state = {
  options: [],
  family: null,
  option: null,
  shot: "",
  navigation: null,
  prediction: null,
};

const elements = {};

function cssColor(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function applyTheme(theme, persist = true) {
  const dark = theme === "dark";
  document.documentElement.classList.toggle("dark", dark);
  elements["theme-toggle"]?.setAttribute("aria-pressed", dark ? "true" : "false");
  if (elements["theme-label"]) elements["theme-label"].textContent = dark ? "日间模式" : "黑夜模式";
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.content = dark ? "#181818" : "#ffffff";
  if (persist) localStorage.setItem("predict-sawtooth-theme", theme);
  if (elements["window-view"]?.classList.contains("is-active")) {
    requestAnimationFrame(() => {
      drawWaveforms();
      if (state.prediction) drawPredictionResult();
    });
  }
}

function cacheElements() {
  [
    "catalog-count", "model-cards", "variant-count", "variant-list",
    "readiness-badge", "selected-model", "selected-variant", "param-history",
    "param-gap", "param-prediction", "shot-select", "shot-hint", "input-summary",
    "input-channels", "continue-button", "selection-view", "window-view", "back-button",
    "window-model", "window-shot", "window-shot-select", "window-output-rate",
    "window-slider", "window-range", "ip-canvas", "sxr-canvas",
    "theme-toggle", "theme-label",
    "window-min-label", "window-max-label", "prediction-status", "run-prediction",
    "prediction-panel", "inference-latency", "metric-nrmse", "metric-persistence",
    "metric-mae", "metric-correlation", "metric-r2", "metric-sampling-rate",
    "result-canvas", "prediction-provenance",
  ].forEach((id) => { elements[id] = document.getElementById(id); });
}

function uniqueResultCount(options) {
  return new Set(options.map((option) => option.result_dir)).size;
}

function familyOptions(family) {
  return state.options.filter((option) => option.family === family);
}

function renderFamilies() {
  elements["model-cards"].innerHTML = "";
  Object.entries(FAMILY_META).forEach(([family, meta]) => {
    const options = familyOptions(family);
    if (!options.length) return;
    const resultCount = uniqueResultCount(options);
    const readyCount = options.filter((option) => option.inference_ready).length;
    const countLabel = options.every((option) => option.evaluation_scope === "live_inference")
      ? "个模型"
      : "组结果";
    const button = document.createElement("button");
    button.type = "button";
    button.className = `model-card${state.family === family ? " is-active" : ""}`;
    button.setAttribute("aria-pressed", state.family === family ? "true" : "false");
    button.innerHTML = `
      <div class="model-card-top">
        <span class="model-monogram">${meta.mark}</span>
        <h3>${meta.label}</h3>
      </div>
      <p>${meta.description}</p>
      <div class="model-meta">
        <span><strong>${resultCount}</strong> ${countLabel}</span>
        <span>${readyCount ? `${readyCount} 个可实时推理` : "仅保存结果"}</span>
      </div>`;
    button.addEventListener("click", () => selectFamily(family));
    elements["model-cards"].appendChild(button);
  });
}

function renderVariants() {
  const options = familyOptions(state.family);
  elements["variant-count"].textContent = `${options.length} 项`;
  elements["variant-list"].innerHTML = "";
  options.forEach((option) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `variant-option${state.option?.id === option.id ? " is-active" : ""}`;
    button.setAttribute("aria-pressed", state.option?.id === option.id ? "true" : "false");
    const scope = option.evaluation_scope === "validation_only"
      ? "VALIDATION"
      : (option.evaluation_scope === "live_inference" ? "LIVE INFERENCE" : "SHOT METRICS");
    button.innerHTML = `
      <span class="radio-mark" aria-hidden="true"></span>
      <span class="variant-copy">
        <strong>${option.variant_label}</strong>
        <small>${scope} · ${option.shots.length} SHOTS · H160 / G${option.gap_ms} / P${option.prediction_ms}</small>
      </span>
      <span class="variant-status${option.inference_ready ? " ready" : ""}">
        ${option.inference_ready ? "● 可实时推理" : "○ 未接入推理"}
      </span>`;
    button.addEventListener("click", () => selectOption(option));
    elements["variant-list"].appendChild(button);
  });
}

function selectFamily(family) {
  state.family = family;
  state.option = familyOptions(family)[0] || null;
  state.shot = state.option?.shots[0] || "";
  resetInferenceState();
  renderAll();
}

function selectOption(option) {
  state.option = option;
  state.shot = option.shots[0] || "";
  resetInferenceState();
  renderVariants();
  renderSelectionPanel();
}

function renderShots(option) {
  const select = elements["shot-select"];
  select.innerHTML = "";
  if (!option || !option.shots.length) {
    select.disabled = true;
    select.innerHTML = '<option value="">当前结果未找到逐炮记录</option>';
    elements["shot-hint"].textContent = "结果配置存在，但没有可用于选择页的炮号字段。";
    return;
  }
  option.shots.forEach((shot) => {
    const item = document.createElement("option");
    item.value = shot;
    item.textContent = `Shot ${shot}`;
    item.selected = state.shot === shot;
    select.appendChild(item);
  });
  select.disabled = false;
  elements["shot-hint"].textContent = `当前结果包含 ${option.shots.length} 炮；切换结果配置会同步更新。`;
}

function renderWindowShots() {
  const select = elements["window-shot-select"];
  select.innerHTML = "";
  (state.option?.shots || []).forEach((shot) => {
    const item = document.createElement("option");
    item.value = shot;
    item.textContent = `Shot ${shot}`;
    item.selected = state.shot === shot;
    select.appendChild(item);
  });
  select.disabled = !state.option?.shots?.length;
}

function renderChannels(option) {
  elements["input-channels"].innerHTML = "";
  if (!option) return;
  const visible = option.input_channels.slice(0, 7);
  visible.forEach((channel) => {
    const tag = document.createElement("span");
    tag.className = "channel-tag";
    tag.textContent = channel;
    elements["input-channels"].appendChild(tag);
  });
  if (option.input_channels.length > visible.length) {
    const tag = document.createElement("span");
    tag.className = "channel-tag more";
    tag.textContent = `+${option.input_channels.length - visible.length} 路`;
    elements["input-channels"].appendChild(tag);
  }
}

function renderSelectionPanel() {
  const option = state.option;
  if (!option) {
    elements["selected-model"].textContent = "—";
    elements["selected-variant"].textContent = "请选择一个结果配置";
    elements["continue-button"].disabled = true;
    return;
  }
  elements["selected-model"].textContent = option.family_label;
  elements["selected-variant"].textContent = option.variant_label;
  elements["param-history"].textContent = `${option.history_ms} ms`;
  elements["param-gap"].textContent = `${option.gap_ms} ms`;
  elements["param-prediction"].textContent = `${option.prediction_ms} ms`;
  elements["input-summary"].textContent = option.input_summary;
  elements["readiness-badge"].textContent = option.inference_ready ? "可实时推理" : "未接入推理";
  elements["readiness-badge"].className = `readiness-badge${option.inference_ready ? " ready" : ""}`;
  renderShots(option);
  renderChannels(option);
  elements["continue-button"].disabled = !state.shot || !option.inference_ready;
}

function renderAll() {
  renderFamilies();
  renderVariants();
  renderSelectionPanel();
}

function resetInferenceState() {
  state.navigation = null;
  state.prediction = null;
  if (elements["prediction-panel"]) elements["prediction-panel"].hidden = true;
}

async function showWindowView() {
  if (!state.option || !state.shot) return;
  elements["selection-view"].classList.remove("is-active");
  elements["window-view"].classList.add("is-active");
  document.querySelectorAll(".step").forEach((step) => {
    const value = Number(step.dataset.step);
    step.classList.toggle("is-active", value === 2);
    step.classList.toggle("is-complete", value === 1);
  });
  elements["window-model"].textContent = `${state.option.family_label} · ${state.option.variant_label}`;
  elements["window-shot"].textContent = `Shot ${state.shot}`;
  elements["window-output-rate"].textContent = formatSamplingRate(state.option.prediction_sampling_hz);
  renderWindowShots();
  window.scrollTo({ top: 0, behavior: "smooth" });
  await loadNavigation();
}

function showSelectionView() {
  elements["window-view"].classList.remove("is-active");
  elements["selection-view"].classList.add("is-active");
  document.querySelectorAll(".step").forEach((step) => {
    step.classList.toggle("is-active", Number(step.dataset.step) === 1);
    step.classList.remove("is-complete");
  });
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function formatTime(value) {
  return Number.isInteger(value) ? `${value}` : Number(value).toFixed(1);
}

function formatSamplingRate(value) {
  const rate = Number(value);
  if (!Number.isFinite(rate) || rate <= 0) return "—";
  if (rate >= 1000) {
    const kilohertz = rate / 1000;
    return `${Number.isInteger(kilohertz) ? kilohertz : kilohertz.toFixed(1)} kHz`;
  }
  return `${rate} Hz`;
}

function nearestValidHistoryEnd(value) {
  const ranges = state.navigation?.valid_history_end_ranges_ms || [];
  if (!ranges.length) return value;
  for (const [start, end] of ranges) {
    if (value >= start && value <= end) return value;
  }
  return ranges.flat().reduce((nearest, boundary) => (
    Math.abs(boundary - value) < Math.abs(nearest - value) ? boundary : nearest
  ), ranges[0][0]);
}

async function loadNavigation() {
  resetInferenceState();
  elements["prediction-status"].className = "";
  elements["prediction-status"].textContent = "正在加载真实 IP + SXR 波形…";
  elements["run-prediction"].disabled = true;
  elements["window-shot-select"].disabled = true;
  try {
    const query = new URLSearchParams({
      model_package_id: state.option.model_package_id,
      shot: state.shot,
    });
    const response = await fetch(`/api/navigation?${query}`, { cache: "no-store" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    state.navigation = payload;
    elements["window-output-rate"].textContent = formatSamplingRate(payload.prediction_sampling_hz);
    const ranges = payload.valid_history_end_ranges_ms;
    const minimum = Math.min(...ranges.map((range) => range[0]));
    const maximum = Math.max(...ranges.map((range) => range[1]));
    const slider = elements["window-slider"];
    slider.min = minimum;
    slider.max = maximum;
    slider.step = 1000 / payload.sampling_hz;
    slider.value = payload.default_history_end_ms;
    elements["window-min-label"].textContent = `${formatTime(minimum)} ms`;
    elements["window-max-label"].textContent = `${formatTime(maximum)} ms`;
    elements["prediction-status"].textContent = `真实数据已就绪 · ${ranges.length} 个有效 NBI 区间`;
    elements["prediction-status"].className = "ready";
    elements["run-prediction"].disabled = false;
    drawWaveforms();
  } catch (error) {
    elements["prediction-status"].textContent = `无法加载：${error.message}`;
    elements["prediction-status"].className = "error";
    elements["run-prediction"].disabled = true;
    drawWaveforms();
  } finally {
    elements["window-shot-select"].disabled = !state.option?.shots?.length;
  }
}

function canvasContext(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, canvas.clientWidth);
  const height = Math.max(1, canvas.clientHeight);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { context, width, height };
}

function drawGrid(context, width, height) {
  context.strokeStyle = cssColor("--plot-grid");
  context.lineWidth = 1;
  for (let index = 1; index < 8; index += 1) {
    const x = (index / 8) * width;
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, height);
    context.stroke();
  }
  for (let index = 1; index < 4; index += 1) {
    const y = (index / 4) * height;
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(width, y);
    context.stroke();
  }
}

function drawSelection(context, width, height) {
  if (!state.navigation?.time_ms?.length) return;
  const minimum = state.navigation.time_ms[0];
  const maximum = state.navigation.time_ms.at(-1);
  const historyEnd = Number(elements["window-slider"].value);
  const historyStart = historyEnd - state.navigation.history_ms;
  const left = ((historyStart - minimum) / (maximum - minimum)) * width;
  const bandWidth = (state.navigation.history_ms / (maximum - minimum)) * width;
  context.fillStyle = cssColor("--selection-fill");
  context.fillRect(left, 0, bandWidth, height);
  context.strokeStyle = cssColor("--blue");
  context.lineWidth = 1.5;
  context.strokeRect(left + 0.75, 0.75, bandWidth - 1.5, height - 1.5);
}

function drawSignal(canvas, kind) {
  const { context, width, height } = canvasContext(canvas);
  context.clearRect(0, 0, width, height);
  context.fillStyle = cssColor("--card");
  context.fillRect(0, 0, width, height);
  drawGrid(context, width, height);
  drawSelection(context, width, height);
  const navigation = state.navigation;
  if (!navigation?.time_ms?.length) return;
  const values = kind === "ip" ? navigation.ip : navigation.sxr;
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  const span = Math.max(1e-9, maximum - minimum);
  context.beginPath();
  for (let index = 0; index < values.length; index += 1) {
    const x = (index / (values.length - 1)) * width;
    const normalized = (values[index] - minimum) / span;
    const y = height - 8 - normalized * (height - 16);
    if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
  }
  context.strokeStyle = kind === "ip" ? cssColor("--cyan") : cssColor("--red");
  context.lineWidth = 1.55;
  context.stroke();
}

function updateWindowReadout() {
  if (!state.navigation) {
    elements["window-range"].textContent = "—";
    return;
  }
  const end = Number(elements["window-slider"].value);
  const start = end - state.navigation.history_ms;
  elements["window-range"].textContent = `${formatTime(start)}–${formatTime(end)} ms`;
}

function drawWaveforms() {
  updateWindowReadout();
  drawSignal(elements["ip-canvas"], "ip");
  drawSignal(elements["sxr-canvas"], "sxr");
}

function drawLine(context, times, values, bounds, color, width = 1.7, dash = []) {
  if (!times?.length || !values?.length) return;
  const { minTime, maxTime, minValue, maxValue, canvasWidth, canvasHeight } = bounds;
  const timeSpan = Math.max(1e-9, maxTime - minTime);
  const valueSpan = Math.max(1e-9, maxValue - minValue);
  context.beginPath();
  values.forEach((value, index) => {
    const x = ((times[index] - minTime) / timeSpan) * canvasWidth;
    const y = canvasHeight - 12 - ((value - minValue) / valueSpan) * (canvasHeight - 24);
    if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
  });
  context.strokeStyle = color;
  context.lineWidth = width;
  context.setLineDash(dash);
  context.stroke();
  context.setLineDash([]);
}

function drawPredictionResult() {
  const result = state.prediction;
  if (!result) return;
  const { context, width, height } = canvasContext(elements["result-canvas"]);
  context.clearRect(0, 0, width, height);
  context.fillStyle = cssColor("--card");
  context.fillRect(0, 0, width, height);
  drawGrid(context, width, height);
  const historyTimes = result.history.time_ms;
  const forecastTimes = result.forecast.time_ms;
  const allValues = [
    ...result.history.sxr,
    ...result.forecast.ground_truth,
    ...result.forecast.prediction,
    ...result.forecast.persistence,
  ];
  const minValue = Math.min(...allValues);
  const maxValue = Math.max(...allValues);
  const padding = Math.max(1e-6, (maxValue - minValue) * 0.08);
  const bounds = {
    minTime: historyTimes[0],
    maxTime: forecastTimes.at(-1),
    minValue: minValue - padding,
    maxValue: maxValue + padding,
    canvasWidth: width,
    canvasHeight: height,
  };
  drawLine(context, historyTimes, result.history.sxr, bounds, cssColor("--foreground"), 1.35);
  drawLine(context, forecastTimes, result.forecast.ground_truth, bounds, cssColor("--red"), 2.1);
  drawLine(context, forecastTimes, result.forecast.prediction, bounds, cssColor("--blue"), 2.1);
  drawLine(context, forecastTimes, result.forecast.persistence, bounds, cssColor("--muted-foreground"), 1.4, [5, 4]);
}

function metricText(value) {
  return value === null || value === undefined || !Number.isFinite(value) ? "—" : Number(value).toFixed(3);
}

function renderPrediction(payload) {
  state.prediction = payload;
  elements["prediction-panel"].hidden = false;
  elements["inference-latency"].textContent = `${payload.inference_ms.toFixed(1)} ms · ${payload.device.toUpperCase()}`;
  elements["metric-nrmse"].textContent = metricText(payload.metrics.nrmse);
  elements["metric-persistence"].textContent = metricText(payload.metrics.persistence_nrmse);
  elements["metric-mae"].textContent = metricText(payload.metrics.mae);
  elements["metric-correlation"].textContent = metricText(payload.metrics.correlation);
  elements["metric-r2"].textContent = metricText(payload.metrics.r2);
  elements["metric-sampling-rate"].textContent = formatSamplingRate(payload.provenance.output_sampling_hz);
  elements["prediction-provenance"].textContent = `实时推理 · Shot ${payload.shot} · ${payload.prediction_range_ms.map(formatTime).join("–")} ms · ${formatSamplingRate(payload.provenance.output_sampling_hz)} · 回顾性窗口 · 指纹 ${payload.provenance.package_fingerprint.slice(0, 12)}`;
  requestAnimationFrame(drawPredictionResult);
}

async function runPrediction() {
  if (!state.navigation || !state.option?.model_package_id) return;
  elements["run-prediction"].disabled = true;
  elements["prediction-status"].className = "";
  elements["prediction-status"].textContent = "正在加载 checkpoint 并执行推理…";
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model_package_id: state.option.model_package_id,
        shot: state.shot,
        history_end_ms: Number(elements["window-slider"].value),
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    renderPrediction(payload);
    elements["prediction-status"].textContent = `预测完成 · ${payload.model_display_name}`;
    elements["prediction-status"].className = "ready";
  } catch (error) {
    elements["prediction-status"].textContent = `预测失败：${error.message}`;
    elements["prediction-status"].className = "error";
  } finally {
    elements["run-prediction"].disabled = false;
  }
}

function renderEmptyState(message) {
  elements["model-cards"].innerHTML = `<div class="empty-state"><strong>未发现可用结果</strong>${message}</div>`;
  elements["catalog-count"].textContent = "0 项结果";
  elements["variant-list"].innerHTML = "";
  elements["variant-count"].textContent = "0 项";
}

async function initialize() {
  cacheElements();
  applyTheme(document.documentElement.classList.contains("dark") ? "dark" : "light", false);
  elements["theme-toggle"].addEventListener("click", () => {
    applyTheme(document.documentElement.classList.contains("dark") ? "light" : "dark");
  });
  elements["shot-select"].addEventListener("change", (event) => {
    state.shot = event.target.value;
    resetInferenceState();
    elements["continue-button"].disabled = !state.shot || !state.option?.inference_ready;
  });
  elements["window-shot-select"].addEventListener("change", async (event) => {
    state.shot = event.target.value;
    elements["shot-select"].value = state.shot;
    elements["window-shot"].textContent = `Shot ${state.shot}`;
    await loadNavigation();
  });
  elements["continue-button"].addEventListener("click", showWindowView);
  elements["back-button"].addEventListener("click", showSelectionView);
  elements["window-slider"].addEventListener("input", () => {
    const snapped = nearestValidHistoryEnd(Number(elements["window-slider"].value));
    elements["window-slider"].value = snapped;
    state.prediction = null;
    elements["prediction-panel"].hidden = true;
    elements["prediction-status"].textContent = "窗口已更新，可以运行预测";
    elements["prediction-status"].className = "ready";
    drawWaveforms();
  });
  elements["run-prediction"].addEventListener("click", runPrediction);
  window.addEventListener("resize", () => {
    if (elements["window-view"].classList.contains("is-active")) {
      drawWaveforms();
      if (state.prediction) drawPredictionResult();
    }
  });

  try {
    const response = await fetch("/api/catalog", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    state.options = Array.isArray(payload.options) ? payload.options : [];
    if (!state.options.length) {
      renderEmptyState("请检查 config.toml 中的只读结果目录。");
      return;
    }
    const family = ["timexer", "wpmixer", "timesfm", "timesfm3"].find((name) => familyOptions(name).length);
    state.family = family;
    state.option = familyOptions(family)[0];
    state.shot = state.option.shots[0] || "";
    elements["catalog-count"].textContent = `${state.options.length} 项配置`;
    renderAll();
  } catch (error) {
    renderEmptyState("目录接口暂时不可用，请重新启动本地服务。");
    console.error(error);
  }
}

document.addEventListener("DOMContentLoaded", initialize);
