import { TrackVisualizer } from "../web/static/visualizer.js";

const THEME_KEY = "fly-gui-theme";

const state = {
  defaults: null,
  currentJobId: null,
  resultLoadedFor: null,
  pollTimer: null,
  externalVideoPath: "",
};

const elements = {
  form: document.querySelector("#jobForm"),
  videoSelect: document.querySelector("#videoSelect"),
  modelSelect: document.querySelector("#modelSelect"),
  gtSelect: document.querySelector("#gtSelect"),
  pickVideo: document.querySelector("#pickVideo"),
  pickedVideoLabel: document.querySelector("#pickedVideoLabel"),
  confInput: document.querySelector("#confInput"),
  confValue: document.querySelector("#confValue"),
  iouInput: document.querySelector("#iouInput"),
  iouValue: document.querySelector("#iouValue"),
  trailInput: document.querySelector("#trailInput"),
  trailValue: document.querySelector("#trailValue"),
  statusPill: document.querySelector("#statusPill"),
  statusText: document.querySelector("#statusText"),
  startButton: document.querySelector("#startButton"),
  cancelButton: document.querySelector("#cancelButton"),
  stageTitle: document.querySelector("#stageTitle"),
  frameCounter: document.querySelector("#frameCounter"),
  progressBar: document.querySelector("#progressBar"),
  resultVideo: document.querySelector("#resultVideo"),
  videoEmpty: document.querySelector("#videoEmpty"),
  logOutput: document.querySelector("#logOutput"),
  metricsGrid: document.querySelector("#metricsGrid"),
  eventList: document.querySelector("#eventList"),
  artifactList: document.querySelector("#artifactList"),
  trackStride: document.querySelector("#trackStride"),
  reloadTracks: document.querySelector("#reloadTracks"),
  copyJobId: document.querySelector("#copyJobId"),
};

const visualizer = new TrackVisualizer(document.querySelector("#trajectoryStage"));

init();

async function init() {
  bindTheme();
  bindControls();
  renderEmptyResults();
  await loadDefaults();
}

function bindTheme() {
  const savedTheme = localStorage.getItem(THEME_KEY) || "microscope-dark";
  applyTheme(savedTheme);
  document.querySelectorAll("[data-theme-choice]").forEach((button) => {
    button.addEventListener("click", () => applyTheme(button.dataset.themeChoice));
  });
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem(THEME_KEY, theme);
  document.querySelectorAll("[data-theme-choice]").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.themeChoice === theme));
  });
  requestAnimationFrame(() => visualizer.syncTheme());
}

function bindControls() {
  elements.confInput.addEventListener("input", () => {
    elements.confValue.textContent = Number(elements.confInput.value).toFixed(3);
  });
  elements.iouInput.addEventListener("input", () => {
    elements.iouValue.textContent = Number(elements.iouInput.value).toFixed(2);
  });
  elements.trailInput.addEventListener("input", () => {
    elements.trailValue.textContent = elements.trailInput.value;
  });
  elements.videoSelect.addEventListener("change", () => {
    state.externalVideoPath = "";
    elements.pickedVideoLabel.textContent = "选择外部视频";
  });
  elements.pickVideo.addEventListener("click", pickExternalVideo);
  elements.form.addEventListener("submit", submitJob);
  elements.cancelButton.addEventListener("click", cancelJob);
  elements.reloadTracks.addEventListener("click", () => loadTracks());
  elements.trackStride.addEventListener("change", () => loadTracks());
  elements.copyJobId.addEventListener("click", async () => {
    if (!state.currentJobId) return;
    await navigator.clipboard?.writeText(state.currentJobId);
  });
}

async function loadDefaults() {
  try {
    ensureBridge();
    state.defaults = await window.flyGui.getDefaults();
    populateSelect(elements.videoSelect, state.defaults.videos, "未发现项目视频");
    populateSelect(elements.modelSelect, state.defaults.models, "未发现模型");
    populateSelect(elements.gtSelect, [{ name: "不使用 GT", path: "" }, ...state.defaults.groundTruth], "不使用 GT");
    hydrateDefaults(state.defaults.config);
    setStatus("idle", "待命");
  } catch (error) {
    setLog(`加载本地配置失败：${error.message}`);
    setStatus("failed", "配置失败");
  }
}

function populateSelect(select, items, emptyLabel) {
  select.replaceChildren();
  if (!items.length) {
    select.append(new Option(emptyLabel, ""));
    return;
  }
  for (const item of items) {
    select.append(new Option(item.name, item.path));
  }
}

function hydrateDefaults(config) {
  if (!config) return;
  setFormValue("imgsz", config.detection?.imgsz);
  setFormValue("max_det", config.detection?.max_det);
  setFormValue("num_flies", config.track?.num_flies);
  setFormValue("identity_slots", config.track?.identity_slots);
  elements.confInput.value = config.detection?.conf_thres ?? elements.confInput.value;
  elements.iouInput.value = config.detection?.iou_thres ?? elements.iouInput.value;
  elements.trailInput.value = config.render?.trail_len ?? elements.trailInput.value;
  elements.confValue.textContent = Number(elements.confInput.value).toFixed(3);
  elements.iouValue.textContent = Number(elements.iouInput.value).toFixed(2);
  elements.trailValue.textContent = elements.trailInput.value;
}

function setFormValue(name, value) {
  if (value !== null && value !== undefined && elements.form.elements[name]) {
    elements.form.elements[name].value = value;
  }
}

async function pickExternalVideo() {
  try {
    const filePath = await window.flyGui.pickVideo();
    if (!filePath) return;
    state.externalVideoPath = filePath;
    elements.pickedVideoLabel.textContent = `外部：${basename(filePath)}`;
  } catch (error) {
    setLog(`选择视频失败：${error.message}`);
  }
}

async function submitJob(event) {
  event.preventDefault();
  setStatus("running", "启动中");
  document.body.classList.add("is-running");
  elements.startButton.disabled = true;
  elements.cancelButton.disabled = false;
  state.resultLoadedFor = null;
  setLog("");
  renderEmptyResults();

  try {
    const payload = collectPayload();
    const job = await window.flyGui.startJob(payload);
    state.currentJobId = job.job_id;
    renderJob(job);
    schedulePoll(900);
  } catch (error) {
    setStatus("failed", "启动失败");
    setLog(error.message);
    elements.startButton.disabled = false;
    elements.cancelButton.disabled = true;
    document.body.classList.remove("is-running");
  }
}

function collectPayload() {
  const form = elements.form.elements;
  return {
    videoChoice: form.video_choice.value,
    videoPath: state.externalVideoPath,
    modelPath: form.model_path.value,
    gtCsvPath: form.gt_csv_path.value,
    confThres: form.conf_thres.value,
    iouThres: form.iou_thres.value,
    imgsz: form.imgsz.value,
    maxDet: form.max_det.value,
    numFlies: form.num_flies.value,
    identitySlots: form.identity_slots.value,
    maxFrames: form.max_frames.value,
    useCuda: form.use_cuda.checked,
    halfPrecision: form.half_precision.checked,
    saveVideo: form.save_video.checked,
    evaluationEnabled: form.evaluation_enabled.checked,
    trailLen: form.trail_len.value,
    drawLabels: form.draw_labels.checked,
  };
}

async function cancelJob() {
  if (!state.currentJobId) return;
  elements.cancelButton.disabled = true;
  try {
    const payload = await window.flyGui.cancelJob(state.currentJobId);
    renderJob(payload);
  } catch (error) {
    setLog(error.message);
  }
}

function schedulePoll(delay = 1500) {
  window.clearTimeout(state.pollTimer);
  state.pollTimer = window.setTimeout(pollJob, delay);
}

async function pollJob() {
  if (!state.currentJobId) return;
  try {
    const payload = await window.flyGui.getJob(state.currentJobId);
    renderJob(payload);
    await loadLogs();
    if (payload.status === "running" || payload.status === "queued") {
      schedulePoll();
      return;
    }
    elements.startButton.disabled = false;
    elements.cancelButton.disabled = true;
    document.body.classList.remove("is-running");
    if (payload.status === "succeeded") await loadResults();
  } catch (error) {
    setStatus("failed", "轮询失败");
    setLog(error.message);
    elements.startButton.disabled = false;
    elements.cancelButton.disabled = true;
    document.body.classList.remove("is-running");
  }
}

function renderJob(job) {
  state.currentJobId = job.job_id;
  setStatus(job.status, statusLabel(job.status));
  const phase = job.progress?.phase || job.status;
  elements.stageTitle.textContent = `任务 ${job.job_id} · ${phase}`;
  const processed = job.progress?.processedFrame ?? 0;
  const total = job.progress?.totalFrames ?? 0;
  elements.frameCounter.textContent = `${processed} / ${total || "?"}`;
  const percent = job.progress?.percent ?? 0;
  elements.progressBar.style.width = `${Math.round(percent * 100)}%`;
  if (job.error) setLog(job.error);
}

async function loadLogs() {
  if (!state.currentJobId) return;
  const text = await window.flyGui.getLogs(state.currentJobId);
  setLog(text);
}

async function loadResults() {
  if (!state.currentJobId || state.resultLoadedFor === state.currentJobId) return;
  const payload = await window.flyGui.getResults(state.currentJobId);
  state.resultLoadedFor = state.currentJobId;
  renderMetrics(payload.metrics);
  renderEvents(payload.events);
  renderArtifacts(payload.artifacts);
  if (payload.artifacts?.result_video?.exists) {
    elements.resultVideo.src = `${payload.artifacts.result_video.url}?v=${Date.now()}`;
    elements.videoEmpty.style.display = "none";
  }
  await loadTracks();
}

async function loadTracks() {
  if (!state.currentJobId) return;
  const stride = elements.trackStride.value || "5";
  const payload = await window.flyGui.getTracks(state.currentJobId, { stride });
  visualizer.loadTracks(payload);
}

function renderEmptyResults() {
  elements.metricsGrid.replaceChildren(emptyLine("等待指标"));
  elements.eventList.replaceChildren(emptyLine("等待事件"));
  elements.artifactList.replaceChildren(emptyLine("等待输出文件"));
  elements.resultVideo.removeAttribute("src");
  elements.resultVideo.load();
  elements.videoEmpty.style.display = "grid";
}

function renderMetrics(metrics) {
  const keys = [
    ["idf1", "IDF1"],
    ["point_hota", "Point HOTA"],
    ["mota_like", "MOTA-like"],
    ["det_a", "DetA"],
    ["assoc_a", "AssocA"],
    ["num_tracks", "Tracks"],
    ["matched_points", "Matched"],
    ["idsw", "IDSW"],
  ];
  const nodes = keys.map(([key, label]) => {
    const wrapper = document.createElement("div");
    wrapper.className = "metric";
    const value = document.createElement("b");
    value.textContent = formatValue(metrics?.[key]);
    const name = document.createElement("span");
    name.textContent = label;
    wrapper.append(value, name);
    return wrapper;
  });
  elements.metricsGrid.replaceChildren(...nodes);
}

function renderEvents(events = []) {
  if (!events.length) {
    elements.eventList.replaceChildren(emptyLine("没有事件"));
    return;
  }
  const nodes = events.slice(0, 14).map((event) => {
    const row = document.createElement("div");
    row.className = "event-row";
    const title = document.createElement("strong");
    title.textContent = String(event.type || "event");
    const detail = document.createElement("small");
    const frame = event.frame_idx ?? event.frame ?? "-";
    const pair = [event.display_track_a, event.display_track_b].filter((value) => value !== null && value !== undefined).join(" / ");
    detail.textContent = `frame ${frame}${pair ? ` · ID ${pair}` : ""}`;
    row.append(title, detail);
    return row;
  });
  elements.eventList.replaceChildren(...nodes);
}

function renderArtifacts(artifacts = {}) {
  const order = ["result_video", "tracks_csv", "events_csv", "metrics_csv", "detections_csv", "log"];
  const nodes = order.map((kind) => {
    const item = artifacts[kind] || { exists: false, path: "", size: 0 };
    const row = document.createElement("div");
    row.className = `artifact-row${item.exists ? "" : " is-muted"}`;
    const label = document.createElement("span");
    label.textContent = artifactLabel(kind);
    const meta = document.createElement("small");
    meta.textContent = item.exists ? formatBytes(item.size) : "pending";
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = "打开";
    button.disabled = !item.exists;
    button.addEventListener("click", () => window.flyGui.openPath(item.path));
    row.append(label, meta, button);
    return row;
  });
  elements.artifactList.replaceChildren(...nodes);
}

function emptyLine(text) {
  const node = document.createElement("div");
  node.className = "event-row";
  node.textContent = text;
  return node;
}

function setStatus(status, label) {
  elements.statusPill.dataset.status = status;
  elements.statusText.textContent = label;
}

function setLog(text) {
  elements.logOutput.textContent = text || "";
  elements.logOutput.scrollTop = elements.logOutput.scrollHeight;
}

function statusLabel(status) {
  return {
    idle: "待命",
    queued: "排队",
    running: "运行中",
    succeeded: "完成",
    failed: "失败",
    canceled: "已停止",
  }[status] || status;
}

function artifactLabel(kind) {
  return {
    result_video: "结果视频",
    tracks_csv: "轨迹 CSV",
    events_csv: "事件 CSV",
    metrics_csv: "指标 CSV",
    detections_csv: "检测 CSV",
    log: "日志",
  }[kind] || kind;
}

function formatValue(value) {
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value !== "number") return String(value);
  if (Math.abs(value) >= 100) return Math.round(value).toLocaleString();
  if (Math.abs(value) >= 10) return value.toFixed(1);
  return value.toFixed(3);
}

function formatBytes(bytes) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let size = bytes;
  let idx = 0;
  while (size >= 1024 && idx < units.length - 1) {
    size /= 1024;
    idx += 1;
  }
  return `${size.toFixed(idx === 0 ? 0 : 1)} ${units[idx]}`;
}

function basename(filePath) {
  return String(filePath).split(/[\\/]/).pop() || filePath;
}

function ensureBridge() {
  if (!window.flyGui) {
    throw new Error("请通过 Electron 启动本地 GUI，而不是直接打开 HTML 文件。");
  }
}
