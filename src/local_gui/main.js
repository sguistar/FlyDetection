const { app, BrowserWindow, dialog, ipcMain, shell } = require("electron");
const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");
const { pathToFileURL } = require("url");

const repoRoot = path.resolve(__dirname, "..", "..");
const srcRoot = path.join(repoRoot, "src");
const guiRoot = path.join(repoRoot, "outputs", "local_gui");
const runsRoot = path.join(guiRoot, "runs");
const uploadsRoot = path.join(guiRoot, "external_inputs");

const jobs = new Map();

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function pythonExecutable() {
  const localPython = path.join(repoRoot, ".venv", "Scripts", "python.exe");
  return fs.existsSync(localPython) ? localPython : "python";
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1440,
    height: 960,
    minWidth: 1120,
    minHeight: 720,
    title: "Fly MOT Local GUI",
    backgroundColor: "#0c1110",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });
  win.loadFile(path.join(__dirname, "index.html"));
}

app.whenReady().then(() => {
  ensureDir(runsRoot);
  ensureDir(uploadsRoot);
  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

ipcMain.handle("gui:get-defaults", () => {
  return {
    config: {
      detection: { conf_thres: 0.05, iou_thres: 0.8, imgsz: 2560, max_det: 32 },
      track: { num_flies: 6, identity_slots: 6 },
      render: { trail_len: 24 },
    },
    videos: discoverFiles([repoRoot], [".mp4", ".avi", ".mkv", ".mov"]),
    models: discoverFiles([repoRoot, path.join(repoRoot, "outputs", "models")], [".pt"]).filter(
      (item) => !item.name.toLowerCase().startsWith("appearance_encoder"),
    ),
    groundTruth: discoverFiles([path.join(repoRoot, "coords")], [".csv"]),
  };
});

ipcMain.handle("gui:pick-video", async (event) => {
  const result = await dialog.showOpenDialog(BrowserWindow.fromWebContents(event.sender), {
    title: "选择视频",
    properties: ["openFile"],
    filters: [{ name: "Videos", extensions: ["mp4", "avi", "mkv", "mov"] }],
  });
  if (result.canceled || result.filePaths.length === 0) return null;
  return result.filePaths[0];
});

ipcMain.handle("gui:start-job", async (_event, payload) => {
  const jobId = `${Math.floor(Date.now() / 1000)}-${Math.random().toString(16).slice(2, 10)}`;
  const jobDir = path.join(runsRoot, jobId);
  ensureDir(jobDir);

  const videoPath = resolveVideoPath(payload.videoPath || payload.videoChoice);
  const modelPath = resolveProjectFile(payload.modelPath, [".pt"]);
  const gtPath = payload.gtCsvPath ? resolveProjectFile(payload.gtCsvPath, [".csv"]) : "";
  const outputRoot = path.join(jobDir, "outputs");
  const totalFrames = readVideoFrameCount(videoPath);
  const maxFrames = payload.maxFrames ? Number(payload.maxFrames) : null;
  const expectedFrames = totalFrames && maxFrames ? Math.min(totalFrames, maxFrames) : totalFrames;

  const spec = {
    job_id: jobId,
    job_dir: jobDir,
    output_root: outputRoot,
    video_path: videoPath,
    model_path: modelPath,
    gt_csv_path: gtPath,
    overrides: cleanOverrides(payload),
  };
  const specPath = path.join(jobDir, "spec.json");
  fs.writeFileSync(specPath, JSON.stringify(spec, null, 2), "utf8");

  const stdoutPath = path.join(jobDir, "runner.stdout.log");
  const stdout = fs.openSync(stdoutPath, "a");
  const child = spawn(pythonExecutable(), ["-m", "src.web.runner", specPath], {
    cwd: repoRoot,
    env: {
      ...process.env,
      PYTHONPATH: [repoRoot, srcRoot, process.env.PYTHONPATH || ""].filter(Boolean).join(path.delimiter),
    },
    stdio: ["ignore", stdout, stdout],
    windowsHide: true,
  });
  fs.closeSync(stdout);

  const record = {
    job_id: jobId,
    job_dir: jobDir,
    output_root: outputRoot,
    video_path: videoPath,
    model_path: modelPath,
    created_at: Date.now() / 1000,
    status: "running",
    max_frames: maxFrames,
    total_frames: expectedFrames,
    error: null,
    return_code: null,
  };
  jobs.set(jobId, { record, child });
  writeJobRecord(record);
  child.on("error", (error) => {
    record.status = "failed";
    record.error = error.message;
    writeJobRecord(record);
  });
  child.on("exit", (code) => {
    record.return_code = code;
    if (record.status !== "canceled") {
      record.status = code === 0 ? "succeeded" : "failed";
      if (code !== 0) record.error = readError(jobDir) || tailText(stdoutPath, 40);
    }
    writeJobRecord(record);
  });
  return jobPayload(record);
});

ipcMain.handle("gui:get-job", (_event, jobId) => {
  const record = requireJob(jobId).record;
  refreshJob(record);
  return jobPayload(record);
});

ipcMain.handle("gui:cancel-job", (_event, jobId) => {
  const job = requireJob(jobId);
  if (!["succeeded", "failed", "canceled"].includes(job.record.status)) {
    job.record.status = "canceled";
    try {
      job.child?.kill();
    } catch {
      // best effort
    }
    writeJobRecord(job.record);
  }
  return jobPayload(job.record);
});

ipcMain.handle("gui:get-logs", (_event, jobId) => {
  const record = requireJob(jobId).record;
  return tailText(path.join(record.output_root, "logs", "run.log"), 180) || tailText(path.join(record.job_dir, "runner.stdout.log"), 180);
});

ipcMain.handle("gui:get-results", async (_event, jobId) => {
  const record = requireJob(jobId).record;
  const outputRoot = record.output_root;
  const resultVideo = path.join(outputRoot, "videos", "result.mp4");
  const browserVideo = fs.existsSync(resultVideo) ? await ensureBrowserVideo(resultVideo) : null;
  return {
    job: jobPayload(record),
    metrics: readCsv(path.join(outputRoot, "csv", "metrics.csv"))[0] || {},
    events: readCsv(path.join(outputRoot, "csv", "events.csv")).slice(0, 300),
    artifacts: artifactPayload(record, browserVideo),
  };
});

ipcMain.handle("gui:get-tracks", (_event, jobId, options = {}) => {
  const record = requireJob(jobId).record;
  return parseTracksCsv(path.join(record.output_root, "csv", "tracks.csv"), Number(options.stride || 1));
});

ipcMain.handle("gui:open-path", async (_event, filePath) => {
  if (!filePath || !fs.existsSync(filePath)) return false;
  const result = await shell.openPath(filePath);
  return result === "";
});

function discoverFiles(roots, extensions) {
  const seen = new Set();
  const files = [];
  for (const root of roots) {
    if (!fs.existsSync(root)) continue;
    for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
      if (!entry.isFile()) continue;
      const fullPath = path.resolve(root, entry.name);
      const ext = path.extname(entry.name).toLowerCase();
      if (!extensions.includes(ext) || seen.has(fullPath)) continue;
      if (isInside(fullPath, guiRoot)) continue;
      seen.add(fullPath);
      const stat = fs.statSync(fullPath);
      files.push({ name: entry.name, path: relativeOrAbsolute(fullPath), absolutePath: fullPath, size: stat.size, modified: stat.mtimeMs / 1000 });
    }
  }
  return files.sort((a, b) => a.name.localeCompare(b.name));
}

function resolveVideoPath(value) {
  if (!value) throw new Error("No video selected");
  if (path.isAbsolute(value)) return assertExistingFile(value, [".mp4", ".avi", ".mkv", ".mov"]);
  return resolveProjectFile(value, [".mp4", ".avi", ".mkv", ".mov"]);
}

function resolveProjectFile(value, extensions) {
  const candidate = path.resolve(repoRoot, value || "");
  if (!isInside(candidate, repoRoot)) throw new Error("Path must stay inside project");
  return assertExistingFile(candidate, extensions);
}

function assertExistingFile(filePath, extensions) {
  const resolved = path.resolve(filePath);
  if (!fs.existsSync(resolved) || !fs.statSync(resolved).isFile()) throw new Error(`File not found: ${filePath}`);
  if (!extensions.includes(path.extname(resolved).toLowerCase())) throw new Error(`Unsupported file type: ${filePath}`);
  return resolved;
}

function cleanOverrides(payload) {
  const overrides = {
    "detection.conf_thres": Number(payload.confThres ?? 0.05),
    "detection.iou_thres": Number(payload.iouThres ?? 0.8),
    "detection.imgsz": Number(payload.imgsz ?? 2560),
    "detection.max_det": Number(payload.maxDet ?? 32),
    "track.num_flies": Number(payload.numFlies ?? 6),
    "track.identity_slots": Number(payload.identitySlots ?? 6),
    "runtime.use_cuda": Boolean(payload.useCuda),
    "runtime.half_precision": Boolean(payload.halfPrecision),
    "runtime.save_video": Boolean(payload.saveVideo),
    "evaluation.enabled": Boolean(payload.evaluationEnabled),
    "render.trail_len": Number(payload.trailLen ?? 24),
    "render.draw_labels": Boolean(payload.drawLabels),
  };
  if (payload.maxFrames) overrides["runtime.max_frames"] = Number(payload.maxFrames);
  return overrides;
}

function readVideoFrameCount(videoPath) {
  const script = "import cv2,sys; cap=cv2.VideoCapture(sys.argv[1]); print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)); cap.release()";
  try {
    const result = spawnSyncText(pythonExecutable(), ["-c", script, videoPath]);
    const frames = Number(result.trim());
    return Number.isFinite(frames) && frames > 0 ? frames : null;
  } catch {
    return null;
  }
}

function refreshJob(record) {
  if (["succeeded", "failed", "canceled"].includes(record.status)) return;
  const summaryPath = path.join(record.job_dir, "summary.json");
  const errorPath = path.join(record.job_dir, "error.json");
  if (fs.existsSync(summaryPath)) {
    record.status = "succeeded";
    record.return_code = 0;
    writeJobRecord(record);
  } else if (fs.existsSync(errorPath)) {
    record.status = "failed";
    record.return_code = 1;
    record.error = readError(record.job_dir);
    writeJobRecord(record);
  }
}

function jobPayload(record) {
  const progress = parseLogProgress(path.join(record.output_root, "logs", "run.log"));
  const processed = progress.processed_frame;
  const total = record.total_frames;
  const percent = processed !== null && total ? Math.min(Math.max((processed + 1) / Math.max(total, 1), 0), 1) : null;
  return {
    ...record,
    progress: {
      processedFrame: processed,
      totalFrames: total,
      percent,
      phase: record.status === "running" ? (percent !== null && percent >= 0.98 ? "exporting" : "tracking") : record.status,
      activeTracks: progress.active_tracks,
      numDetections: progress.num_detections,
    },
    artifacts: artifactPayload(record, null),
  };
}

function artifactPayload(record, browserVideo) {
  const outputRoot = record.output_root;
  const items = {
    result_video: browserVideo || path.join(outputRoot, "videos", "result.browser.mp4"),
    tracks_csv: path.join(outputRoot, "csv", "tracks.csv"),
    events_csv: path.join(outputRoot, "csv", "events.csv"),
    metrics_csv: path.join(outputRoot, "csv", "metrics.csv"),
    detections_csv: path.join(outputRoot, "csv", "detections.csv"),
    log: path.join(outputRoot, "logs", "run.log"),
  };
  const payload = {};
  for (const [kind, filePath] of Object.entries(items)) {
    const exists = fs.existsSync(filePath);
    payload[kind] = {
      exists,
      path: filePath,
      url: exists ? pathToFileURL(filePath).href : "",
      size: exists ? fs.statSync(filePath).size : 0,
    };
  }
  return payload;
}

async function ensureBrowserVideo(videoPath) {
  const browserPath = videoPath.replace(/\.mp4$/i, ".browser.mp4");
  if (fs.existsSync(browserPath) && fs.statSync(browserPath).mtimeMs >= fs.statSync(videoPath).mtimeMs) return browserPath;
  const result = await spawnTextAsync(pythonExecutable(), ["-m", "web.video", videoPath], {
    cwd: repoRoot,
    env: { ...process.env, PYTHONPATH: [repoRoot, srcRoot, process.env.PYTHONPATH || ""].filter(Boolean).join(path.delimiter) },
  });
  return result.trim() || browserPath;
}

function requireJob(jobId) {
  const existing = jobs.get(jobId);
  if (existing) return existing;
  const record = loadJobRecord(jobId);
  if (!record) throw new Error(`Job not found: ${jobId}`);
  const job = { record, child: null };
  jobs.set(jobId, job);
  return job;
}

function loadJobRecord(jobId) {
  if (!/^[0-9]+-[a-f0-9]{8}$/.test(jobId)) return null;
  const jobDir = path.join(runsRoot, jobId);
  const jobPath = path.join(jobDir, "job.json");
  if (fs.existsSync(jobPath)) return JSON.parse(fs.readFileSync(jobPath, "utf8"));
  return null;
}

function writeJobRecord(record) {
  ensureDir(record.job_dir);
  fs.writeFileSync(path.join(record.job_dir, "job.json"), JSON.stringify(record, null, 2), "utf8");
}

function readError(jobDir) {
  const errorPath = path.join(jobDir, "error.json");
  if (!fs.existsSync(errorPath)) return null;
  try {
    return JSON.parse(fs.readFileSync(errorPath, "utf8")).error || null;
  } catch {
    return null;
  }
}

function parseLogProgress(logPath) {
  const progress = { processed_frame: null, active_tracks: null, num_detections: null, finished: false };
  if (!fs.existsSync(logPath)) return progress;
  for (const line of fs.readFileSync(logPath, "utf8").split(/\r?\n/)) {
    const payload = jsonPayloadFromLog(line);
    if (line.includes("Processed frame") && payload) {
      progress.processed_frame = payload.frame_idx ?? progress.processed_frame;
      progress.active_tracks = payload.active_tracks ?? progress.active_tracks;
      progress.num_detections = payload.num_detections ?? progress.num_detections;
    } else if (line.includes("Finished MOT pipeline")) {
      progress.finished = true;
    }
  }
  return progress;
}

function jsonPayloadFromLog(line) {
  const index = line.indexOf(" | {");
  if (index < 0) return null;
  try {
    return JSON.parse(line.slice(index + 3));
  } catch {
    return null;
  }
}

function tailText(filePath, lines) {
  if (!fs.existsSync(filePath)) return "";
  return fs.readFileSync(filePath, "utf8").split(/\r?\n/).slice(-lines).join("\n");
}

function readCsv(filePath) {
  if (!fs.existsSync(filePath) || fs.statSync(filePath).size === 0) return [];
  const text = fs.readFileSync(filePath, "utf8").trim();
  if (!text) return [];
  const [headerLine, ...lines] = text.split(/\r?\n/);
  const headers = splitCsvLine(headerLine);
  return lines.filter(Boolean).map((line) => {
    const values = splitCsvLine(line);
    const row = {};
    headers.forEach((key, index) => {
      row[key] = coerce(values[index] ?? "");
    });
    return row;
  });
}

function splitCsvLine(line) {
  const out = [];
  let current = "";
  let quoted = false;
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (char === '"' && line[i + 1] === '"') {
      current += '"';
      i += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      out.push(current);
      current = "";
    } else {
      current += char;
    }
  }
  out.push(current);
  return out;
}

function coerce(value) {
  const text = String(value ?? "").trim();
  if (text === "") return null;
  if (/^-?\d+$/.test(text)) return Number(text);
  if (/^-?\d+\.\d+(e-?\d+)?$/i.test(text)) return Number(text);
  return text;
}

function parseTracksCsv(filePath, stride) {
  const tracks = new Map();
  const bounds = { minX: null, maxX: null, minY: null, maxY: null, minFrame: null, maxFrame: null };
  for (const row of readCsv(filePath)) {
    const frame = Number(row.frame);
    if (frame % Math.max(stride, 1) !== 0) continue;
    const displayId = Number(row.display_id ?? row.identity_slot ?? row.track_id);
    const x = Number(row.x);
    const y = Number(row.y);
    const point = {
      f: frame,
      x,
      y,
      trackId: Number(row.track_id),
      displayId,
      state: row.state || "",
      conf: Number(row.conf || 0),
      interpolated: Boolean(Number(row.interpolated || 0)),
    };
    if (!tracks.has(displayId)) tracks.set(displayId, []);
    tracks.get(displayId).push(point);
    updateBounds(bounds, x, y, frame);
  }
  return {
    tracks: [...tracks.entries()].sort((a, b) => a[0] - b[0]).map(([id, points]) => ({ id, points: points.sort((a, b) => a.f - b.f) })),
    bounds,
    frameCount: bounds.minFrame === null ? 0 : bounds.maxFrame - bounds.minFrame + 1,
  };
}

function updateBounds(bounds, x, y, frame) {
  bounds.minX = bounds.minX === null ? x : Math.min(bounds.minX, x);
  bounds.maxX = bounds.maxX === null ? x : Math.max(bounds.maxX, x);
  bounds.minY = bounds.minY === null ? y : Math.min(bounds.minY, y);
  bounds.maxY = bounds.maxY === null ? y : Math.max(bounds.maxY, y);
  bounds.minFrame = bounds.minFrame === null ? frame : Math.min(bounds.minFrame, frame);
  bounds.maxFrame = bounds.maxFrame === null ? frame : Math.max(bounds.maxFrame, frame);
}

function spawnSyncText(command, args) {
  const proc = require("child_process").spawnSync(command, args, { cwd: repoRoot, encoding: "utf8", windowsHide: true });
  if (proc.error) throw proc.error;
  if (proc.status !== 0) throw new Error(proc.stderr || `Command failed: ${command}`);
  return proc.stdout;
}

function spawnTextAsync(command, args, options) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { ...options, windowsHide: true });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("error", reject);
    child.on("exit", (code) => {
      if (code === 0) resolve(stdout);
      else reject(new Error(stderr || `Command failed with code ${code}`));
    });
  });
}

function isInside(child, parent) {
  const relative = path.relative(path.resolve(parent), path.resolve(child));
  return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
}

function relativeOrAbsolute(filePath) {
  return isInside(filePath, repoRoot) ? path.relative(repoRoot, filePath).replace(/\\/g, "/") : filePath;
}
