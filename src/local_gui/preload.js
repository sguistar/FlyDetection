const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("flyGui", {
  getDefaults: () => ipcRenderer.invoke("gui:get-defaults"),
  pickVideo: () => ipcRenderer.invoke("gui:pick-video"),
  startJob: (payload) => ipcRenderer.invoke("gui:start-job", payload),
  getJob: (jobId) => ipcRenderer.invoke("gui:get-job", jobId),
  cancelJob: (jobId) => ipcRenderer.invoke("gui:cancel-job", jobId),
  getLogs: (jobId) => ipcRenderer.invoke("gui:get-logs", jobId),
  getResults: (jobId) => ipcRenderer.invoke("gui:get-results", jobId),
  getTracks: (jobId, options) => ipcRenderer.invoke("gui:get-tracks", jobId, options),
  openPath: (filePath) => ipcRenderer.invoke("gui:open-path", filePath),
});
