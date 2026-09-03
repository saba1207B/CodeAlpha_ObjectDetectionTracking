/**
 * CodeAlpha Object Detection & Tracking — Laptop Dashboard Engine
 * Handles real-time polling, frame rendering, settings controls,
 * tracking analytics, and export actions.
 */

// Server configuration - dynamically uses current origin or defaults to port 5000
const API_BASE = window.location.port ? window.location.origin : 'http://127.0.0.1:5000';

// Global state
const state = {
  serverConnected: false,
  phoneConnected: false,
  detectionActive: true,
  viewMode: 'processed', // 'processed', 'split', 'raw'
  isRecording: false,
  selectedClasses: ['all'],
  lastFrameTimestamp: 0,
  pollTimer: null,
  framePollTimer: null,
  stats: {},
  detections: [],
};

// DOM Element Selectors
const el = {
  // Badges
  phoneStatusBadge: document.getElementById('phoneStatusBadge'),
  phoneStatusText: document.getElementById('phoneStatusText'),
  serverStatusBadge: document.getElementById('serverStatusBadge'),
  serverStatusText: document.getElementById('serverStatusText'),
  detectionStatusBadge: document.getElementById('detectionStatusBadge'),
  detectionStatusText: document.getElementById('detectionStatusText'),

  // Banner & IP
  apkUrlCode: document.getElementById('apkUrlCode'),
  waitingUrlCode: document.getElementById('waitingUrlCode'),
  copyUrlBtn: document.getElementById('copyUrlBtn'),
  laptopIpText: document.getElementById('laptopIpText'),
  clientIpText: document.getElementById('clientIpText'),
  lastFrameTimeText: document.getElementById('lastFrameTimeText'),

  // Metrics
  metricFps: document.getElementById('metricFps'),
  metricLatency: document.getElementById('metricLatency'),
  metricResolution: document.getElementById('metricResolution'),
  metricFrameCount: document.getElementById('metricFrameCount'),
  metricActiveTracks: document.getElementById('metricActiveTracks'),
  metricTotalUniqueIds: document.getElementById('metricTotalUniqueIds'),

  // Viewport
  videoContainer: document.getElementById('videoContainer'),
  waitingPlaceholder: document.getElementById('waitingPlaceholder'),
  processedViewWrap: document.getElementById('processedViewWrap'),
  rawViewWrap: document.getElementById('rawViewWrap'),
  processedFrameImg: document.getElementById('processedFrameImg'),
  rawFrameImg: document.getElementById('rawFrameImg'),
  overlayTrackerName: document.getElementById('overlayTrackerName'),
  overlayFps: document.getElementById('overlayFps'),
  streamStatusIndicator: document.getElementById('streamStatusIndicator'),
  streamStatusText: document.getElementById('streamStatusText'),
  recordingStatusText: document.getElementById('recordingStatusText'),
  inferenceTimeBadge: document.getElementById('inferenceTimeBadge'),

  // View Mode Tabs
  tabProcessed: document.getElementById('tabProcessed'),
  tabSplit: document.getElementById('tabSplit'),
  tabRaw: document.getElementById('tabRaw'),

  // Controls
  startDetectionBtn: document.getElementById('startDetectionBtn'),
  stopDetectionBtn: document.getElementById('stopDetectionBtn'),
  clearResultsBtn: document.getElementById('clearResultsBtn'),
  modelSelect: document.getElementById('modelSelect'),
  trackerSelect: document.getElementById('trackerSelect'),
  confRange: document.getElementById('confRange'),
  confValue: document.getElementById('confValue'),
  iouRange: document.getElementById('iouRange'),
  iouValue: document.getElementById('iouValue'),
  toggleTrails: document.getElementById('toggleTrails'),
  toggleHud: document.getElementById('toggleHud'),
  classFilterSummary: document.getElementById('classFilterSummary'),
  applySettingsBtn: document.getElementById('applySettingsBtn'),

  // Table
  trackCountBadge: document.getElementById('trackCountBadge'),
  tracksTableBody: document.getElementById('tracksTableBody'),
  refreshTracksBtn: document.getElementById('refreshTracksBtn'),

  // Export
  exportScreenshotBtn: document.getElementById('exportScreenshotBtn'),
  quickScreenshotBtn: document.getElementById('quickScreenshotBtn'),
  toggleRecordBtn: document.getElementById('toggleRecordBtn'),
  quickRecordBtn: document.getElementById('quickRecordBtn'),
  toggleRecordText: document.getElementById('toggleRecordText'),
  exportAlert: document.getElementById('exportAlert'),
  exportAlertMsg: document.getElementById('exportAlertMsg'),
  fullscreenBtn: document.getElementById('fullscreenBtn'),

  // Log
  logToggleHeader: document.getElementById('logToggleHeader'),
  logToggleIcon: document.getElementById('logToggleIcon'),
  logBody: document.getElementById('logBody'),
  logEntries: document.getElementById('logEntries'),
  toastContainer: document.getElementById('toastContainer'),
};

/* ==========================================================================
   Initialization & Event Listeners
   ========================================================================== */
function init() {
  bindEvents();
  setupClassChips();
  addLog('Dashboard loaded. Connecting to backend at ' + API_BASE + '...', 'info');

  // Fetch initial status and settings
  fetchStatus();
  fetchSettings();

  // Start polling loops
  startPolling();
}

function bindEvents() {
  // Sliders input events
  el.confRange.addEventListener('input', (e) => {
    el.confValue.textContent = parseFloat(e.target.value).toFixed(2);
  });

  el.iouRange.addEventListener('input', (e) => {
    el.iouValue.textContent = parseFloat(e.target.value).toFixed(2);
  });

  // Settings submit
  el.applySettingsBtn.addEventListener('click', applySettings);

  // Pipeline control buttons
  el.startDetectionBtn.addEventListener('click', startDetection);
  el.stopDetectionBtn.addEventListener('click', stopDetection);
  el.clearResultsBtn.addEventListener('click', clearResults);

  // Copy APK URL
  el.copyUrlBtn.addEventListener('click', copyApkUrl);

  // View Mode Tabs
  el.tabProcessed.addEventListener('click', () => setViewMode('processed'));
  el.tabSplit.addEventListener('click', () => setViewMode('split'));
  el.tabRaw.addEventListener('click', () => setViewMode('raw'));

  // Export & Media
  el.exportScreenshotBtn.addEventListener('click', captureScreenshot);
  el.quickScreenshotBtn.addEventListener('click', captureScreenshot);
  el.toggleRecordBtn.addEventListener('click', toggleRecording);
  el.quickRecordBtn.addEventListener('click', toggleRecording);
  el.fullscreenBtn.addEventListener('click', toggleFullscreen);

  // Refresh tracks button
  el.refreshTracksBtn.addEventListener('click', fetchDetections);

  // Collapsible Log
  el.logToggleHeader.addEventListener('click', () => {
    el.logBody.classList.toggle('collapsed');
    el.logToggleIcon.textContent = el.logBody.classList.contains('collapsed') ? '▸' : '▾';
  });

  // Keyboard Shortcuts
  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    if (e.key === 's' || e.key === 'S') {
      captureScreenshot();
    } else if (e.key === 't' || e.key === 'T') {
      el.toggleTrails.checked = !el.toggleTrails.checked;
      applySettings();
    } else if (e.key === 'h' || e.key === 'H') {
      el.toggleHud.checked = !el.toggleHud.checked;
      applySettings();
    }
  });
}

function setupClassChips() {
  const chips = document.querySelectorAll('.class-chip');
  chips.forEach((chip) => {
    chip.addEventListener('click', () => {
      const cls = chip.getAttribute('data-class');
      if (cls === 'all') {
        chips.forEach((c) => c.classList.remove('active'));
        chip.classList.add('active');
        state.selectedClasses = ['all'];
      } else {
        document.querySelector('.class-chip[data-class="all"]').classList.remove('active');
        chip.classList.toggle('active');

        // Collect all active chips
        const active = Array.from(document.querySelectorAll('.class-chip.active'))
          .map((c) => c.getAttribute('data-class'))
          .filter((c) => c !== 'all');

        if (active.length === 0) {
          document.querySelector('.class-chip[data-class="all"]').classList.add('active');
          state.selectedClasses = ['all'];
        } else {
          state.selectedClasses = active.map(Number);
        }
      }

      // Update summary text
      if (state.selectedClasses.includes('all')) {
        el.classFilterSummary.textContent = 'All 80 classes';
      } else {
        el.classFilterSummary.textContent = `${state.selectedClasses.length} selected`;
      }
    });
  });
}

/* ==========================================================================
   Polling & Frame Streaming
   ========================================================================== */
function startPolling() {
  // Status check every 1000ms
  state.pollTimer = setInterval(fetchStatus, 1000);

  // Detection and frame polling (smooth ~15-20 FPS)
  state.framePollTimer = setInterval(updateFramesAndDetections, 75);
}

async function fetchStatus() {
  try {
    const res = await fetch(`${API_BASE}/status`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    state.stats = data;
    state.serverConnected = true;
    updateServerBadge('connected', 'Backend Running');

    // Update IP and URL display
    if (data.apk_url) {
      el.apkUrlCode.textContent = data.apk_url;
      el.waitingUrlCode.textContent = data.apk_url;
    }
    if (data.laptop_ips && data.laptop_ips.length > 0) {
      el.laptopIpText.textContent = data.laptop_ips.join(', ');
    }
    if (data.client_address && data.client_address !== 'None') {
      el.clientIpText.textContent = data.client_address;
    }

    // Phone connection status
    const isReceiving = data.is_receiving || (data.seconds_since_last_frame !== null && data.seconds_since_last_frame < 3.5);
    state.phoneConnected = isReceiving;

    if (isReceiving) {
      updatePhoneBadge('connected', `Streaming (${data.client_address || 'Phone'})`);
      el.streamStatusIndicator.className = 'indicator-dot dot-emerald';
      el.streamStatusText.textContent = `Streaming active from ${data.client_address}`;
      el.waitingPlaceholder.classList.add('hidden');
    } else if (data.frame_count > 0) {
      updatePhoneBadge('disconnected', 'Phone Paused');
      el.streamStatusIndicator.className = 'indicator-dot dot-amber';
      el.streamStatusText.textContent = `Phone stream paused (${data.seconds_since_last_frame}s ago)`;
    } else {
      updatePhoneBadge('waiting', 'Waiting for Phone');
      el.streamStatusIndicator.className = 'indicator-dot dot-amber';
      el.streamStatusText.textContent = 'Waiting for phone camera connection...';
      el.waitingPlaceholder.classList.remove('hidden');
    }

    // Update Metrics
    el.metricFps.textContent = (data.current_fps || 0.0).toFixed(1);
    el.overlayFps.textContent = `${(data.current_fps || 0.0).toFixed(1)} FPS`;
    el.metricResolution.textContent = data.resolution || '-- x --';
    el.metricFrameCount.textContent = (data.frame_count || 0).toLocaleString();

    // Latency
    if (data.seconds_since_last_frame !== null && data.seconds_since_last_frame !== undefined) {
      const ms = Math.round(data.seconds_since_last_frame * 1000);
      el.metricLatency.textContent = isReceiving ? ms : '--';
      el.lastFrameTimeText.textContent = `${data.seconds_since_last_frame}s ago`;
    }

    // Recording status
    if (data.recording !== undefined) {
      setRecordingState(data.recording);
    }

    // Detection status
    if (data.detection_active !== undefined) {
      state.detectionActive = data.detection_active;
      if (data.detection_active) {
        el.detectionStatusBadge.className = 'status-pill active';
        el.detectionStatusText.textContent = 'Detection Active';
      } else {
        el.detectionStatusBadge.className = 'status-pill disconnected';
        el.detectionStatusText.textContent = 'Detection Paused';
      }
    }

  } catch (err) {
    state.serverConnected = false;
    state.phoneConnected = false;
    updateServerBadge('error', 'Server Offline');
    updatePhoneBadge('disconnected', 'Disconnected');
    el.streamStatusIndicator.className = 'indicator-dot dot-rose';
    el.streamStatusText.textContent = 'Unable to reach backend server on ' + API_BASE;
  }
}

let isFrameFetching = false;
async function updateFramesAndDetections() {
  if (!state.serverConnected || !state.phoneConnected) return;
  if (isFrameFetching) return;

  isFrameFetching = true;
  const timestamp = Date.now();

  try {
    // 1. Update Processed Frame if active
    if (state.viewMode === 'processed' || state.viewMode === 'split') {
      const frameUrl = `${API_BASE}/processed_frame?t=${timestamp}`;
      el.processedFrameImg.src = frameUrl;
    }

    // 2. Update Raw Frame if active
    if (state.viewMode === 'raw' || state.viewMode === 'split') {
      const rawUrl = `${API_BASE}/latest_frame?t=${timestamp}`;
      el.rawFrameImg.src = rawUrl;
    }

    // 3. Fetch current detections & tracking data
    await fetchDetections();
  } catch (err) {
    // Silent catch for stream frames
  } finally {
    isFrameFetching = false;
  }
}

async function fetchDetections() {
  try {
    const res = await fetch(`${API_BASE}/detections`);
    if (!res.ok) return;
    const data = await res.json();

    state.detections = data.active_tracks || [];

    // Update Track metrics
    el.metricActiveTracks.textContent = data.active_count || 0;
    el.metricTotalUniqueIds.textContent = data.total_unique_tracks || 0;
    el.trackCountBadge.textContent = `${data.active_count || 0} Active`;

    if (data.inference_time_ms) {
      el.inferenceTimeBadge.textContent = `Inference: ${data.inference_time_ms.toFixed(1)}ms`;
    }

    renderTracksTable(data.active_tracks || []);
  } catch (e) {
    // console.warn(e);
  }
}

/* ==========================================================================
   UI View Mode Switching
   ========================================================================== */
function setViewMode(mode) {
  state.viewMode = mode;

  // Update tabs
  [el.tabProcessed, el.tabSplit, el.tabRaw].forEach((btn) => btn.classList.remove('active'));
  if (mode === 'processed') el.tabProcessed.classList.add('active');
  if (mode === 'split') el.tabSplit.classList.add('active');
  if (mode === 'raw') el.tabRaw.classList.add('active');

  // Update containers
  el.videoContainer.className = `video-container mode-${mode}`;

  if (mode === 'processed') {
    el.processedViewWrap.className = 'frame-wrap active';
    el.rawViewWrap.className = 'frame-wrap hidden';
  } else if (mode === 'raw') {
    el.processedViewWrap.className = 'frame-wrap hidden';
    el.rawViewWrap.className = 'frame-wrap active';
  } else if (mode === 'split') {
    el.processedViewWrap.className = 'frame-wrap active';
    el.rawViewWrap.className = 'frame-wrap active';
  }
}

/* ==========================================================================
   Table Rendering
   ========================================================================== */
function renderTracksTable(tracks) {
  if (!tracks || tracks.length === 0) {
    el.tracksTableBody.innerHTML = `
      <tr class="empty-row">
        <td colspan="5">No active objects tracked in the current frame</td>
      </tr>
    `;
    return;
  }

  const rowsHtml = tracks.map((obj) => {
    const idBadge = `<span class="id-badge">#${obj.track_id !== null ? obj.track_id : '?'}</span>`;
    const confBadge = `<span class="conf-pill">${(obj.conf * 100).toFixed(0)}%</span>`;
    const centroid = obj.centroid ? `(${obj.centroid[0]}, ${obj.centroid[1]})` : '--';
    const lastSeen = obj.last_seen_time || 'live';

    return `
      <tr>
        <td>${idBadge}</td>
        <td><strong>${escapeHtml(obj.class_name)}</strong></td>
        <td>${confBadge}</td>
        <td><code>${centroid}</code></td>
        <td>${escapeHtml(lastSeen)}</td>
      </tr>
    `;
  }).join('');

  el.tracksTableBody.innerHTML = rowsHtml;
}

/* ==========================================================================
   Detection Controls & Settings Actions
   ========================================================================== */
async function startDetection() {
  try {
    const res = await fetch(`${API_BASE}/start-detection`, { method: 'POST' });
    const data = await res.json();
    state.detectionActive = true;
    showToast('Detection pipeline started', 'success');
    addLog('[DETECTION] Real-time detection pipeline started.', 'success');
    fetchStatus();
  } catch (err) {
    showToast('Failed to start detection', 'error');
  }
}

async function stopDetection() {
  try {
    const res = await fetch(`${API_BASE}/stop-detection`, { method: 'POST' });
    const data = await res.json();
    state.detectionActive = false;
    showToast('Detection pipeline stopped', 'info');
    addLog('[DETECTION] Detection pipeline paused.', 'info');
    fetchStatus();
  } catch (err) {
    showToast('Failed to stop detection', 'error');
  }
}

async function clearResults() {
  try {
    const res = await fetch(`${API_BASE}/clear-results`, { method: 'POST' });
    const data = await res.json();
    showToast('Tracking results & trails reset', 'info');
    addLog('[TRACKER] Cleared unique track IDs and motion trajectory history.', 'info');
    fetchDetections();
  } catch (err) {
    showToast('Failed to clear results', 'error');
  }
}

async function fetchSettings() {
  try {
    const res = await fetch(`${API_BASE}/settings`);
    if (!res.ok) return;
    const settings = await res.json();

    if (settings.model) el.modelSelect.value = settings.model;
    if (settings.tracker) {
      el.trackerSelect.value = settings.tracker;
      el.overlayTrackerName.textContent = settings.tracker.includes('botsort') ? 'BoT-SORT' : 'ByteTrack';
    }
    if (settings.conf !== undefined) {
      el.confRange.value = settings.conf;
      el.confValue.textContent = parseFloat(settings.conf).toFixed(2);
    }
    if (settings.iou !== undefined) {
      el.iouRange.value = settings.iou;
      el.iouValue.textContent = parseFloat(settings.iou).toFixed(2);
    }
    if (settings.show_trails !== undefined) {
      el.toggleTrails.checked = settings.show_trails;
    }
    if (settings.show_hud !== undefined) {
      el.toggleHud.checked = settings.show_hud;
    }
  } catch (e) {
    // Ignore setting fetch failure on load
  }
}

async function applySettings() {
  const conf = parseFloat(el.confRange.value);
  const iou = parseFloat(el.iouRange.value);
  const model = el.modelSelect.value;
  const tracker = el.trackerSelect.value;
  const show_trails = el.toggleTrails.checked;
  const show_hud = el.toggleHud.checked;
  const classes = state.selectedClasses.includes('all') ? null : state.selectedClasses;

  const payload = {
    model,
    tracker,
    conf,
    iou,
    show_trails,
    show_hud,
    classes,
  };

  try {
    const res = await fetch(`${API_BASE}/settings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.message || 'Error updating settings');
    }

    el.overlayTrackerName.textContent = tracker.includes('botsort') ? 'BoT-SORT' : 'ByteTrack';
    showToast('Settings applied successfully', 'success');
    addLog(`[SETTINGS] Model: ${model}, Tracker: ${tracker}, Conf: ${conf}, IoU: ${iou}`, 'info');
  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
    addLog(`[ERROR] Settings update failed: ${err.message}`, 'error');
  }
}

/* ==========================================================================
   Screenshot & Recording
   ========================================================================== */
async function captureScreenshot() {
  try {
    const res = await fetch(`${API_BASE}/screenshot`, { method: 'POST' });
    const data = await res.json();

    if (data.status === 'ok') {
      showToast('Screenshot saved to screenshots/', 'success');
      addLog(`[SCREENSHOT] Saved snapshot: ${data.file}`, 'success');

      el.exportAlert.classList.remove('hidden');
      el.exportAlertMsg.textContent = `Snapshot saved: ${data.file}`;
      setTimeout(() => el.exportAlert.classList.add('hidden'), 5000);

      // Download file to browser automatically
      if (data.url) {
        const a = document.createElement('a');
        a.href = `${API_BASE}${data.url}`;
        a.download = data.file.split('/').pop();
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
      }
    } else {
      throw new Error(data.message || 'Capture failed');
    }
  } catch (err) {
    showToast(`Screenshot failed: ${err.message}`, 'error');
  }
}

async function toggleRecording() {
  const targetAction = state.isRecording ? 'stop' : 'start';

  try {
    const res = await fetch(`${API_BASE}/recording/${targetAction}`, { method: 'POST' });
    const data = await res.json();

    if (data.status === 'ok') {
      setRecordingState(!state.isRecording);
      if (state.isRecording) {
        showToast('Video recording started', 'success');
        addLog(`[RECORDING] Started output recording: ${data.file || 'output/record.mp4'}`, 'success');
      } else {
        showToast('Recording saved to output/', 'info');
        addLog(`[RECORDING] Finalized output video.`, 'info');
      }
    } else {
      throw new Error(data.message || 'Recording toggle failed');
    }
  } catch (err) {
    showToast(`Recording error: ${err.message}`, 'error');
  }
}

function setRecordingState(isRec) {
  state.isRecording = isRec;
  if (isRec) {
    el.toggleRecordText.textContent = 'Stop Recording';
    el.recordingStatusText.classList.remove('hidden');
    el.quickRecordBtn.classList.add('recording');
  } else {
    el.toggleRecordText.textContent = 'Start Recording Video';
    el.recordingStatusText.classList.add('hidden');
    el.quickRecordBtn.classList.remove('recording');
  }
}

function toggleFullscreen() {
  if (!document.fullscreenElement) {
    el.videoContainer.requestFullscreen().catch((err) => {
      showToast(`Fullscreen error: ${err.message}`, 'error');
    });
  } else {
    document.exitFullscreen();
  }
}

/* ==========================================================================
   Helpers: Badge Updates, Copy URL, Toasts, Logs
   ========================================================================== */
function updateServerBadge(status, label) {
  el.serverStatusBadge.className = `status-pill ${status}`;
  el.serverStatusText.textContent = label;
}

function updatePhoneBadge(status, label) {
  el.phoneStatusBadge.className = `status-pill ${status}`;
  el.phoneStatusText.textContent = label;
}

function copyApkUrl() {
  const url = el.apkUrlCode.textContent;
  navigator.clipboard.writeText(url).then(() => {
    showToast('Copied APK URL to clipboard!', 'success');
  }).catch(() => {
    showToast('Failed to copy. Please manually copy: ' + url, 'info');
  });
}

function showToast(msg, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = msg;

  el.toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(8px)';
    setTimeout(() => toast.remove(), 250);
  }, 3500);
}

function addLog(message, type = 'info') {
  const line = document.createElement('div');
  line.className = `log-line ${type}`;
  const time = new Date().toLocaleTimeString();
  line.textContent = `[${time}] ${message}`;

  el.logEntries.appendChild(line);
  el.logEntries.scrollTop = el.logEntries.scrollHeight;
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// Start on DOM ready
document.addEventListener('DOMContentLoaded', init);
