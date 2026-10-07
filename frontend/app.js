/* ==========================================================================
   Drum Practice Generator - frontend logic
   Plain ES2020, no build step. The server does all the work; this file only
   collects parameters, polls job status and renders results.
   ========================================================================== */
'use strict';

const $ = (id) => document.getElementById(id);

const state = {
  system: null,
  sourceId: null,
  sourcePath: null,
  fileMeta: null,
  detectedBpm: null,
  jobId: null,
  pollTimer: null,
  models: [],
  syncing: false,          // guards the speed <-> BPM two-way binding
};

/* ------------------------------------------------------------------ utils */
function toast(message, isError = false) {
  const el = $('toast');
  el.textContent = message;
  el.classList.toggle('err', isError);
  el.classList.add('show');
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.remove('show'), 3200);
}

function fmtBytes(bytes) {
  if (!bytes) return '—';
  const mb = bytes / (1024 * 1024);
  return mb >= 1 ? `${mb.toFixed(2)} MB` : `${(bytes / 1024).toFixed(0)} KB`;
}

function fmtDuration(seconds) {
  if (!seconds) return '—';
  const s = Math.round(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  return h > 0
    ? `${h}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
    : `${m}:${String(sec).padStart(2, '0')}`;
}

/** Render a structured backend error (code/message/suggestions/detail). */
function showError(err) {
  const box = $('error');
  const payload = (err && err.error) ? err.error : {
    message: (err && err.message) || '未知错误',
    suggestions: [],
    detail: null,
  };

  $('errorMessage').textContent = payload.message || '未知错误';

  const list = $('errorSuggestions');
  list.innerHTML = '';
  (payload.suggestions || []).forEach((item) => {
    const li = document.createElement('li');
    li.textContent = item;
    list.appendChild(li);
  });

  if (payload.detail) {
    $('errorDetailBox').style.display = '';
    $('errorDetail').textContent = payload.detail;
  } else {
    $('errorDetailBox').style.display = 'none';
  }

  box.classList.add('show');
  box.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function clearError() {
  $('error').classList.remove('show');
}

/** fetch wrapper that normalises our {"error": {...}} envelope. */
async function api(path, options = {}) {
  const response = await fetch(path, options);
  const text = await response.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch (_) {
    data = { error: { message: `服务器返回了非 JSON 响应（HTTP ${response.status}）`, detail: text.slice(0, 800) } };
  }
  if (!response.ok) {
    const error = new Error((data && data.error && data.error.message) || `HTTP ${response.status}`);
    error.payload = data;
    throw error;
  }
  return data;
}

/* ------------------------------------------------------------- system info */
async function loadSystem() {
  const dot = $('sysDot');
  const text = $('sysText');
  try {
    const info = await api('/api/system');
    state.system = info;

    const gpu = info.gpu;
    if (info.cuda_available && gpu) {
      dot.className = 'dot ok';
      text.textContent = `${gpu.name} · CUDA`;
    } else if (info.torch && !info.cuda_available) {
      dot.className = 'dot warn';
      text.textContent = 'CPU 模式（CUDA 不可用）';
    } else {
      dot.className = 'dot err';
      text.textContent = '缺少依赖';
    }

    renderSystemModal(info);
    $('generateHint').textContent = info.problems && info.problems.length
      ? '环境存在问题，请点击右上角状态查看详情。'
      : '准备就绪。点击生成后会执行：分析 → AI 分离鼓轨 → 调鼓音量 → 变速 → 混音 → 输出。';
  } catch (err) {
    dot.className = 'dot err';
    text.textContent = '无法连接后端';
    showError(err.payload || err);
  }
}

function renderSystemModal(info) {
  const rows = [
    ['操作系统', info.os],
    ['Python', `${info.python}${info.python_executable ? ` (${info.python_executable})` : ''}`],
    ['PyTorch', info.torch || '未安装'],
    ['CUDA 构建', info.torch_cuda_build || 'CPU 版'],
    ['CUDA 可用', info.cuda_available ? '是' : '否'],
    ['GPU', info.gpu ? info.gpu.name : '未检测到'],
    ['显存', info.gpu ? `${info.gpu.free_vram_mb} / ${info.gpu.total_vram_mb} MB 可用` : '—'],
    ['计算能力', info.gpu ? `sm_${String(info.gpu.compute_capability).replace('.', '')}` : '—'],
    ['FFmpeg', (info.ffmpeg && info.ffmpeg.found) ? `${info.ffmpeg.version} (${info.ffmpeg.source})` : '未找到'],
    ['Demucs', info.demucs || '未安装'],
    ['磁盘可用', `${info.free_disk_gb} GB`],
    ['输出目录', info.output_dir],
  ];

  $('sysBody').innerHTML = rows
    .map(([k, v]) => `<div class="k">${escapeHtml(k)}</div><div class="v">${escapeHtml(String(v))}</div>`)
    .join('');

  const problems = [...(info.problems || []), ...(info.warnings || [])];
  $('sysProblems').innerHTML = problems.length
    ? `<div class="field"><div class="k">问题与提示</div><ul class="notes">${problems
        .map((p) => `<li>${escapeHtml(p)}</li>`).join('')}</ul></div>`
    : '<div class="hint">未检测到问题。</div>';

  const backends = info.stretch_backends || {};
  $('speedHint').textContent = backends.rubberband
    ? '变速使用 Rubber Band（高质量，音高不变）。'
    : '变速使用 FFmpeg atempo（音高不变）。把 rubberband.exe 放进 tools\\rubberband 可获得更高质量。';
}

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (ch) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]
  ));
}

/* ----------------------------------------------------------------- models */
async function loadModels() {
  try {
    const data = await api('/api/models');
    state.models = data.models || [];

    const select = $('model');
    select.innerHTML = '';
    state.models.forEach((model) => {
      const option = document.createElement('option');
      option.value = model.signature;
      option.textContent = `${model.display_name}${model.installed ? '' : '（未下载）'}`;
      if (model.signature === data.default) option.selected = true;
      select.appendChild(option);
    });
    updateModelHint();
  } catch (err) {
    // Non-fatal: the user can still see the rest of the UI.
    console.warn('loadModels failed', err);
  }
}

function updateModelHint() {
  const signature = $('model').value;
  const model = state.models.find((m) => m.signature === signature);
  const button = $('btnDownloadModel');
  if (!model) {
    $('modelHint').textContent = '';
    button.disabled = true;
    return;
  }
  $('modelHint').textContent = `${model.description} 约 ${model.size_mb} MB`;
  button.disabled = model.installed;
  button.textContent = model.installed ? '已安装' : '下载模型';
}

/* ------------------------------------------------------------ file import */
async function uploadFile(file) {
  clearError();
  $('result').classList.remove('show');

  const extension = (file.name.match(/\.[^.]+$/) || [''])[0].toLowerCase();
  const allowed = (state.system && state.system.supported_extensions) ||
    ['.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac'];
  if (!allowed.includes(extension)) {
    showError({ error: {
      message: `不支持的文件格式：${file.name}`,
      suggestions: [`支持：${allowed.join('、')}`],
    } });
    return;
  }

  const drop = $('drop');
  drop.querySelector('.big').innerHTML = '<span class="spinner"></span>上传中…';
  drop.querySelector('.small').textContent = `${file.name} · ${fmtBytes(file.size)}`;

  const form = new FormData();
  form.append('file', file);

  try {
    const data = await api('/api/upload', { method: 'POST', body: form });
    state.sourceId = data.source_id;
    state.sourcePath = null;
    state.fileMeta = data;
    state.detectedBpm = null;

    renderFileInfo(data);
    drop.querySelector('.big').textContent = '把歌曲拖到这里，或点击选择文件';
    drop.querySelector('.small').textContent = '支持 MP3 / WAV / FLAC / M4A / OGG / AAC';
    $('btnGenerate').disabled = false;
    $('generateHint').textContent = '准备就绪，点击生成练习曲。';

    analyzeBpm();
  } catch (err) {
    showError(err.payload || err);
    drop.querySelector('.big').textContent = '把歌曲拖到这里，或点击选择文件';
    drop.querySelector('.small').textContent = '支持 MP3 / WAV / FLAC / M4A / OGG / AAC';
  }
}

function renderFileInfo(data) {
  const audio = data.audio || {};
  $('fileInfoBox').style.display = '';
  $('fiName').textContent = data.filename;
  $('fiDuration').textContent = audio.duration_hms || fmtDuration(audio.duration);
  $('fiFormat').textContent = (audio.format_name || '—').split(',')[0].toUpperCase();
  $('fiRate').textContent = audio.sample_rate ? `${(audio.sample_rate / 1000).toFixed(1)} kHz` : '—';
  $('fiChannels').textContent = audio.channels === 2 ? '立体声 (2ch)' : `${audio.channels} 声道`;
  $('fiSize').textContent = fmtBytes(data.size_bytes);
  $('fiBpm').textContent = '检测中…';
}

async function analyzeBpm() {
  if (!state.sourceId && !state.sourcePath) return;
  $('fiBpm').textContent = '检测中…';

  try {
    const data = await api('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source_id: state.sourceId, source_path: state.sourcePath }),
    });

    const bpm = data.bpm || {};
    state.detectedBpm = bpm.ok ? bpm.bpm : (bpm.bpm || null);

    if (bpm.ok) {
      $('fiBpm').textContent = `${Math.round(bpm.bpm)} BPM（置信度 ${(bpm.confidence * 100).toFixed(0)}%）`;
      if (!$('originalBpm').value) {
        $('originalBpm').value = bpm.bpm.toFixed(1);
        syncFromOriginalBpm();
      }
    } else {
      $('fiBpm').textContent = '检测失败（请手动填写）';
      $('fiBpm').title = bpm.reason || '';
      toast(bpm.reason || 'BPM 自动检测失败，请手动填写 Original BPM', true);
    }
  } catch (err) {
    $('fiBpm').textContent = '检测失败（请手动填写）';
    console.warn('analyzeBpm failed', err);
  }
}

/* ----------------------------------------------------- speed <-> BPM sync */
function currentDrumVolume() {
  return Number($('drumVolume').value) / 100;
}

function syncFromOriginalBpm() {
  if (state.syncing) return;
  const original = Number($('originalBpm').value);
  const target = Number($('targetBpm').value);
  if (!original || !target) return;
  state.syncing = true;
  const speed = Math.max(0.5, Math.min(2, target / original));
  $('speed').value = Math.round(speed * 100);
  $('speedVal').textContent = `${Math.round(speed * 100)}%`;
  state.syncing = false;
}

function syncFromSpeed() {
  if (state.syncing) return;
  state.syncing = true;
  const speed = Number($('speed').value) / 100;
  const original = Number($('originalBpm').value) || state.detectedBpm;
  if (original) {
    const target = original * speed;
    $('targetBpm').value = target.toFixed(1);
  }
  state.syncing = false;
}

/* -------------------------------------------------------------- job status */
function setProgress(job) {
  $('progressWrap').classList.add('show');
  $('progressBar').style.width = `${job.percent || 0}%`;
  $('percentText').textContent = `${Math.round(job.percent || 0)}%`;

  const note = job.note || job.stage_label || '处理中…';
  $('stageNote').innerHTML = `<span class="spinner"></span>${escapeHtml(note)}`;
  $('progressTitle').textContent = job.stage_label ? `处理中 · ${job.stage_label}` : '处理中…';
  $('btnCancel').disabled = job.status !== 'running' && job.status !== 'queued';
}

function stopPolling() {
  if (state.pollTimer) {
    clearTimeout(state.pollTimer);
    state.pollTimer = null;
  }
}

async function poll(jobId) {
  try {
    const job = await api(`/api/jobs/${jobId}`);
    setProgress(job);

    if (job.status === 'running' || job.status === 'queued') {
      state.pollTimer = setTimeout(() => poll(jobId), 700);
      return;
    }

    // Terminal state
    stopPolling();
    $('btnCancel').disabled = true;
    $('generateHint').textContent = '准备就绪，点击生成练习曲。';
    $('btnGenerate').disabled = !(state.sourceId || state.sourcePath);

    if (job.status === 'done') {
      $('progressBar').style.width = '100%';
      $('percentText').textContent = '100%';
      $('stageNote').innerHTML = '✓ 已完成';
      showResult(job.result || {});
      setTimeout(() => $('progressWrap').classList.remove('show'), 1200);
      loadOutputs();
    } else if (job.status === 'cancelled') {
      $('progressWrap').classList.remove('show');
      toast('任务已取消');
    } else {
      $('progressWrap').classList.remove('show');
      showError({ error: job.error || { message: job.note || '任务失败' } });
    }
  } catch (err) {
    stopPolling();
    $('progressWrap').classList.remove('show');
    showError(err.payload || err);
  }
}

function showResult(result) {
  $('result').classList.add('show');
  $('rOutput').textContent = result.output_name || '—';
  $('rDuration').textContent = fmtDuration(result.duration_seconds);
  $('rParams').textContent = `鼓声 ${result.drum_volume_pct}% · 速度 ${result.speed_pct}%`;
  $('rBpm').textContent = result.target_bpm
    ? `${result.original_bpm ? `${Math.round(result.original_bpm)} → ` : ''}${Math.round(result.target_bpm)}`
    : (result.original_bpm ? `${Math.round(result.original_bpm)}（未变速）` : '—');
  $('rTime').textContent = `${result.processing_seconds}s`;
  $('rLoud').textContent = result.loudness_lufs
    ? `${result.loudness_lufs} LUFS`
    : '未测量';

  const pills = [
    `模型 ${result.model}`,
    `设备 ${String(result.device).toUpperCase()}`,
    `变速 ${result.stretch_backend}`,
    result.metronome_clicks ? `节拍器 ${result.metronome_clicks} 拍` : '无节拍器',
    result.mix_report && result.mix_report.peak_after_limiter !== undefined
      ? `峰值 ${Number(result.mix_report.peak_after_limiter).toFixed(3)}` : null,
  ].filter(Boolean);
  $('rPills').innerHTML = pills.map((p) => `<span class="pill">${escapeHtml(p)}</span>`).join('');

  const notes = result.notes || [];
  $('rNotes').innerHTML = notes.map((n) => `<li>${escapeHtml(n)}</li>`).join('');

  const player = $('player');
  player.src = `/api/audio/${state.jobId}`;
  player.load();

  $('result').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

async function cancelJob() {
  if (!state.jobId) return;
  $('btnCancel').disabled = true;
  try {
    await api(`/api/jobs/${state.jobId}/cancel`, { method: 'POST' });
    toast('已请求取消，等待当前步骤结束…');
  } catch (err) {
    showError(err.payload || err);
  }
}

/* ------------------------------------------------------------- generation */
async function generate() {
  clearError();
  $('result').classList.remove('show');

  if (!state.sourceId && !state.sourcePath) {
    showError({ error: { message: '请先导入歌曲。', suggestions: ['拖入或选择一个音频文件。'] } });
    return;
  }

  const payload = {
    source_id: state.sourceId,
    source_path: state.sourcePath,
    drum_volume: currentDrumVolume(),
    speed: Number($('speed').value) / 100,
    output_format: $('outputFormat').value,
    output_bits: Number($('outputBits').value),
    model: $('model').value,
    segment: Number($('segment').value),
    overlap: Number($('overlap').value),
    shifts: Number($('shifts').value),
    stretch_backend: $('stretchBackend').value,
    detect_bpm: true,
    metronome_enabled: $('metroEnabled').checked,
    metronome_volume: Number($('metroVolume').value) / 100,
    time_signature: $('timeSig').value,
    metronome_offset_ms: Number($('metroOffset').value),
    keep_stems: $('keepStems').checked,
    loudness_normalize: $('loudness').checked,
  };

  const original = Number($('originalBpm').value);
  const target = Number($('targetBpm').value);
  if (original > 0) payload.original_bpm = original;
  if (target > 0) payload.target_bpm = target;
  const metroBpm = Number($('metroBpm').value);
  if (metroBpm > 0) payload.metronome_bpm = metroBpm;

  $('btnGenerate').disabled = true;
  $('progressWrap').classList.add('show');
  $('progressBar').style.width = '2%';
  $('percentText').textContent = '0%';
  $('stageNote').innerHTML = '<span class="spinner"></span>正在排队…';

  try {
    const data = await api('/api/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    state.jobId = data.job_id;
    poll(data.job_id);
  } catch (err) {
    $('progressWrap').classList.remove('show');
    $('btnGenerate').disabled = false;
    showError(err.payload || err);
  }
}

/* ---------------------------------------------------------------- outputs */
async function loadOutputs() {
  try {
    const data = await api('/api/output');
    const select = $('outputList');
    const previous = select.value;
    select.innerHTML = '';

    (data.files || []).forEach((file) => {
      const option = document.createElement('option');
      option.value = file.path;
      option.textContent = `${file.name}  (${file.size_mb} MB)`;
      select.appendChild(option);
    });

    if (!data.files || !data.files.length) {
      const option = document.createElement('option');
      option.textContent = '（output 目录还没有文件）';
      option.disabled = true;
      select.appendChild(option);
    } else if (previous) {
      select.value = previous;
    }
  } catch (err) {
    console.warn('loadOutputs failed', err);
  }
}

async function downloadModel() {
  const model = $('model').value;
  const button = $('btnDownloadModel');
  button.disabled = true;
  button.innerHTML = '<span class="spinner"></span>下载中…';

  try {
    const data = await api('/api/models/download', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model }),
    });

    if (data.already_installed) {
      toast('模型已安装');
      await loadModels();
      return;
    }

    // Poll the download job.
    const jobId = data.job_id;
    const check = async () => {
      const job = await api(`/api/jobs/${jobId}`);
      setProgress(job);
      $('progressWrap').classList.add('show');
      if (job.status === 'running' || job.status === 'queued') {
        setTimeout(check, 900);
      } else if (job.status === 'done') {
        toast('模型下载完成');
        $('progressWrap').classList.remove('show');
        await loadModels();
      } else {
        $('progressWrap').classList.remove('show');
        showError({ error: job.error || { message: '模型下载失败' } });
        button.disabled = false;
        button.textContent = '下载模型';
      }
    };
    check();
  } catch (err) {
    showError(err.payload || err);
    button.disabled = false;
    button.textContent = '下载模型';
  }
}

/* ------------------------------------------------------------------- wire */
function wire() {
  // --- drag & drop ---
  const drop = $('drop');
  drop.addEventListener('click', () => $('fileInput').click());
  $('fileInput').addEventListener('change', (event) => {
    if (event.target.files[0]) uploadFile(event.target.files[0]);
  });

  ['dragenter', 'dragover'].forEach((type) => {
    drop.addEventListener(type, (event) => {
      event.preventDefault();
      drop.classList.add('dragover');
    });
  });
  ['dragleave', 'drop'].forEach((type) => {
    drop.addEventListener(type, (event) => {
      event.preventDefault();
      drop.classList.remove('dragover');
    });
  });
  drop.addEventListener('drop', (event) => {
    const file = event.dataTransfer && event.dataTransfer.files[0];
    if (file) uploadFile(file);
  });
  // Stop the browser from navigating away when a file misses the drop zone.
  ['dragover', 'drop'].forEach((type) => {
    window.addEventListener(type, (event) => event.preventDefault());
  });

  // --- local path ---
  $('btnLocalPath').addEventListener('click', async () => {
    const path = $('localPath').value.trim();
    if (!path) { toast('请输入本机文件路径', true); return; }
    clearError();
    try {
      const data = await api('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source_path: path }),
      });
      state.sourcePath = data.path;
      state.sourceId = null;
      state.fileMeta = data;
      state.detectedBpm = data.bpm && (data.bpm.bpm || null);
      renderFileInfo({ ...data, filename: data.filename, size_bytes: 0 });
      if (data.bpm && data.bpm.ok) {
        $('fiBpm').textContent = `${Math.round(data.bpm.bpm)} BPM`;
        $('originalBpm').value = data.bpm.bpm.toFixed(1);
        syncFromOriginalBpm();
      } else {
        $('fiBpm').textContent = '检测失败（请手动填写）';
      }
      $('btnGenerate').disabled = false;
      $('generateHint').textContent = '准备就绪，点击生成练习曲。';
      toast('已载入本机文件');
    } catch (err) {
      showError(err.payload || err);
    }
  });

  // --- drum volume ---
  $('drumVolume').addEventListener('input', () => {
    $('drumVolumeVal').textContent = `${$('drumVolume').value}%`;
  });
  document.querySelectorAll('[data-preset]').forEach((button) => {
    button.addEventListener('click', () => {
      const value = Number(button.dataset.preset);
      $('drumVolume').value = value;
      $('drumVolumeVal').textContent = `${value}%`;
    });
  });

  // --- speed / BPM ---
  $('speed').addEventListener('input', () => {
    $('speedVal').textContent = `${$('speed').value}%`;
    syncFromSpeed();
  });
  document.querySelectorAll('[data-speed]').forEach((button) => {
    button.addEventListener('click', () => {
      const value = Number(button.dataset.speed);
      $('speed').value = value;
      $('speedVal').textContent = `${value}%`;
      syncFromSpeed();
    });
  });
  $('targetBpm').addEventListener('input', syncFromOriginalBpm);
  $('originalBpm').addEventListener('input', syncFromOriginalBpm);

  // Octave quick-fix: periodicity-based detection cannot always tell a tempo
  // from its half/double, so give one-click corrections.
  const scaleBpm = (factor) => {
    const current = Number($('originalBpm').value) || state.detectedBpm;
    if (!current) { toast('请先检测或填写 Original BPM', true); return; }
    const scaled = Math.max(20, Math.min(400, current * factor));
    $('originalBpm').value = scaled.toFixed(1);
    state.detectedBpm = scaled;
    syncFromOriginalBpm();
    toast(`Original BPM 已设为 ${scaled.toFixed(1)}`);
  };
  $('btnBpmHalf').addEventListener('click', () => scaleBpm(0.5));
  $('btnBpmDouble').addEventListener('click', () => scaleBpm(2.0));

  // --- metronome ---
  $('metroVolume').addEventListener('input', () => {
    $('metroVolumeVal').textContent = `${$('metroVolume').value}%`;
  });

  // --- misc ---
  $('btnAnalyze').addEventListener('click', analyzeBpm);
  $('btnClear').addEventListener('click', () => {
    state.sourceId = null;
    state.sourcePath = null;
    state.fileMeta = null;
    state.detectedBpm = null;
    $('fileInfoBox').style.display = 'none';
    $('btnGenerate').disabled = true;
    $('generateHint').textContent = '请先导入歌曲。';
    $('result').classList.remove('show');
    $('originalBpm').value = '';
    $('targetBpm').value = '';
  });
  $('btnGenerate').addEventListener('click', generate);
  $('btnCancel').addEventListener('click', cancelJob);
  $('btnRegenerate').addEventListener('click', () => {
    $('result').classList.remove('show');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
  $('model').addEventListener('change', updateModelHint);
  $('btnDownloadModel').addEventListener('click', downloadModel);

  $('btnOpenFolder').addEventListener('click', async () => {
    try {
      await api('/api/open-folder', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
    } catch (err) {
      showError(err.payload || err);
    }
  });

  $('btnLoadOutput').addEventListener('click', () => {
    const path = $('outputList').value;
    if (!path) return;
    $('abPlayer').src = `/api/audio-file?path=${encodeURIComponent(path)}`;
    $('abPlayer').load();
    $('abPlayer').play().catch(() => {});
  });
  $('btnRefreshOutput').addEventListener('click', loadOutputs);

  // --- system modal ---
  $('sysPill').addEventListener('click', () => $('sysModal').classList.add('show'));
  $('btnCloseSys').addEventListener('click', () => $('sysModal').classList.remove('show'));
  $('sysModal').addEventListener('click', (event) => {
    if (event.target === $('sysModal')) $('sysModal').classList.remove('show');
  });
  $('btnShowLog').addEventListener('click', async () => {
    try {
      const data = await api('/api/logs?lines=200');
      $('logText').textContent = data.log || '（日志为空）';
      $('logBox').style.display = '';
      $('logBox').open = true;
    } catch (err) {
      toast('读取日志失败', true);
    }
  });
}

/* ------------------------------------------------------------------- init */
(async function init() {
  wire();
  await loadSystem();
  await loadModels();
  await loadOutputs();
  // Refresh the output list periodically so new files appear.
  setInterval(loadOutputs, 15000);
})();
