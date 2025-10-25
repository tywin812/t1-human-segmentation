// seg.js - Webcam person segmentation with YOLO11n-seg ONNX (onnxruntime-web)

// Basic configurable options
const MODEL_URL = 'best.onnx'; // root-level ONNX provided by user
const INPUT_SIZE = 640; // assumed YOLO input; adjust if your model differs
const CONF_THRESHOLD = 0.25;
const IOU_THRESHOLD = 0.5;
const PERSON_CLASS_ID = 0; // COCO 'person'
const CROP_WITH_BOX = false; // set true to restrict mask by bbox once geometry is verified

// DOM elements
const webcamEl = document.getElementById('webcam');
const segCanvas = document.getElementById('segCanvas');
const segCtx = segCanvas.getContext('2d');
const modelStatusEl = document.getElementById('modelStatus');
const bgFileEl = document.getElementById('bgFile');
const startBtn = document.getElementById('startSeg');
const stopBtn = document.getElementById('stopSeg');

// State
let session = null;
let running = false;
let rafId = null;
let bgImage = null;
let usingRVFC = false;
let inferBusy = false;
let frameCounter = 0;

// Offscreen helpers
const offscreenVideoCanvas = document.createElement('canvas');
const offscreenVideoCtx = offscreenVideoCanvas.getContext('2d', { willReadFrequently: true });
const offscreenMaskCanvas = document.createElement('canvas');
const offscreenMaskCtx = offscreenMaskCanvas.getContext('2d');
const offscreenFGCanvas = document.createElement('canvas');
const offscreenFGCtx = offscreenFGCanvas.getContext('2d');
// Reusable helper canvases to reduce allocations per frame
const scaledMaskCanvas = document.createElement('canvas'); // INPUT_SIZE x INPUT_SIZE
const noPadMaskCanvas = document.createElement('canvas');   // nw x nh
const targetMaskCanvas = document.createElement('canvas');  // segCanvas size

// Configure ORT WASM assets path (required for CDN usage)
if (window.ort && ort.env && ort.env.wasm) {
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
  try { ort.env.wasm.simd = true; } catch(_) {}
}

async function loadModel() {
  try {
    modelStatusEl.textContent = 'Модель: загружается…';
    const providers = [];
    const hasWebGPU = typeof navigator !== 'undefined' && 'gpu' in navigator;
    if (hasWebGPU && ort.env && ort.env.webgpu && typeof ort.env.webgpu.init === 'function') {
      try {
        await ort.env.webgpu.init();
        providers.push('webgpu');
      } catch (e) {
        console.warn('WebGPU init failed, fallback to WASM:', e);
      }
    }
    providers.push('wasm');
    session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: providers,
      graphOptimizationLevel: 'all'
    });
    modelStatusEl.textContent = 'Модель: загружена';
  } catch (e) {
    console.error('Ошибка загрузки модели:', e);
    modelStatusEl.textContent = 'Модель: ошибка загрузки (см. консоль)';
  }
}

function logTensorInfoMap(map, label = 'outputs') {
  try {
    const entries = Object.entries(map).map(([name, t]) => ({ name, dims: t.dims }));
    console.log(`[debug] ${label}:`, entries);
  } catch (e) {
    console.warn('[debug] failed to log tensor info', e);
  }
}

async function startWebcam() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
  webcamEl.srcObject = stream;
  await webcamEl.play();
}

function stopWebcam() {
  const s = webcamEl.srcObject;
  if (s) {
    s.getTracks().forEach(t => t.stop());
  }
  webcamEl.srcObject = null;
}

bgFileEl.addEventListener('change', e => {
  const file = e.target.files && e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    const img = new Image();
    img.onload = () => {
      bgImage = img;
    };
    img.src = reader.result;
  };
  reader.readAsDataURL(file);
});

startBtn.addEventListener('click', async () => {
  if (!session) await loadModel();
  await startWebcam();
  running = true;
  startBtn.disabled = true;
  stopBtn.disabled = false;
  usingRVFC = typeof webcamEl.requestVideoFrameCallback === 'function';
  if (usingRVFC) {
    const cb = async () => {
      if (!running) return;
      if (!inferBusy) await inferOnce();
      webcamEl.requestVideoFrameCallback(cb);
    };
    webcamEl.requestVideoFrameCallback(cb);
  } else {
    loop();
  }
});

stopBtn.addEventListener('click', () => {
  running = false;
  if (rafId) cancelAnimationFrame(rafId);
  stopWebcam();
  startBtn.disabled = false;
  stopBtn.disabled = true;
});

function drawWebcamFull() {
  const targetW = segCanvas.width;
  const targetH = segCanvas.height;
  const vw = webcamEl.videoWidth;
  const vh = webcamEl.videoHeight;
  if (!vw || !vh) return;
  const ratio = Math.max(targetW / vw, targetH / vh);
  const dw = Math.round(vw * ratio);
  const dh = Math.round(vh * ratio);
  const dx = Math.floor((targetW - dw) / 2);
  const dy = Math.floor((targetH - dh) / 2);
  segCtx.clearRect(0, 0, targetW, targetH);
  segCtx.drawImage(webcamEl, 0, 0, vw, vh, dx, dy, dw, dh);
}

// Letterbox resize to square INPUT_SIZE keeping aspect ratio
function letterbox(video, size) {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const scale = Math.min(size / vw, size / vh);
  const nw = Math.round(vw * scale);
  const nh = Math.round(vh * scale);
  const padX = Math.floor((size - nw) / 2);
  const padY = Math.floor((size - nh) / 2);

  offscreenVideoCanvas.width = size;
  offscreenVideoCanvas.height = size;
  offscreenVideoCtx.clearRect(0, 0, size, size);
  offscreenVideoCtx.fillStyle = 'black';
  offscreenVideoCtx.fillRect(0, 0, size, size);
  offscreenVideoCtx.drawImage(video, 0, 0, vw, vh, padX, padY, nw, nh);

  return { scale, nw, nh, padX, padY };
}

function toTensorFromCanvas(canvas) {
  const { width, height } = canvas;
  const imgData = offscreenVideoCtx.getImageData(0, 0, width, height);
  const data = imgData.data;
  const size = width * height;
  const out = new Float32Array(3 * size);
  // Normalize to 0..1 and convert to CHW
  for (let i = 0; i < size; i++) {
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;
    out[i] = r;
    out[i + size] = g;
    out[i + 2 * size] = b;
  }
  return new ort.Tensor('float32', out, [1, 3, height, width]);
}

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

function iou(boxA, boxB) {
  const x1 = Math.max(boxA[0], boxB[0]);
  const y1 = Math.max(boxA[1], boxB[1]);
  const x2 = Math.min(boxA[2], boxB[2]);
  const y2 = Math.min(boxA[3], boxB[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
  const areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);
  const union = areaA + areaB - inter + 1e-6;
  return inter / union;
}

function nms(boxes, scores, iouThr) {
  const idxs = scores.map((s, i) => [s, i]).sort((a, b) => b[0] - a[0]).map(p => p[1]);
  const selected = [];
  idxs.forEach(i => {
    const keep = selected.every(j => iou(boxes[i], boxes[j]) < iouThr);
    if (keep) selected.push(i);
  });
  return selected;
}

function parseOutputs(outputs) {
  logTensorInfoMap(outputs);
  // Identify prediction and proto outputs by dims
  let predTensor = null;
  let protoTensor = null;
  for (const name of Object.keys(outputs)) {
    const t = outputs[name];
    const d = t.dims;
    if (d.length === 3 && d[0] === 1) predTensor = t;
    if (d.length === 4 && d[0] === 1) protoTensor = t;
  }
  if (!predTensor || !protoTensor) {
    console.error('[parseOutputs] Cannot find prediction(1xNxD or 1xDxN) and proto(1xCxHxW) among:', Object.keys(outputs));
    throw new Error('Unexpected model outputs');
  }

  const pred = predTensor.data; // Float32Array
  const predDims = predTensor.dims; // [1, N, D] or [1, D, N]
  let N = predDims[1];
  let D = predDims[2];
  let transposed = false;
  // If second dim (likely D) is smaller than third dim (likely N), then layout is [1, D, N]
  if (predDims.length === 3 && predDims[1] < predDims[2]) {
    N = predDims[2];
    D = predDims[1];
    transposed = true;
  }
  const proto = protoTensor.data; // Float32Array
  const protoDims = protoTensor.dims; // [1, C, H, W]
  const C = protoDims[1];
  const H = protoDims[2];
  const W = protoDims[3];

  const boxes = [];
  const scores = [];
  const classes = [];
  const masksCoeff = [];

  const nm = C; // number of mask coefficients equals proto channels (proto C)
  // Infer layout: either 4 + 1 + nc + nm (with obj) or 4 + nc + nm (no obj)
  let hasObj = false;
  const nc_with_obj = D - 4 - 1 - nm;
  const nc_no_obj = D - 4 - nm;
  let nc;
  if (nc_with_obj >= 1 && nc_no_obj >= 1) {
    // Prefer no-objectness layout for seg heads when both are plausible
    hasObj = false;
    nc = nc_no_obj;
  } else if (nc_with_obj >= 1) {
    hasObj = true;
    nc = nc_with_obj;
  } else {
    hasObj = false;
    nc = Math.max(1, nc_no_obj);
  }

  for (let i = 0; i < N; i++) {
    const base = transposed ? i : i * D;
    const read = (idx) => transposed ? pred[idx * N + i] : pred[base + idx];
    const cx = read(0);
    const cy = read(1);
    const w = read(2);
    const h = read(3);
    const x1 = cx - w / 2;
    const y1 = cy - h / 2;
    const x2 = cx + w / 2;
    const y2 = cy + h / 2;

    const obj = hasObj ? sigmoid(read(4)) : 1.0;
    // classes (sigmoid)
    let bestCls = 0;
    let bestProb = 0;
    const clsStart = hasObj ? 5 : 4;
    for (let c = 0; c < nc; c++) {
      const p = sigmoid(read(clsStart + c));
      if (p > bestProb) { bestProb = p; bestCls = c; }
    }
    const conf = obj * bestProb;
    if (conf < CONF_THRESHOLD) continue;
    if (bestCls !== PERSON_CLASS_ID) continue;

    boxes.push([x1, y1, x2, y2]);
    scores.push(conf);
    classes.push(bestCls);

    const coeff = new Float32Array(nm);
    const coffStart = clsStart + nc;
    for (let k = 0; k < nm; k++) coeff[k] = read(coffStart + k);
    masksCoeff.push(coeff);
  }

  // If boxes look normalized (<=1.5), scale to input size
  if (boxes.length) {
    let maxCoord = 0;
    for (const b of boxes) maxCoord = Math.max(maxCoord, b[0], b[1], b[2], b[3]);
    if (maxCoord <= 1.5) {
      for (const b of boxes) {
        b[0] *= INPUT_SIZE; b[1] *= INPUT_SIZE; b[2] *= INPUT_SIZE; b[3] *= INPUT_SIZE;
      }
    }
  }

  return { boxes, scores, classes, masksCoeff, proto, protoDims };
}

function buildMask(coeff, proto, protoDims, inputSize, box) {
  // proto: [1, C, H, W] flattened; coeff: [C]
  const C = protoDims[1];
  const H = protoDims[2];
  const W = protoDims[3];
  const area = H * W;

  // Linear combination: mask = sigmoid(sum_k coeff[k] * proto[k,:,:])
  const mask = new Float32Array(area);
  for (let k = 0; k < C; k++) {
    const coeffK = coeff[k];
    const offset = k * area;
    for (let i = 0; i < area; i++) {
      mask[i] += coeffK * proto[offset + i];
    }
  }
  for (let i = 0; i < area; i++) mask[i] = sigmoid(mask[i]);

  // Upscale to inputSize x inputSize using canvas
  offscreenMaskCanvas.width = W;
  offscreenMaskCanvas.height = H;
  const imgData = offscreenMaskCtx.createImageData(W, H);
  for (let i = 0; i < area; i++) {
    const v = Math.max(0, Math.min(255, Math.round(mask[i] * 255)));
    const idx = i * 4;
    imgData.data[idx] = 255;
    imgData.data[idx + 1] = 255;
    imgData.data[idx + 2] = 255;
    imgData.data[idx + 3] = v; // store in alpha for convenience
  }
  offscreenMaskCtx.putImageData(imgData, 0, 0);

  // Scale to model input size (reuse canvas)
  scaledMaskCanvas.width = inputSize;
  scaledMaskCanvas.height = inputSize;
  const sctx = scaledMaskCanvas.getContext('2d');
  sctx.imageSmoothingEnabled = true;
  sctx.drawImage(offscreenMaskCanvas, 0, 0, W, H, 0, 0, inputSize, inputSize);

  if (!box) return scaledMaskCanvas;

  // Optionally apply bbox crop to reduce spill
  const [x1, y1, x2, y2] = box.map(v => Math.max(0, Math.min(inputSize, v)));
  const cropData = sctx.getImageData(0, 0, inputSize, inputSize);
  const d = cropData.data;
  for (let y = 0; y < inputSize; y++) {
    for (let x = 0; x < inputSize; x++) {
      const idx = (y * inputSize + x) * 4 + 3; // alpha
      if (x < x1 || x > x2 || y < y1 || y > y2) d[idx] = 0;
    }
  }
  sctx.putImageData(cropData, 0, 0);
  return scaledMaskCanvas;
}

function composeAndDraw(maskCanvas, letterboxInfo) {
  const { padX, padY, nw, nh } = letterboxInfo;
  const targetW = segCanvas.width;
  const targetH = segCanvas.height;

  // Draw background first
  if (bgImage) {
    // cover background
    const ratio = Math.max(targetW / bgImage.width, targetH / bgImage.height);
    const bw = Math.round(bgImage.width * ratio);
    const bh = Math.round(bgImage.height * ratio);
    const bx = Math.floor((targetW - bw) / 2);
    const by = Math.floor((targetH - bh) / 2);
    segCtx.drawImage(bgImage, 0, 0, bgImage.width, bgImage.height, bx, by, bw, bh);
  } else {
    // if нет фона — оставим прозрачность/чистый холст; далее при пустой маске покажем вебкамеру
    segCtx.clearRect(0, 0, targetW, targetH);
  }

  // Prepare foreground from webcam frame
  offscreenFGCanvas.width = targetW;
  offscreenFGCanvas.height = targetH;
  offscreenFGCtx.clearRect(0, 0, targetW, targetH);

  // Draw webcam to FG canvas, covering target
  const vw = webcamEl.videoWidth;
  const vh = webcamEl.videoHeight;
  const ratio = Math.max(targetW / vw, targetH / vh);
  const dw = Math.round(vw * ratio);
  const dh = Math.round(vh * ratio);
  const dx = Math.floor((targetW - dw) / 2);
  const dy = Math.floor((targetH - dh) / 2);
  offscreenFGCtx.drawImage(webcamEl, 0, 0, vw, vh, dx, dy, dw, dh);

  // Prepare mask aligned to target: remove letterbox then scale to target (reuse canvases)
  noPadMaskCanvas.width = nw;
  noPadMaskCanvas.height = nh;
  const npctx = noPadMaskCanvas.getContext('2d');
  npctx.drawImage(maskCanvas, padX, padY, nw, nh, 0, 0, nw, nh);

  targetMaskCanvas.width = targetW;
  targetMaskCanvas.height = targetH;
  const tmctx = targetMaskCanvas.getContext('2d');
  tmctx.imageSmoothingEnabled = true;
  // scale noPad mask to match FG draw (cover)
  tmctx.drawImage(noPadMaskCanvas, 0, 0, nw, nh, dx, dy, dw, dh);

  // If mask is effectively empty, fallback to raw webcam view
  // Throttle expensive readback check: only every ~20 frames
  try {
    frameCounter = (frameCounter + 1) % 20;
    if (frameCounter === 0) {
      const m = tmctx.getImageData(0, 0, targetW, targetH).data;
      let alphaSum = 0;
      for (let i = 3; i < m.length; i += 4) alphaSum += m[i];
      if (alphaSum < 1000) { // tiny coverage => likely failure
        drawWebcamFull();
        return;
      }
    }
  } catch (e) {
    // getImageData can fail in rare cases; ignore and continue
  }

  // Apply mask: keep FG where mask alpha>0
  offscreenFGCtx.globalCompositeOperation = 'destination-in';
  offscreenFGCtx.drawImage(targetMaskCanvas, 0, 0);
  offscreenFGCtx.globalCompositeOperation = 'source-over';

  // Composite onto background
  segCtx.drawImage(offscreenFGCanvas, 0, 0);
}

async function inferOnce() {
  if (!session || !webcamEl.videoWidth || !webcamEl.videoHeight) return;
  if (inferBusy) return;
  inferBusy = true;

  // Preprocess
  const lb = letterbox(webcamEl, INPUT_SIZE);
  const input = toTensorFromCanvas(offscreenVideoCanvas);

  // Run
  const feeds = {};
  feeds[session.inputNames[0]] = input;
  let rawOutputs;
  try {
    rawOutputs = await session.run(feeds);
  } catch (e) {
    console.error('[inferOnce] session.run failed:', e);
    drawWebcamFull();
    return;
  }

  // Normalize outputs to a map name->tensor
  const outputs = {};
  session.outputNames.forEach(n => { outputs[n] = rawOutputs[n]; });

  // Postprocess
  let boxes, scores, masksCoeff, proto, protoDims;
  try {
    ({ boxes, scores, masksCoeff, proto, protoDims } = parseOutputs(outputs));
  } catch (e) {
    drawWebcamFull();
    return;
  }
  if (boxes.length === 0) {
    // No person: show webcam directly so background doesn't hide everything
    drawWebcamFull();
    return;
  }

  const keep = nms(boxes, scores, IOU_THRESHOLD);
  // For now, use the best one (first after NMS)
  const k = keep[0];
  const maskCanvas = buildMask(masksCoeff[k], proto, protoDims, INPUT_SIZE, CROP_WITH_BOX ? boxes[k] : null);

  composeAndDraw(maskCanvas, lb);
  inferBusy = false;
}

async function loop() {
  if (!running) return;
  await inferOnce();
  rafId = requestAnimationFrame(loop);
}

// Preload model on page load (does not start webcam)
loadModel();
