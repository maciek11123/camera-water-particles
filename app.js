/**
 * Gesture-Controlled 3D Earth
 * MediaPipe Hand Landmarker + Google Maps 3D API
 *
 * SETUP: Replace YOUR_GOOGLE_MAPS_API_KEY in index.html with your key.
 * Enable in Google Cloud Console: Map Tiles API + Maps JavaScript API.
 */

import { HandLandmarker, FilesetResolver } from
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.js';

// ─── Tuneable constants ───────────────────────────────────────────────────────

const CFG = {
  // Initial camera view (Paris at street-district scale)
  initialLat:     48.8566,
  initialLng:     2.3522,
  initialRange:   4000,   // meters from center to camera
  initialHeading: 0,      // degrees (0 = north)
  initialTilt:    45,     // degrees (0 = top-down, 90 = horizon)

  // Gesture classification thresholds (landmark coords are normalised [0,1])
  pinchThreshold: 0.07,   // thumb-tip ↔ index-tip distance  → ZOOM mode
  fistThreshold:  0.17,   // avg fingertip-wrist distance     → ROTATE mode
  palmThreshold:  0.28,   // avg fingertip-wrist distance     → PAN mode
  gestureDebounce: 5,     // frames before a new gesture activates

  // Sensitivity multipliers (increase = faster response)
  panSensitivity:    1.0,
  zoomSensitivity:   4.5,  // exponential factor; increase for faster zoom
  rotateSensitivity: 2.2,  // heading degrees per degree of hand rotation
  tiltSensitivity:   80,   // maps wrist.y [0–1] to tilt degrees [0–80]

  // LERP smoothing per frame (0 = frozen, 1 = instant)
  smoothing: 0.13,

  // Map limits
  minRange:  50,
  maxRange:  20_000_000,
  minTilt:   0,
  maxTilt:   85,

  // Preview canvas dimensions (px) — aspect should match webcam 4:3
  previewW: 240,
  previewH: 180,

  mediapipeVersion: '0.10.18',
};

// ─── Gesture identifiers ──────────────────────────────────────────────────────

const G = { IDLE: 'IDLE', PAN: 'PAN', ZOOM: 'ZOOM', ROTATE: 'ROTATE' };

const GESTURE_META = {
  [G.IDLE]:   { icon: '—',  label: 'No gesture',          color: '#888' },
  [G.PAN]:    { icon: '✋', label: 'Panning',              color: '#4CAF50' },
  [G.ZOOM]:   { icon: '🤏', label: 'Zooming',              color: '#2196F3' },
  [G.ROTATE]: { icon: '✊', label: 'Rotating & Tilting',   color: '#FF9800' },
};

// Hand skeleton connection pairs (MediaPipe indices)
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],           // thumb
  [0,5],[5,6],[6,7],[7,8],           // index
  [0,9],[9,10],[10,11],[11,12],      // middle
  [0,13],[13,14],[14,15],[15,16],    // ring
  [0,17],[17,18],[18,19],[19,20],    // pinky
  [5,9],[9,13],[13,17],              // palm arch
];

// ─── Camera state ─────────────────────────────────────────────────────────────

// `cam` is the smoothed value sent to the map each frame.
// `tgt` is what the gesture logic writes to; cam lerps toward tgt.
const cam = {
  lat: CFG.initialLat, lng: CFG.initialLng,
  range: CFG.initialRange, heading: CFG.initialHeading, tilt: CFG.initialTilt,
};
const tgt = { ...cam };

// ─── Gesture state ────────────────────────────────────────────────────────────

let activeGesture  = G.IDLE;
let pendingGesture = G.IDLE;
let pendingFrames  = 0;
let prevLm         = null;   // landmark array from previous frame

// ─── DOM ──────────────────────────────────────────────────────────────────────

const webcamEl    = document.getElementById('webcam');
const map3dEl     = document.getElementById('map3d');
const previewCv   = document.getElementById('preview-canvas');
const previewCtx  = previewCv.getContext('2d');
const gestureIcon = document.getElementById('gesture-icon');
const gestureLabel= document.getElementById('gesture-label');
const statusEl    = document.getElementById('status-text');
const trackDot    = document.getElementById('tracking-dot');
const loadingEl   = document.getElementById('loading-overlay');
const loadingTxt  = document.getElementById('loading-text');
const errorEl     = document.getElementById('error-overlay');
const apiWarning  = document.getElementById('api-warning');

// ─── Math helpers ─────────────────────────────────────────────────────────────

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

function lerp(a, b, t) { return a + (b - a) * t; }

// Lerp that takes the shortest path around a 360° circle
function lerpAngle(a, b, t) {
  const diff = ((b - a) % 360 + 540) % 360 - 180;
  return (a + diff * t + 360) % 360;
}

function dist2D(a, b) {
  const dx = a.x - b.x, dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

// ─── Gesture classification ───────────────────────────────────────────────────

function classify(lm) {
  const wrist = lm[0];

  // Priority 1: pinch — thumb tip (4) near index tip (8)
  if (dist2D(lm[4], lm[8]) < CFG.pinchThreshold) return G.ZOOM;

  // Average distance of four fingertips from wrist
  const avgDist = (dist2D(wrist, lm[8]) + dist2D(wrist, lm[12]) +
                   dist2D(wrist, lm[16]) + dist2D(wrist, lm[20])) / 4;

  // Priority 2: fist — all fingertips close to wrist
  if (avgDist < CFG.fistThreshold) return G.ROTATE;

  // Priority 3: open palm — fingertips spread far from wrist
  if (avgDist > CFG.palmThreshold) return G.PAN;

  return G.IDLE;
}

// ─── Gesture → map target ─────────────────────────────────────────────────────

function applyGesture(gesture, lm) {
  if (!prevLm) { prevLm = lm; return; }
  const p = prevLm;

  switch (gesture) {

    case G.PAN: {
      // Track palm centre (landmark 9: middle-finger MCP joint)
      const dx = lm[9].x - p[9].x;
      const dy = lm[9].y - p[9].y;

      // Scale proportionally to altitude so panning feels consistent at any zoom
      const k = (tgt.range * 0.0000014) * CFG.panSensitivity;
      tgt.lng -= dx * k * (180 / Math.PI) * 100;  // normalised → degrees
      tgt.lat += dy * k * (180 / Math.PI) * 60;
      tgt.lat  = clamp(tgt.lat, -85, 85);
      break;
    }

    case G.ZOOM: {
      // Track vertical movement of the pinch midpoint
      const midY     = (lm[4].y  + lm[8].y)  / 2;
      const prevMidY = (p[4].y   + p[8].y)   / 2;
      const dy = midY - prevMidY;
      // Moving hand upward (dy < 0) → zoom in (smaller range)
      tgt.range *= Math.exp(dy * CFG.zoomSensitivity);
      tgt.range  = clamp(tgt.range, CFG.minRange, CFG.maxRange);
      break;
    }

    case G.ROTATE: {
      // Heading: angle of wrist→index-MCP vector
      const angle     = Math.atan2(lm[5].y - lm[0].y, lm[5].x - lm[0].x) * (180 / Math.PI);
      const prevAngle = Math.atan2(p[5].y  - p[0].y,  p[5].x  - p[0].x)  * (180 / Math.PI);
      let dAngle = angle - prevAngle;
      if (dAngle >  180) dAngle -= 360;
      if (dAngle < -180) dAngle += 360;
      tgt.heading = ((tgt.heading + dAngle * CFG.rotateSensitivity) % 360 + 360) % 360;

      // Tilt: wrist's vertical position in the frame drives tilt angle
      tgt.tilt = clamp(lm[0].y * CFG.tiltSensitivity, CFG.minTilt, CFG.maxTilt);
      break;
    }
  }

  prevLm = lm;
}

// ─── Map update (called every frame) ─────────────────────────────────────────

function updateMap() {
  cam.lat     = lerp(cam.lat,     tgt.lat,     CFG.smoothing);
  cam.lng     = lerp(cam.lng,     tgt.lng,     CFG.smoothing);
  cam.range   = lerp(cam.range,   tgt.range,   CFG.smoothing);
  cam.tilt    = lerp(cam.tilt,    tgt.tilt,    CFG.smoothing);
  cam.heading = lerpAngle(cam.heading, tgt.heading, CFG.smoothing);

  map3dEl.center  = { lat: cam.lat, lng: cam.lng, altitude: 0 };
  map3dEl.range   = cam.range;
  map3dEl.heading = cam.heading;
  map3dEl.tilt    = cam.tilt;
}

// ─── Preview canvas: webcam + skeleton ────────────────────────────────────────

function drawPreview(lm, gesture) {
  const w = previewCv.width;
  const h = previewCv.height;
  previewCtx.clearRect(0, 0, w, h);

  // Draw mirrored webcam thumbnail
  if (webcamEl.readyState >= 2) {
    previewCtx.save();
    previewCtx.translate(w, 0);
    previewCtx.scale(-1, 1);
    previewCtx.drawImage(webcamEl, 0, 0, w, h);
    previewCtx.restore();
    // Slight dark overlay for contrast
    previewCtx.fillStyle = 'rgba(0,0,0,0.28)';
    previewCtx.fillRect(0, 0, w, h);
  }

  if (!lm || !lm.length) return;

  const meta  = GESTURE_META[gesture] || GESTURE_META[G.IDLE];
  const color = meta.color;

  // Mirror x so skeleton matches the mirrored webcam image
  const px = lmx => (1 - lmx.x) * w;
  const py = lmy => lmy.y * h;

  // Skeleton bones
  previewCtx.strokeStyle = color;
  previewCtx.lineWidth   = 2;
  previewCtx.globalAlpha = 0.82;
  for (const [a, b] of HAND_CONNECTIONS) {
    previewCtx.beginPath();
    previewCtx.moveTo(px(lm[a]), py(lm[a]));
    previewCtx.lineTo(px(lm[b]), py(lm[b]));
    previewCtx.stroke();
  }

  // Joint dots
  previewCtx.globalAlpha = 1;
  for (let i = 0; i < lm.length; i++) {
    previewCtx.beginPath();
    previewCtx.arc(px(lm[i]), py(lm[i]), i === 0 ? 5 : 3.5, 0, Math.PI * 2);
    previewCtx.fillStyle = i === 0 ? '#ffffff' : color;
    previewCtx.fill();
  }

  // Gesture badge
  previewCtx.fillStyle = 'rgba(0,0,0,0.55)';
  previewCtx.fillRect(4, 4, 145, 22);
  previewCtx.fillStyle    = color;
  previewCtx.font         = 'bold 11px system-ui, sans-serif';
  previewCtx.fillText(meta.label, 8, 18);
}

// ─── UI updates ───────────────────────────────────────────────────────────────

function updateUI(gesture, handPresent) {
  const meta = GESTURE_META[gesture] || GESTURE_META[G.IDLE];
  gestureIcon.textContent  = meta.icon;
  gestureLabel.textContent = meta.label;
  gestureLabel.style.color = meta.color;
  trackDot.classList.toggle('active', handPresent);
}

function setStatus(msg) {
  if (loadingTxt) loadingTxt.textContent = msg;
  if (statusEl)   statusEl.textContent   = msg;
}

function showError(title, msg) {
  loadingEl.style.display = 'none';
  errorEl.classList.remove('hidden');
  document.getElementById('error-title').textContent   = title;
  document.getElementById('error-message').textContent = msg;
}

// ─── Detection loop ───────────────────────────────────────────────────────────

let handLandmarker = null;

function detectionLoop(timestamp) {
  requestAnimationFrame(detectionLoop);

  if (!handLandmarker || webcamEl.readyState < 2) return;

  const results = handLandmarker.detectForVideo(webcamEl, timestamp);
  const hasHand = results.landmarks && results.landmarks.length > 0;

  if (hasHand) {
    const lm       = results.landmarks[0];
    const detected = classify(lm);

    // Debounce: require N consistent frames before committing a gesture change
    if (detected === pendingGesture) {
      pendingFrames++;
    } else {
      pendingGesture = detected;
      pendingFrames  = 0;
    }

    if (pendingFrames >= CFG.gestureDebounce && detected !== activeGesture) {
      prevLm        = null;   // reset per-gesture tracking on transition
      activeGesture = detected;
    }

    applyGesture(activeGesture, lm);
    drawPreview(lm, activeGesture);
  } else {
    activeGesture  = G.IDLE;
    pendingGesture = G.IDLE;
    pendingFrames  = 0;
    prevLm         = null;
    drawPreview(null, G.IDLE);
  }

  updateUI(activeGesture, hasHand);
  updateMap();
}

// ─── Initialisation ───────────────────────────────────────────────────────────

async function init() {
  previewCv.width  = CFG.previewW;
  previewCv.height = CFG.previewH;

  // Check for placeholder API key
  const scripts = [...document.querySelectorAll('script[src]')];
  const mapsScript = scripts.find(s => s.src.includes('maps.googleapis.com'));
  if (mapsScript && mapsScript.src.includes('YOUR_GOOGLE_MAPS_API_KEY')) {
    apiWarning.textContent =
      '⚠ Google Maps API key not set — open index.html and replace YOUR_GOOGLE_MAPS_API_KEY';
    apiWarning.classList.add('visible');
  }

  // 1. Webcam
  setStatus('Requesting camera access…');
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    webcamEl.srcObject = stream;
    await new Promise((res, rej) => {
      webcamEl.onloadeddata = res;
      webcamEl.onerror      = rej;
      webcamEl.play().catch(rej);
    });
  } catch (err) {
    const msg = err.name === 'NotAllowedError'
      ? 'Camera permission denied. Allow access in browser settings and reload.'
      : err.name === 'NotFoundError'
        ? 'No camera found. Connect a webcam and reload.'
        : `Camera error: ${err.message}`;
    showError('Camera Required', msg);
    return;
  }

  // 2. MediaPipe Hand Landmarker
  setStatus('Loading hand-tracking model…');
  try {
    const fs = await FilesetResolver.forVisionTasks(
      `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${CFG.mediapipeVersion}/wasm`
    );
    handLandmarker = await HandLandmarker.createFromOptions(fs, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numHands: 1,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence:  0.5,
      minTrackingConfidence:      0.5,
    });
  } catch (err) {
    console.error('MediaPipe load error:', err);
    showError('Hand Tracking Error',
      'Failed to load the hand-tracking model. Check your network connection and reload.');
    return;
  }

  // 3. Wait for Google Maps 3D custom element to be registered
  setStatus('Loading 3D map…');
  if (!customElements.get('gmp-map-3d')) {
    await customElements.whenDefined('gmp-map-3d');
  }

  // Set initial camera view via JS properties (more reliable than HTML attrs)
  map3dEl.center  = { lat: CFG.initialLat, lng: CFG.initialLng, altitude: 0 };
  map3dEl.range   = CFG.initialRange;
  map3dEl.heading = CFG.initialHeading;
  map3dEl.tilt    = CFG.initialTilt;

  // 4. Start
  loadingEl.style.opacity = '0';
  setTimeout(() => { loadingEl.style.display = 'none'; }, 500);
  setStatus('Show your hand to the camera');

  requestAnimationFrame(detectionLoop);
}

init();
