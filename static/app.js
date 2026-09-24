/* ============================================================
   VitalEdge — device viewer + live Flask/ONNX backend polling
   ------------------------------------------------------------
   Architecture:

     Flask (website sensor stream -> 60s rolling window ->
6-feature ONNX VitalEdgeDeepAutoencoder -> reconstruction
MSE -> 1.78 threshold -> NORMAL/ABNORMAL)-> NORMAL/ABNORMAL) is the single
     source of truth. This file never computes an anomaly score.

    sendSensorSample()  -> POST /api/data, once per second
        -> updateFromBackend(result)
        -> setHealthState(mapped status) + updateDashboard(result)
           + updateWatchFace()

   The six model features are exactly, in this order:
     [ HR/BVP, ACC, TEMP, RESP, EDA, ECG ]
   The browser sends plausible physical readings. Flask returns those
   readings in `display_values` and standardizes them for the model.

   Buttons call the backend directly:
     Simulate abnormality -> POST /api/simulate-abnormal
     Resume normal         -> POST /api/reset
  Neither one touches a local variable — both just ask the backend
  to change mode, then immediately send a sample through /api/data.

   If Flask can't be reached, the connection indicator switches to
   "BACKEND OFFLINE" and the dashboard is NOT filled with fabricated
   data — it simply stops updating until the next successful poll.

   The 3D watch body, strap, buttons, sensors and exploded-view
   parts are untouched by any of this — only the screen texture
   (updateWatchFace, further down) reacts to backend status.
   ============================================================ */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

// Set this to a URL (e.g. './device.glb') once a fabricated/scanned
// housing model exists. Left null -> uses the procedural reference model.
const EXTERNAL_MODEL_URL = null;

/* ---------------------------------------------------------------
   1. STATE — filled entirely from Flask, never computed locally
   --------------------------------------------------------------- */
const GRAPH_LEN = 60;
const POLL_INTERVAL_MS = 1000;

const deviceState = {
  status: 'normal',        // mapped for UI: 'normal' | 'warning' | 'emergency'
  statusRaw: null,         // as sent by Flask: 'NORMAL' | 'ABNORMAL' | 'COLLECTING'
  connected: false,
  windowReady: false,
  secondsCollected: 0,
  anomalyScore: null,
  threshold: null,
  values: null,            // normalized six-feature object from /api/status
  displayValues: null,     // optional future physical-unit object
  environment: null,       // weather and air-quality data from Flask
  environmentMode: null,
  emergencyAlertSent: false,
  updatedAt: new Date(),
};

const graphHistories = { hr: [], respiration: [], ecg: [] };

const els = {
  statusBanner: document.getElementById('status-banner'),
  statusIcon: document.getElementById('status-icon'),
  statusT1: document.getElementById('status-t1'),
  statusT2: document.getElementById('status-t2'),
  alertPanel: document.getElementById('alert-panel'),
  alertLine1: document.getElementById('alert-line1'),
  alertLine2: document.getElementById('alert-line2'),
  mlMse: document.getElementById('ml-mse'),
  mlThreshold: document.getElementById('ml-threshold'),
  mlChip: document.getElementById('ml-chip'),
  connStatus: document.getElementById('conn-status'),
  connDot: document.getElementById('conn-dot'),
  batteryStatus: document.getElementById('battery-status'),
  bufferBadge: document.getElementById('buffer-badge'),
  clock: document.getElementById('state-clock'),
  metricHr: document.getElementById('metric-hr'),
  metricActivity: document.getElementById('metric-activity'),
  metricTemperature: document.getElementById('metric-temperature'),
  metricRespiration: document.getElementById('metric-respiration'),
  metricEda: document.getElementById('metric-eda'),
  metricEcg: document.getElementById('metric-ecg'),
  graphHr: document.getElementById('graph-hr'),
  graphResp: document.getElementById('graph-resp'),
  graphEcg: document.getElementById('graph-ecg'),
  weatherMode: document.getElementById('weather-mode'),
  weatherTemperature: document.getElementById('weather-temperature'),
  weatherHumidity: document.getElementById('weather-humidity'),
  weatherAqi: document.getElementById('weather-aqi'),
  weatherPm25: document.getElementById('weather-pm25'),
  weatherPm10: document.getElementById('weather-pm10'),
  weatherUv: document.getElementById('weather-uv'),
  weatherLocation: document.getElementById('weather-location'),
  weatherSource: document.getElementById('weather-source'),
  healthScore: document.getElementById('health-score'),
  environmentScore: document.getElementById('environment-score'),
  finalRiskScore: document.getElementById('final-risk-score'),
  btnAbnormal: document.getElementById('btn-abnormal'),
  btnResume: document.getElementById('btn-resume'),
  btnEmergencyCall: document.getElementById('btn-emergency-call'),
  simRiskGrid: document.getElementById('sim-risk-grid'),
  simTelemetry: document.querySelector('.sim-telemetry'),
  simRiskMeterFill: document.getElementById('sim-risk-meter-fill'),
  simRiskVisualFill: document.getElementById('sim-risk-visual-fill'),
  simClock: document.getElementById('sim-clock'), simConnection: document.getElementById('sim-connection'), simStatus: document.getElementById('sim-status'), simStatusIcon: document.getElementById('sim-status-icon'), simStatusTitle: document.getElementById('sim-status-title'), simStatusCopy: document.getElementById('sim-status-copy'), simRiskLevel: document.getElementById('sim-risk-level'), simWindowState: document.getElementById('sim-window-state'), simBuffer: document.getElementById('sim-buffer'), simHealthScore: document.getElementById('sim-health-score'), simEnvironmentScore: document.getElementById('sim-environment-score'), simFinalScore: document.getElementById('sim-final-score'), simHr: document.getElementById('sim-hr'), simActivity: document.getElementById('sim-activity'), simTemperature: document.getElementById('sim-temperature'), simRespiration: document.getElementById('sim-respiration'), simEda: document.getElementById('sim-eda'), simEcg: document.getElementById('sim-ecg'), simWeatherMode: document.getElementById('sim-weather-mode'), simWeatherTemperature: document.getElementById('sim-weather-temperature'), simWeatherHumidity: document.getElementById('sim-weather-humidity'), simWeatherAqi: document.getElementById('sim-weather-aqi'), simWeatherUv: document.getElementById('sim-weather-uv'), simMse: document.getElementById('sim-mse'), simRiskButton: document.getElementById('sim-risk-button'), simAnomalyButton: document.getElementById('sim-anomaly-button'), simResumeButton: document.getElementById('sim-resume-button'), simEmergencyButton: document.getElementById('sim-emergency-button'), simResultFragment: document.getElementById('sim-result-fragment'), simResultScore: document.getElementById('sim-result-score'), simResultStatus: document.getElementById('sim-result-status'), simResultReason: document.getElementById('sim-result-reason'), simSosMessage: document.getElementById('sim-sos-message'), simResultClose: document.getElementById('sim-result-close'), simResultBack: document.getElementById('sim-result-back'),
};

function fmtTime(d){
  return d.toLocaleTimeString('en-IN', { hour:'2-digit', minute:'2-digit', second:'2-digit', hour12:false });
}
function clamp(v,min,max){ return Math.max(min, Math.min(max, v)); }
function capitalize(s){ return s.charAt(0).toUpperCase()+s.slice(1); }

/* ---------------------------------------------------------------
   2. LIVE SENSOR STREAM → FLASK

   The WEBSITE is the source of sensor samples.

   Every second:

       generateSensorSample()
              ↓
       POST /api/data
              ↓
       Flask rolling 60s window
              ↓
       ONNX inference
              ↓
       result returned
              ↓
       updateFromBackend(result)

   The website NEVER calculates the anomaly score.
   --------------------------------------------------------------- */

let isSending = false;

// Flask remains the authority for the actual anomaly decision.
let sensorMode = 'normal';


/* ---------------------------------------------------------------
   Generate ONE sensor sample

  These are plausible wearable readings in physical units:
  bpm, g, degrees C, breaths/min, microsiemens and millivolts.

   Model feature order:

       [HR/BVP, ACC, TEMP, RESP, EDA, ECG]

   Consecutive samples are kept smooth so they resemble
   a continuous physiological signal.
   --------------------------------------------------------------- */

const sensorState = {
  hr: 72,
  acc: 0.02,
  temp: 36.7,
  resp: 15,
  eda: 2.5,
  ecg: 0.02
};

const physicalMeans = {
  'HR/BVP': 72,
  ACC: 0,
  TEMP: 36.7,
  RESP: 15,
  EDA: 2.5,
  ECG: 0
};

const physicalStds = {
  'HR/BVP': 15,
  ACC: 0.5,
  TEMP: 0.8,
  RESP: 5,
  EDA: 3,
  ECG: 1
};

function getPhysicalValues(data){
  if (data.display_values) return data.display_values;
  if (!data.values) return null;

  const values = data.values;
  const looksStandardized = Math.abs(Number(values['HR/BVP'])) < 10;
  return Object.fromEntries(
    Object.keys(physicalMeans).map((feature) => {
      const value = Number(values[feature]);
      return [
        feature,
        looksStandardized
          ? value * physicalStds[feature] + physicalMeans[feature]
          : value
      ];
    })
  );
}


/* ---------------------------------------------------------------
   Gaussian random number
   --------------------------------------------------------------- */

function randomNormal() {

  let u = 0;
  let v = 0;

  while (u === 0) {
    u = Math.random();
  }

  while (v === 0) {
    v = Math.random();
  }

  return Math.sqrt(
    -2 * Math.log(u)
  ) * Math.cos(
    2 * Math.PI * v
  );
}


/* ---------------------------------------------------------------
   Smooth sensor value
   --------------------------------------------------------------- */

function smoothValue(current, noise, min, max) {

  const next =
    current * 0.92 +
    noise * 0.08;

  return clamp(
    next,
    min,
    max
  );
}


/* ---------------------------------------------------------------
   Generate ONE six-feature sensor vector
   --------------------------------------------------------------- */

function generateSensorSample() {

  /* ---------------- NORMAL MODE ---------------- */

  if (sensorMode === 'normal') {

    sensorState.hr = smoothValue(
      sensorState.hr,
      72 + randomNormal() * 8,
      55,
      100
    );

    sensorState.acc = smoothValue(
      sensorState.acc,
      randomNormal() * 0.15,
      -0.4,
      0.4
    );

    sensorState.temp = smoothValue(
      sensorState.temp,
      36.7 + randomNormal() * 0.35,
      36.0,
      37.8
    );

    sensorState.resp = smoothValue(
      sensorState.resp,
      15 + randomNormal() * 2,
      10,
      24
    );

    sensorState.eda = smoothValue(
      sensorState.eda,
      2.5 + randomNormal() * 1.2,
      0.5,
      8
    );

    sensorState.ecg = smoothValue(
      sensorState.ecg,
      randomNormal() * 0.08,
      -1.0,
      1.0
    );

  }

  /* ---------------- ABNORMAL MODE ---------------- */

  else {

    // Move gradually toward abnormal targets instead of jumping in one tick.
    sensorState.hr = smoothValue(sensorState.hr, 145 + randomNormal() * 12, 55, 180);
    sensorState.acc = smoothValue(sensorState.acc, 1.8 + randomNormal() * 0.35, -0.4, 3.0);
    sensorState.temp = smoothValue(sensorState.temp, 39.2 + randomNormal() * 0.4, 36.0, 41.0);
    sensorState.resp = smoothValue(sensorState.resp, 30 + randomNormal() * 3, 10, 45);
    sensorState.eda = smoothValue(sensorState.eda, 14 + randomNormal() * 2, 0.5, 20);
    sensorState.ecg = smoothValue(sensorState.ecg, -2.2 + randomNormal() * 0.25, -4.0, 1.0);
  }


  /* -------------------------------------------------------------
     IMPORTANT:
     These names MUST match the Flask feature names.
     ------------------------------------------------------------- */

  return {

    "HR/BVP": sensorState.hr,

    "ACC": sensorState.acc,

    "TEMP": sensorState.temp,

    "RESP": sensorState.resp,

    "EDA": sensorState.eda,

    "ECG": sensorState.ecg
  };
}


/* ---------------------------------------------------------------
   CONNECTION STATUS
   --------------------------------------------------------------- */

function setConnected(isConnected) {

  const wasConnected =
    deviceState.connected;

  deviceState.connected =
    isConnected;


  /* Connection text */

  if (els.connStatus) {

    els.connStatus.textContent =
      isConnected
        ? '● CONNECTED'
        : '● BACKEND OFFLINE';
  }


  /* Connection indicator */

  if (els.connDot) {

    els.connDot.classList.toggle(
      'is-offline',
      !isConnected
    );
  }


  /* Device visual state */

  const deviceElement =
    document.getElementById('device');

  if (deviceElement) {

    deviceElement.classList.toggle(
      'is-offline',
      !isConnected
    );
  }


  /* Buffer status */

  if (!isConnected) {

    if (els.bufferBadge) {

      els.bufferBadge.textContent =
        'BACKEND OFFLINE · RETRYING…';
    }

  }
}


/* ---------------------------------------------------------------
   RECEIVE RESULT FROM FLASK

   Flask is the source of truth.

   Flask returns something like:

   {
       status: "NORMAL",
       anomaly_score: 0.42,
       threshold: 1.78,
       window_ready: true,
       seconds_collected: 60,
       mode: "normal",
       values: {...}
   }
   --------------------------------------------------------------- */

function updateFromBackend(data) {

  if (!data) {
    return;
  }


  /* -------------------------------------------------------------
     Determine anomaly state

     We accept both:
       NORMAL
       ANOMALY

     and lowercase versions.
     ------------------------------------------------------------- */

  const status =
    String(data.status || '').toUpperCase();

  const isAbnormal = status === 'ABNORMAL' || status === 'ANOMALY';
  const isWarning = status === 'WARNING';
  const isEmergency = status === 'EMERGENCY' || status === 'CRITICAL';
  const enteredEmergency = isEmergency && deviceState.status !== 'emergency';


  /* -------------------------------------------------------------
     Persist this result onto deviceState.

     THIS WAS THE MISSING PIECE: updateDashboard()/updateWatchFace()
     only ever read from `deviceState` (never from arguments), but
     nothing here was writing to it. That's why the sensor squares
     stayed on "—" even though the backend was responding fine.
     ------------------------------------------------------------- */

  deviceState.statusRaw = status;
  deviceState.windowReady = !!data.window_ready;
  deviceState.secondsCollected = data.seconds_collected || 0;

  deviceState.anomalyScore =
    (data.anomaly_score !== undefined && data.anomaly_score !== null)
      ? Number(data.anomaly_score)
      : null;

  deviceState.threshold =
    (data.threshold !== undefined && data.threshold !== null)
      ? Number(data.threshold)
      : deviceState.threshold;

  deviceState.values = data.values || null;
  deviceState.displayValues = getPhysicalValues(data);
  deviceState.environment = data.environment || null;
  deviceState.environmentMode = data.environment_mode || null;
  deviceState.updatedAt = new Date();


  /* -------------------------------------------------------------
     Update main health state

    Flask only reports NORMAL / ABNORMAL, while the UI has a
    normal/warning/emergency scale. MSE above 3 is emergency.
     ------------------------------------------------------------- */

  if (typeof setHealthState === 'function') {

    setHealthState(
      isEmergency ? 'emergency' : isWarning || isAbnormal ? 'warning' : 'normal'
    );
  }

  if (els.btnEmergencyCall) {
    els.btnEmergencyCall.disabled = !isEmergency;
    els.btnEmergencyCall.classList.toggle('is-active', isEmergency);
    if (isEmergency && !deviceState.emergencyAlertSent) {
      els.btnEmergencyCall.textContent = 'Sending SOS alert...';
    } else if (!isEmergency) {
      els.btnEmergencyCall.textContent = 'No active emergency';
    }
  }

  if (enteredEmergency && !deviceState.emergencyAlertSent) {
    sendSosAlert();
  }
  if (!isEmergency) {
    deviceState.emergencyAlertSent = false;
  }

  if (els.mlMse) {

    if (data.anomaly_score !== undefined &&
        data.anomaly_score !== null) {

      els.mlMse.textContent =
        Number(
          data.anomaly_score
        ).toFixed(4);

    } else {

      els.mlMse.textContent =
        '--';
    }
  }


  /* -------------------------------------------------------------
     Update threshold
     ------------------------------------------------------------- */

  if (els.mlThreshold) {

    if (data.threshold !== undefined &&
        data.threshold !== null) {

      els.mlThreshold.textContent =
        Number(
          data.threshold
        ).toFixed(2);

    } else {

      els.mlThreshold.textContent =
        '1.78';
    }
  }


  /* -------------------------------------------------------------
     Update ML status chip
     ------------------------------------------------------------- */

  if (els.mlChip) {

    els.mlChip.textContent = status || 'WAITING';
    els.mlChip.classList.toggle('is-warning', (isWarning || isAbnormal) && !isEmergency);
    els.mlChip.classList.toggle('is-emergency', isEmergency);
  }

  updateRiskScores(data);


  /* -------------------------------------------------------------
     Update rolling buffer status
     ------------------------------------------------------------- */

  if (els.bufferBadge) {

    if (data.window_ready) {

      els.bufferBadge.textContent =
        '60s WINDOW READY';

    } else {

      const collected =
        data.seconds_collected || 0;

      els.bufferBadge.textContent =
        `${collected}/60s BUFFERING`;
    }
  }


  /* -------------------------------------------------------------
     Update dashboard sensor values + watch screen.

     Both functions read from `deviceState` (set above), not from
     arguments — call them plainly so that's not misleading.
     ------------------------------------------------------------- */

  if (typeof updateDashboard === 'function') {
    updateDashboard();
  }

  if (typeof updateWatchFace === 'function') {
    updateWatchFace();
  }
}

function formatBackendScore(value){
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(4) : '—';
}

function updateRiskScores(data){
  els.healthScore.textContent = formatBackendScore(data.health_score);
  els.environmentScore.textContent = formatBackendScore(data.environment_score);
  els.finalRiskScore.textContent = formatBackendScore(data.final_risk_score);
  if (data.risk_reason) {
    els.alertLine2.textContent = data.risk_reason;
  }
}

function formatWeatherValue(value, decimals, unit){ const number = Number(value); return Number.isFinite(number) ? `${number.toFixed(decimals)}${unit}` : '—'; }
function updateMobileSimulator(data){
  if (!els.simStatus || !data) return;
  const status = String(data.status || 'WAITING').toUpperCase();
  const isEmergency = status === 'EMERGENCY' || status === 'CRITICAL';
  const isWarning = status === 'WARNING' || status === 'ABNORMAL' || status === 'ANOMALY';
  const values = deviceState.displayValues || {};
  const environment = deviceState.environment || {};
  const riskScore = Number(data.final_risk_score);
  const riskAlpha = (0.08 + Math.max(0, Math.min(1, Number.isFinite(riskScore) ? riskScore : 0)) * 0.24).toFixed(3);
  const set = (element, value) => { if (element) element.textContent = value; };
  const physical = (key, decimals, unit) => { const number = Number(values[key]); return Number.isFinite(number) ? `${number.toFixed(decimals)}${unit}` : '—'; };
  els.simStatus.classList.toggle('is-warning', isWarning && !isEmergency); els.simStatus.classList.toggle('is-emergency', isEmergency);
  els.simStatus.style.setProperty('--risk-alpha', riskAlpha);
  if (els.simTelemetry) {
    els.simTelemetry.classList.toggle('is-normal', !isWarning && !isEmergency);
    els.simTelemetry.classList.toggle('is-warning', isWarning && !isEmergency);
    els.simTelemetry.classList.toggle('is-critical', isEmergency);
    els.simTelemetry.style.setProperty('--risk-alpha', riskAlpha);
  }
  set(els.simStatusIcon, isEmergency || isWarning ? '!' : '✓'); set(els.simStatusTitle, isEmergency ? 'EMERGENCY' : isWarning ? 'WARNING' : status); set(els.simStatusCopy, isEmergency ? 'Critical abnormality detected' : isWarning ? 'Abnormal pattern detected' : 'All vitals within baseline');
  set(els.simBuffer, `${data.seconds_collected || 0}/60s`); set(els.simWindowState, `${data.seconds_collected || 0}/60s`); set(els.simRiskLevel, isEmergency ? 'CRITICAL' : isWarning ? 'ELEVATED' : 'NORMAL'); set(els.simHealthScore, formatBackendScore(data.health_score)); set(els.simEnvironmentScore, formatBackendScore(data.environment_score)); set(els.simFinalScore, formatBackendScore(data.final_risk_score));
  const normalizedRisk = Math.max(0, Math.min(1, Number.isFinite(riskScore) ? riskScore : 0));
  if (els.simRiskMeterFill) els.simRiskMeterFill.style.width = `${Math.round(normalizedRisk * 100)}%`;
  if (els.simRiskVisualFill) els.simRiskVisualFill.style.width = `${Math.round(normalizedRisk * 100)}%`;
  set(els.simHr, physical('HR/BVP', 0, ' BPM')); set(els.simActivity, physical('ACC', 2, ' g')); set(els.simTemperature, physical('TEMP', 1, '°C')); set(els.simRespiration, physical('RESP', 0, ' br/min')); set(els.simEda, physical('EDA', 2, ' µS')); set(els.simEcg, physical('ECG', 2, ' mV'));
  set(els.simWeatherMode, String(deviceState.environmentMode || 'NORMAL').toUpperCase()); set(els.simWeatherTemperature, formatWeatherValue(environment.temperature, 1, '°C')); set(els.simWeatherHumidity, formatWeatherValue(environment.humidity, 0, '%')); set(els.simWeatherAqi, formatWeatherValue(environment.aqi, 0, '')); set(els.simWeatherUv, formatWeatherValue(environment.uv, 1, '')); set(els.simMse, `MSE ${formatBackendScore(data.anomaly_score)}`); set(els.simClock, fmtTime(deviceState.updatedAt).slice(0, 5));
  els.simConnection.classList.toggle('is-offline', !deviceState.connected); els.simConnection.innerHTML = `<i></i> ${deviceState.connected ? 'LIVE' : 'OFFLINE'}`;
  if (els.simEmergencyButton) { els.simEmergencyButton.disabled = !isEmergency; els.simEmergencyButton.classList.toggle('is-active', isEmergency); els.simEmergencyButton.textContent = isEmergency ? 'Send SOS alert' : 'No active emergency'; }
  updateMobileApp(data);
}

/* ---------------------------------------------------------------
   SEND ONE SENSOR SAMPLE TO FLASK

   This runs once every second.

   IMPORTANT:
   The frontend does NOT calculate MSE.
   It only sends sensor data.

   Flask:
       receives data
       stores rolling 60 seconds
       runs ONNX
       calculates MSE
       compares against 1.78
       returns result
   --------------------------------------------------------------- */

async function sendSensorSample() {

  /* Prevent overlapping requests */

  if (isSending) {
    return;
  }

  isSending = true;


  try {

    /* -----------------------------------------------------------
       Generate current sensor vector
       ----------------------------------------------------------- */

    const values =
      generateSensorSample();


    /* -----------------------------------------------------------
       Send to Flask
       ----------------------------------------------------------- */

    const response =
      await fetch(
        '/api/data',
        {
          method: 'POST',

          headers: {
            'Content-Type':
              'application/json'
          },

          cache: 'no-store',

          body: JSON.stringify({
            values: values
          })
        }
      );


    /* -----------------------------------------------------------
       Check HTTP response
       ----------------------------------------------------------- */

    if (!response.ok) {

      throw new Error(
        `Backend HTTP ${response.status}`
      );
    }


      setConnected(true);

     // Read the authoritative latest result, including all risk fields.
     await fetchBackendStatus();

  }


  catch (err) {

    console.error(
      'Unable to send sensor data:',
      err
    );


    /* Backend unavailable */

    setConnected(false);

  }


  finally {

    isSending = false;
  }
}

async function fetchBackendStatus(){
  const response = await fetch('/api/status', { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`Backend status HTTP ${response.status}`);
  }
  updateFromBackend(await response.json());
}

async function assessRisk(){
  if (isSending || !els.simRiskButton) return;
  const originalLabel = els.simRiskButton.textContent; els.simRiskButton.disabled = true; els.simRiskButton.textContent = 'Assessing...';
  try {
    const response = await fetch('/api/risk-assessment', { method:'POST', headers:{'Content-Type':'application/json'}, cache:'no-store', body:JSON.stringify({ values:generateSensorSample() }) });
    const result = await response.json(); if (!response.ok) throw new Error(result.error || `Risk assessment HTTP ${response.status}`);
    const riskThreshold = Number(result.risk_threshold);
    const overallRisk = Number(result.final_risk_score);
    const thresholdExceeded = result.window_ready && Number.isFinite(overallRisk) && Number.isFinite(riskThreshold) && overallRisk > riskThreshold;
    setConnected(true); updateFromBackend(result);
    if (thresholdExceeded) await simulateAbnormality(false);
    showRiskResult(result, thresholdExceeded);
    els.simRiskButton.textContent = result.window_ready ? `Final risk ${formatBackendScore(result.final_risk_score)}` : `Collecting ${result.seconds_collected}/60s`;
  } catch (err) { setConnected(false); els.simRiskButton.textContent = 'Assessment failed'; throw err; }
  finally { setTimeout(()=>{ els.simRiskButton.disabled=false; els.simRiskButton.textContent=originalLabel; },1800); }
}
function showRiskResult(result, abnormalityNotified = false){
  if (!els.simResultFragment) return;
  veRecordAssessment(result, 'Assessment');
  const status = String(result.status || 'COLLECTING').toUpperCase(); const critical = status === 'CRITICAL' || status === 'EMERGENCY'; const warning = status === 'WARNING' || status === 'ABNORMAL' || status === 'ANOMALY';
  const riskScore = Number(result.final_risk_score);
  const normalizedRisk = Math.max(0, Math.min(1, Number.isFinite(riskScore) ? riskScore : 0));
  const riskAlpha = (0.08 + normalizedRisk * 0.20).toFixed(3);
  if (els.simRiskGrid) {
    els.simRiskGrid.classList.toggle('is-normal', !warning && !critical);
    els.simRiskGrid.classList.toggle('is-warning', warning && !critical);
    els.simRiskGrid.classList.toggle('is-critical', critical);
    els.simRiskGrid.style.setProperty('--risk-alpha', riskAlpha);
  }
  if (els.simRiskVisualFill) {
    els.simRiskVisualFill.style.width = `${Math.round(normalizedRisk * 100)}%`;
  }
  els.simResultFragment.classList.toggle('is-warning', warning && !critical); els.simResultFragment.classList.toggle('is-critical', critical); els.simResultScore.textContent = result.window_ready ? formatBackendScore(result.final_risk_score) : `— (${result.seconds_collected || 0}/60s)`; els.simResultStatus.textContent=status; els.simResultReason.textContent=result.risk_reason || 'Assessment recorded.'; els.simSosMessage.textContent=abnormalityNotified ? 'Overall risk exceeded the threshold. Abnormality simulation enabled.' : critical ? 'Critical risk detected. Send SOS alert now.' : warning ? 'Elevated risk detected. Keep responder under observation.' : 'Assessment recorded. Continue monitoring.'; els.simResultFragment.classList.add('is-visible'); els.simResultFragment.setAttribute('aria-hidden','false');
}
function hideRiskResult(){ if (!els.simResultFragment) return; els.simResultFragment.classList.remove('is-visible'); els.simResultFragment.setAttribute('aria-hidden','true'); }

/* ---------------------------------------------------------------
   3. HEALTH-STATE ENGINE — the one place status text/colour comes from
   --------------------------------------------------------------- */
function setHealthState(state){
  deviceState.status = state;

  els.statusBanner.classList.toggle('is-warning', state === 'warning');
  els.statusBanner.classList.toggle('is-emergency', state === 'emergency');
  els.alertPanel.classList.toggle('is-warning', state === 'warning');
  els.alertPanel.classList.toggle('is-emergency', state === 'emergency');

  if (state === 'normal'){
    els.statusIcon.textContent = '\u2713';
    els.statusT1.textContent = 'NORMAL';
    els.statusT2.textContent = 'All vitals within expected baseline';
    els.alertLine1.textContent = '\u2713 No abnormality detected';
    els.alertLine2.textContent = 'Backend status: NORMAL';
  } else if (state === 'warning'){
    els.statusIcon.textContent = '\u26A0';
    els.statusT1.textContent = 'WARNING';
    els.statusT2.textContent = 'Abnormal pattern detected';
    els.alertLine1.textContent = '\u26A0 Abnormal pattern detected';
    els.alertLine2.textContent = 'Backend status: WARNING';
  } else {
    els.statusIcon.textContent = '\u26A0';
    els.statusT1.textContent = 'EMERGENCY';
    els.statusT2.textContent = 'Critical abnormality detected';
    els.alertLine1.textContent = '\u26A0 CRITICAL ABNORMALITY';
    els.alertLine2.textContent = 'Backend status: EMERGENCY';
  }
}

/* ---------------------------------------------------------------
   4. DASHBOARD — reads only deviceState, which is only ever written
   by updateFromBackend()
   --------------------------------------------------------------- */
function updateDashboard(){
  const dv = deviceState.displayValues;
  const v = deviceState.values;
  const numberOrNull = (value) => Number.isFinite(Number(value)) ? Number(value) : null;

  if (dv){
    const heartRate = numberOrNull(dv['HR/BVP']);
    const activity = numberOrNull(dv.ACC);
    const temperature = numberOrNull(dv.TEMP);
    const respiration = numberOrNull(dv.RESP);
    const eda = numberOrNull(dv.EDA);
    const ecg = numberOrNull(dv.ECG);
    els.metricHr.textContent = heartRate !== null ? `${heartRate.toFixed(0)} BPM` : '—';
    els.metricActivity.textContent = activity !== null ? `${activity.toFixed(2)} g` : '—';
    els.metricTemperature.textContent = temperature !== null ? `${temperature.toFixed(1)}°C` : '—';
    els.metricRespiration.textContent = respiration !== null ? `${respiration.toFixed(0)} br/min` : '—';
    els.metricEda.textContent = eda !== null ? `${eda.toFixed(2)} µS` : '—';
    els.metricEcg.textContent = ecg !== null ? `${ecg.toFixed(2)} mV` : '—';
    pushGraphPoint('hr', heartRate);
    pushGraphPoint('respiration', respiration);
    pushGraphPoint('ecg', ecg);
  } else if (v){
    // Model-space values are never shown as physical measurements.
    els.metricHr.textContent = '—';
    els.metricActivity.textContent = '—';
    els.metricTemperature.textContent = '—';
    els.metricRespiration.textContent = '—';
    els.metricEda.textContent = '—';
    els.metricEcg.textContent = '—';
    pushGraphPoint('hr', v['HR/BVP']);
    pushGraphPoint('respiration', v['RESP']);
    pushGraphPoint('ecg', v['ECG']);
  } else {
    els.metricHr.textContent = '—';
    els.metricActivity.textContent = '—';
    els.metricTemperature.textContent = '—';
    els.metricRespiration.textContent = '—';
    els.metricEda.textContent = '—';
    els.metricEcg.textContent = '—';
  }

  els.batteryStatus.textContent = 'Battery —'; // not part of the current Flask contract
  els.clock.textContent = `Last update ${fmtTime(deviceState.updatedAt)}`;

  drawSparkline(els.graphHr, graphHistories.hr, '#3ddc8f');
  drawSparkline(els.graphResp, graphHistories.respiration, '#7bb8ff');
  drawSparkline(els.graphEcg, graphHistories.ecg, '#f2b84b');

  updateWeather();
  updateMobileSimulator({status:deviceState.statusRaw, anomaly_score:deviceState.anomalyScore, seconds_collected:deviceState.secondsCollected, health_score:els.healthScore.textContent, environment_score:els.environmentScore.textContent, final_risk_score:els.finalRiskScore.textContent});
}

function updateWeather(){
  const environment = deviceState.environment;
  if (!environment) return;

  const formatValue = (value, decimals, unit) => {
    const number = Number(value);
    return Number.isFinite(number) ? `${number.toFixed(decimals)}${unit}` : '—';
  };

  els.weatherTemperature.textContent = formatValue(environment.temperature, 1, '°C');
  els.weatherHumidity.textContent = formatValue(environment.humidity, 0, '%');
  els.weatherAqi.textContent = formatValue(environment.aqi, 0, '');
  els.weatherPm25.textContent = formatValue(environment.pm2_5, 1, '');
  els.weatherPm10.textContent = formatValue(environment.pm10, 1, '');
  els.weatherUv.textContent = formatValue(environment.uv, 1, '');
  els.weatherLocation.textContent = environment.location || '—';
  els.weatherSource.textContent = `Source ${environment.source || '—'}`;
  els.weatherMode.textContent = String(deviceState.environmentMode || 'LIVE').toUpperCase();
}

function pushGraphPoint(key, value){
  if (typeof value !== 'number') return;
  const arr = graphHistories[key];
  arr.push(value);
  if (arr.length > GRAPH_LEN) arr.shift();
}
function syncCanvasSize(canvas){
  if (!canvas) return;
  canvas.width = canvas.clientWidth;
  canvas.height = canvas.clientHeight;
}
function drawSparkline(canvas, data, color){
  if (!canvas || data.length < 2) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0,0,w,h);
  const min = Math.min(...data), max = Math.max(...data);
  const range = (max-min) || 1;
  ctx.beginPath();
  data.forEach((v,i)=>{
    const x = (i/(GRAPH_LEN-1))*w;
    const y = h - ((v-min)/range)*h*0.78 - h*0.11;
    if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  });
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round'; ctx.lineCap = 'round';
  ctx.stroke();
  ctx.lineTo(w,h); ctx.lineTo(0,h); ctx.closePath();
  ctx.fillStyle = color + '22';
  ctx.fill();
}

/* ---------------------------------------------------------------
   5. BUTTONS — these only ever ask the backend to change mode
   --------------------------------------------------------------- */
async function simulateAbnormality(sendSample = true){
  sensorMode = 'abnormal';
  try{
    await fetch('/api/simulate-abnormal', { method:'POST', headers:{ 'Content-Type':'application/json' } });
  } catch (err){ /* sendSensorSample() below will surface the offline state */ }
  if (sendSample) await sendSensorSample();
}
async function resumeNormal(){
  sensorMode = 'normal';
  sensorState.hr = 72;
  sensorState.acc = 0.02;
  sensorState.temp = 36.7;
  sensorState.resp = 15;
  sensorState.eda = 2.5;
  sensorState.ecg = 0.02;
  try{
    await fetch('/api/reset', { method:'POST', headers:{ 'Content-Type':'application/json' } });
  } catch (err){ /* sendSensorSample() below will surface the offline state */ }
  await sendSensorSample();
}

async function sendSosAlert(){
  if (!els.btnEmergencyCall) return;

  deviceState.emergencyAlertSent = true;
  els.btnEmergencyCall.disabled = true;
  els.btnEmergencyCall.textContent = 'Sending SOS alert...';

  try {
    const response = await fetch('/api/send-sos', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status: deviceState.statusRaw || 'UNKNOWN' })
    });

    const result = await response.json();
    if (!response.ok) throw new Error(result.message || 'SOS request failed');

    els.btnEmergencyCall.textContent = 'SOS sent - mark false alarm';
    veRecordAlert('sos', 'SOS alert sent', result.message || 'Your emergency contact has been notified.');
  } catch (err) {
    console.error('Unable to send SOS alert:', err);
    els.btnEmergencyCall.textContent = 'SOS alert failed - retrying';
    deviceState.emergencyAlertSent = false;
    return;
  }

  els.btnEmergencyCall.disabled = false;
}

async function reportFalseAlarm(){
  if (!els.btnEmergencyCall || deviceState.status !== 'emergency') return;

  els.btnEmergencyCall.disabled = true;
  els.btnEmergencyCall.textContent = 'Reporting false alarm...';

  try {
    const response = await fetch('/api/report-false-alarm', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });
    if (!response.ok) throw new Error(`False alarm HTTP ${response.status}`);
    els.btnEmergencyCall.textContent = 'False alarm reported';
    veRecordAlert('info', 'False alarm reported', 'The emergency was marked as a false alarm.');
  } catch (err) {
    console.error('Unable to report false alarm:', err);
    els.btnEmergencyCall.textContent = 'False alarm report failed';
    els.btnEmergencyCall.disabled = false;
  }
}

function flashButton(btn){
  btn.style.transform = 'scale(0.97)';
  setTimeout(()=> btn.style.transform = '', 140);
}

async function runSimulatorAction(button, action, busyLabel, successLabel){
  if (!button || button.disabled) return;
  const originalLabel = button.textContent; button.disabled=true; button.textContent=busyLabel;
  try { await action(); button.textContent=successLabel; if (button === els.simAnomalyButton) showRiskResult({ status:deviceState.statusRaw, window_ready:deviceState.windowReady, seconds_collected:deviceState.secondsCollected, final_risk_score:els.finalRiskScore.textContent, health_score:els.healthScore.textContent, environment_score:els.environmentScore.textContent, risk_reason:deviceState.statusRaw === 'ABNORMAL' ? 'Abnormal pattern detected' : 'Assessment recorded.' }); } catch (err) { button.textContent='Action failed'; console.error('Simulator action failed:',err); }
  finally { setTimeout(()=>{ button.disabled = button === els.simEmergencyButton ? deviceState.status !== 'emergency' : false; button.textContent=originalLabel; },1500); }
}

els.btnAbnormal.addEventListener('click', ()=>{ simulateAbnormality(); flashButton(els.btnAbnormal); });
els.btnResume.addEventListener('click', ()=>{ resumeNormal(); flashButton(els.btnResume); });
els.btnEmergencyCall.addEventListener('click', ()=>{ reportFalseAlarm(); flashButton(els.btnEmergencyCall); });
els.simRiskButton.addEventListener('click', ()=>{ assessRisk().catch(err=>console.error('Risk assessment failed:',err)); flashButton(els.simRiskButton); });
els.simAnomalyButton.addEventListener('click', ()=>{ runSimulatorAction(els.simAnomalyButton,simulateAbnormality,'Simulating...','Abnormality enabled'); flashButton(els.simAnomalyButton); });
els.simResumeButton.addEventListener('click', ()=>{ runSimulatorAction(els.simResumeButton,resumeNormal,'Resetting...','Normal mode enabled'); flashButton(els.simResumeButton); });
els.simEmergencyButton.addEventListener('click', ()=>{ runSimulatorAction(els.simEmergencyButton,sendSosAlert,'Sending SOS...','SOS sent'); flashButton(els.simEmergencyButton); });
els.simResultClose.addEventListener('click',hideRiskResult); els.simResultBack.addEventListener('click',hideRiskResult);

/* ---------------------------------------------------------------
   MOBILE APP SHELL — navigation, Health History and Alerts.
   Presentation-only: everything here is derived from the same
   backend results the simulator already receives. SOS reuses the
   existing sendSosAlert() flow (POST /api/send-sos).
   --------------------------------------------------------------- */
const veApp = { view:'home', history:[], alerts:[], unread:0, lastKind:null, lastRecorded:0, sosTimer:null };
const veLabels = { normal:'Normal', warning:'Elevated', critical:'Critical' };
const veIcons = { normal:'✓', warning:'!', critical:'!', sos:'SOS', info:'i' };
const $ve = (id) => document.getElementById(id);

function veKind(status){
  const s = String(status || '').toUpperCase();
  if (s === 'EMERGENCY' || s === 'CRITICAL') return 'critical';
  if (s === 'WARNING' || s === 'ABNORMAL' || s === 'ANOMALY') return 'warning';
  return 'normal';
}
function veEsc(text){ return String(text).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch])); }
function veScore(value){ const n = Number(value); return Number.isFinite(n) ? n.toFixed(2) : null; }

function veShowView(name){
  veApp.view = name;
  document.querySelectorAll('.ve-view').forEach(view => { view.hidden = view.dataset.view !== name; });
  document.querySelectorAll('.ve-nav-btn').forEach(btn => {
    const active = btn.dataset.view === name;
    btn.classList.toggle('is-active', active);
    if (active) btn.setAttribute('aria-current', 'page'); else btn.removeAttribute('aria-current');
  });
  const scroller = document.querySelector('.ve-views'); if (scroller) scroller.scrollTop = 0;
  if (name === 'alerts') { veApp.unread = 0; veRenderBadge(); }
  if (name !== 'sos') veDisarmSos();
}

function veRecordAssessment(result, source){
  if (!result) return;
  const kind = veKind(result.status);
  const ready = result.window_ready !== false && Number.isFinite(Number(result.final_risk_score));
  veApp.history.unshift({
    time:new Date(), kind, source:source || 'Assessment', ready, seconds:result.seconds_collected || 0,
    score:veScore(result.final_risk_score), health:veScore(result.health_score), env:veScore(result.environment_score),
    note:result.risk_reason || (kind === 'normal' ? 'No significant anomaly detected' : 'Abnormal pattern detected')
  });
  veApp.history.length = Math.min(veApp.history.length, 30);
  veApp.lastRecorded = Date.now();
  veRenderHistory();
}

function veRecordAlert(kind, title, detail){
  veApp.alerts.unshift({ time:new Date(), kind, title, detail });
  veApp.alerts.length = Math.min(veApp.alerts.length, 40);
  if (veApp.view !== 'alerts') veApp.unread += 1;
  if (kind === 'sos' && $ve('ve-sos-last')) $ve('ve-sos-last').textContent = `Last SOS sent at ${fmtTime(new Date())}.`;
  veRenderAlerts(); veRenderBadge();
}

function veRenderBadge(){
  const badge = $ve('ve-alert-badge'); if (!badge) return;
  badge.textContent = veApp.unread > 9 ? '9+' : String(veApp.unread);
  badge.hidden = veApp.unread === 0;
}

function veRenderHistory(){
  const list = $ve('ve-history-list'); if (!list) return;
  const items = veApp.history;
  $ve('ve-history-count').textContent = items.length;
  $ve('ve-history-flagged').textContent = items.filter(e => e.kind !== 'normal').length;
  const latest = items.find(e => e.score);
  $ve('ve-history-latest').textContent = latest ? latest.score : '—';
  list.innerHTML = items.length ? items.map(e => `
    <li class="ve-entry is-${e.kind}">
      <span class="ve-dot">${veIcons[e.kind]}</span>
      <span class="ve-entry-title">${veLabels[e.kind]}</span>
      <span class="ve-entry-score">${e.ready && e.score ? e.score : '—'}</span>
      <span class="ve-entry-meta">${veEsc(e.source)} · ${fmtTime(e.time)}</span>
      <span class="ve-entry-body">${e.ready ? `${veEsc(e.note)}${e.health && e.env ? ` · Health ${e.health} · Environment ${e.env}` : ''}` : `Collecting data (${e.seconds}/60s)`}</span>
    </li>`).join('') : '<li class="ve-empty">No checks yet. Run a risk assessment from Home, or wait for the next automatic check.</li>';
}

function veRenderAlerts(){
  const list = $ve('ve-alert-list'); if (!list) return;
  list.innerHTML = veApp.alerts.length ? veApp.alerts.map(a => `
    <li class="ve-entry is-${a.kind === 'info' ? 'normal' : a.kind}">
      <span class="ve-dot">${veIcons[a.kind] || '!'}</span>
      <span class="ve-entry-title">${veEsc(a.title)}</span>
      <span></span>
      <span class="ve-entry-meta">${fmtTime(a.time)}</span>
      <span class="ve-entry-body">${veEsc(a.detail || '')}</span>
    </li>`).join('') : '<li class="ve-empty">No alerts. Abnormal readings and SOS events will appear here.</li>';
}

/* Called from updateMobileSimulator() on every backend update. */
function updateMobileApp(data){
  const kind = veKind(deviceState.statusRaw);
  const sosNav = document.querySelector('.ve-nav-sos');
  if (sosNav) sosNav.classList.toggle('is-emergency', kind === 'critical');
  const sosCard = $ve('ve-sos-card'); if (sosCard) sosCard.classList.toggle('is-emergency', kind === 'critical');
  const sosState = $ve('ve-sos-state');
  if (sosState) sosState.textContent = kind === 'critical'
    ? 'Critical risk detected. Your emergency contact is being alerted.'
    : "You're not in an emergency. Use SOS only if you need help right now.";

  const result = {
    status:deviceState.statusRaw, window_ready:deviceState.windowReady, seconds_collected:deviceState.secondsCollected,
    final_risk_score:data.final_risk_score, health_score:data.health_score, environment_score:data.environment_score,
    risk_reason:els.alertLine2 ? els.alertLine2.textContent : ''
  };
  const changed = veApp.lastKind !== null && veApp.lastKind !== kind;
  if (changed || (veApp.lastKind === null && kind !== 'normal')) {
    if (kind === 'critical') veRecordAlert('critical', 'Emergency detected', 'Critical risk level reached. Check your SOS options.');
    else if (kind === 'warning') veRecordAlert('warning', 'Abnormal reading detected', result.risk_reason || 'Vitals moved away from your baseline.');
    else veRecordAlert('info', 'Back to normal', 'Your vitals returned to baseline.');
  }
  if (deviceState.windowReady && (changed || Date.now() - veApp.lastRecorded > 30000)) {
    veRecordAssessment(result, changed ? 'Status change' : 'Routine check');
  }
  veApp.lastKind = kind;
}

/* Manual SOS: two taps, then the existing sendSosAlert() flow. */
function veDisarmSos(){
  const btn = $ve('ve-sos-manual'); clearTimeout(veApp.sosTimer);
  if (btn && !btn.disabled) { btn.classList.remove('is-armed'); btn.textContent = 'Send SOS now'; }
}
document.querySelectorAll('.ve-nav-btn').forEach(btn => btn.addEventListener('click', () => veShowView(btn.dataset.view)));
$ve('ve-alerts-clear').addEventListener('click', () => { veApp.alerts = []; veApp.unread = 0; veRenderAlerts(); veRenderBadge(); });
$ve('ve-sos-manual').addEventListener('click', async () => {
  const btn = $ve('ve-sos-manual');
  if (!btn.classList.contains('is-armed')) {
    btn.classList.add('is-armed'); btn.textContent = 'Tap again to confirm SOS';
    veApp.sosTimer = setTimeout(veDisarmSos, 4000); return;
  }
  clearTimeout(veApp.sosTimer); btn.disabled = true; btn.classList.remove('is-armed'); btn.textContent = 'Sending SOS...';
  try { await sendSosAlert(); } finally { btn.disabled = false; btn.textContent = 'Send SOS now'; }
});
veRenderHistory(); veRenderAlerts();

/* ---------------------------------------------------------------
   6. MAIN SENSOR LOOP
   ---------------------------------------------------------------

   The website continuously sends exactly ONE sample per second.

   Flask maintains the 60-second rolling window.

  /api/data receives the sensor sample. /api/status returns the
  authoritative latest inference and environment result.

   Therefore:

       t=1   → sample 1 → Flask
       t=2   → sample 2 → Flask
       ...
       t=60  → sample 60 → ONNX starts
       t=61  → sample 61 → [2...61] → ONNX
       t=62  → sample 62 → [3...62] → ONNX

   This is a genuine rolling window.
   --------------------------------------------------------------- */

function bootstrap(){

  els.connStatus.textContent =
    '\u25CF CONNECTING…';

  els.bufferBadge.textContent =
    'CONNECTING TO BACKEND…';


  // Send the first sample immediately.
  sendSensorSample();


  // Then send one sample every second.
  setInterval(
    sendSensorSample,
    POLL_INTERVAL_MS
  );
}
/* ---------------------------------------------------------------
   7. SCREEN CANVAS TEXTURE — the watch FACE only
   This is the one thing that changes with health state. The 3D
   watch body, strap, buttons and casing are built once in
   buildProceduralDevice() below and are never touched here.
   --------------------------------------------------------------- */
const screenCanvas = document.createElement('canvas');
screenCanvas.width = 512;
screenCanvas.height = 512;
const sctx = screenCanvas.getContext('2d');
const screenTexture = new THREE.CanvasTexture(screenCanvas);
screenTexture.colorSpace = THREE.SRGBColorSpace;

function roundRect(ctx,x,y,w,h,r){
  ctx.beginPath();
  ctx.moveTo(x+r,y);
  ctx.arcTo(x+w,y,x+w,y+h,r);
  ctx.arcTo(x+w,y+h,x,y+h,r);
  ctx.arcTo(x,y+h,x,y,r);
  ctx.arcTo(x,y,x+w,y,r);
  ctx.closePath();
}
function centerText(ctx,text,x,y){
  ctx.textAlign = 'center';
  ctx.fillText(text, x, y);
  ctx.textAlign = 'left';
}

function updateWatchFace(){
  const s = deviceState;
  const W = screenCanvas.width, H = screenCanvas.height;
  const palette = {
    normal:    { bgTop:'#07160f', bgBottom:'#04100a', fg:'#d9fbe8', accent:'#3ddc8f' },
    warning:   { bgTop:'#2a2007', bgBottom:'#180f04', fg:'#ffe7b0', accent:'#f2b84b' },
    emergency: { bgTop:'#3a070c', bgBottom:'#2a0508', fg:'#ffb2ba', accent:'#ff4d5f' },
  }[s.status];

  sctx.clearRect(0,0,W,H);
  const grad = sctx.createLinearGradient(0,0,0,H);
  grad.addColorStop(0, palette.bgTop);
  grad.addColorStop(1, palette.bgBottom);
  sctx.fillStyle = grad;
  roundRect(sctx,0,0,W,H,54);
  sctx.fill();

  // subtle scanlines for a real-display feel
  sctx.globalAlpha = 0.05;
  sctx.fillStyle = '#ffffff';
  for(let y=0;y<H;y+=4){ sctx.fillRect(0,y,W,1); }
  sctx.globalAlpha = 1;

  sctx.textBaseline = 'top';
  sctx.fillStyle = palette.accent;
  sctx.font = '600 26px "IBM Plex Mono", monospace';
  centerText(sctx, 'VITALEDGE', W/2, 40);

  sctx.strokeStyle = 'rgba(255,255,255,0.12)';
  sctx.lineWidth = 2;
  sctx.beginPath(); sctx.moveTo(34,90); sctx.lineTo(W-34,90); sctx.stroke();

  const clockStr = fmtTime(s.updatedAt);

  if (s.status === 'normal'){
    const dv = s.displayValues;
    if (dv && dv['HR/BVP'] !== undefined){
      // Physical units returned by Flask are shown directly.
      sctx.fillStyle = palette.fg;
      sctx.font = '600 24px "IBM Plex Mono", monospace';
      centerText(sctx, '\u2665 ' + Number(dv['HR/BVP']).toFixed(0), W/2, 140);
      sctx.font = '400 22px "IBM Plex Mono", monospace';
      sctx.fillStyle = 'rgba(217,251,232,0.65)';
      centerText(sctx, 'BPM', W/2, 182);

      sctx.fillStyle = 'rgba(61,220,143,0.16)';
      roundRect(sctx, 100, 224, W-200, 56, 16); sctx.fill();
      sctx.strokeStyle = palette.accent; sctx.lineWidth = 2;
      roundRect(sctx, 100, 224, W-200, 56, 16); sctx.stroke();
      sctx.fillStyle = palette.accent;
      sctx.font = '700 24px "IBM Plex Mono", monospace';
      centerText(sctx, '\u25CF HEALTHY', W/2, 244);

      sctx.font = '500 24px "IBM Plex Mono", monospace';
      sctx.fillStyle = palette.fg;
      centerText(sctx, `Resp ${Number(dv.RESP).toFixed(0)} br/min`, W/2, 320);
      centerText(sctx, dv.TEMP !== undefined ? `Temp ${Number(dv.TEMP).toFixed(1)}\u00B0C` : 'Temp —', W/2, 358);
    } else {
      // Current backend contract: no physical units yet, only the ONNX
      // reconstruction score — show that instead of a fabricated BPM.
      sctx.fillStyle = palette.accent;
      sctx.font = '700 26px "IBM Plex Mono", monospace';
      centerText(sctx, '\u25CF HEALTHY', W/2, 150);

      sctx.fillStyle = 'rgba(61,220,143,0.16)';
      roundRect(sctx, 90, 210, W-180, 96, 18); sctx.fill();
      sctx.strokeStyle = palette.accent; sctx.lineWidth = 2;
      roundRect(sctx, 90, 210, W-180, 96, 18); sctx.stroke();
      sctx.fillStyle = 'rgba(217,251,232,0.6)';
      sctx.font = '400 18px "IBM Plex Mono", monospace';
      centerText(sctx, 'RECONSTRUCTION MSE', W/2, 232);
      sctx.fillStyle = palette.fg;
      sctx.font = '700 30px "IBM Plex Mono", monospace';
      centerText(sctx, s.anomalyScore !== null ? s.anomalyScore.toFixed(2) : '—', W/2, 260);

      sctx.font = '400 20px "IBM Plex Mono", monospace';
      sctx.fillStyle = 'rgba(217,251,232,0.55)';
      centerText(sctx, s.threshold !== null ? `threshold ${s.threshold.toFixed(2)}` : '', W/2, 330);
    }
  } else if (s.status === 'warning'){
    sctx.fillStyle = palette.accent;
    sctx.font = '700 34px "IBM Plex Mono", monospace';
    centerText(sctx, '\u26A0 WARNING', W/2, 150);

    sctx.fillStyle = palette.fg;
    sctx.font = '500 26px "IBM Plex Mono", monospace';
    centerText(sctx, 'Abnormal', W/2, 226);
    centerText(sctx, 'pattern detected', W/2, 262);

    sctx.fillStyle = 'rgba(242,184,75,0.16)';
    roundRect(sctx, 90, 328, W-180, 60, 16); sctx.fill();
    sctx.strokeStyle = palette.accent; sctx.lineWidth = 2;
    roundRect(sctx, 90, 328, W-180, 60, 16); sctx.stroke();
    sctx.fillStyle = palette.accent;
    sctx.font = '700 23px "IBM Plex Mono", monospace';
    centerText(sctx, 'RISK: MODERATE', W/2, 350);
  } else {
    sctx.fillStyle = palette.accent;
    sctx.font = '700 32px "IBM Plex Mono", monospace';
    centerText(sctx, '\u26A0 EMERGENCY', W/2, 150);

    sctx.fillStyle = palette.fg;
    sctx.font = '500 25px "IBM Plex Mono", monospace';
    centerText(sctx, 'Critical abnormality', W/2, 226);
    centerText(sctx, 'detected', W/2, 260);

    sctx.fillStyle = 'rgba(255,77,95,0.22)';
    roundRect(sctx, 90, 326, W-180, 64, 16); sctx.fill();
    sctx.strokeStyle = palette.accent; sctx.lineWidth = 2.5;
    roundRect(sctx, 90, 326, W-180, 64, 16); sctx.stroke();
    sctx.fillStyle = palette.accent;
    sctx.font = '700 26px "IBM Plex Mono", monospace';
    centerText(sctx, 'TAKE ACTION', W/2, 350);
  }

  sctx.fillStyle = 'rgba(255,255,255,0.55)';
  sctx.font = '500 22px "IBM Plex Mono", monospace';
  centerText(sctx, clockStr, W/2, H-56);

  screenTexture.needsUpdate = true;
}

/* ---------------------------------------------------------------
   8. THREE.JS SCENE — untouched from the existing model: same
   geometry, materials, straps, buttons, lighting, camera and
   controls. Only the screen texture above changes with health state.
   --------------------------------------------------------------- */
const mount = document.getElementById('viewer-mount');
const loadingEl = document.getElementById('viewer-loading');

const scene = new THREE.Scene();
scene.background = null;

const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
camera.position.set(3.4, 2.6, 5.2);

const renderer = new THREE.WebGLRenderer({ antialias:true, alpha:true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
mount.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 3.4;
controls.maxDistance = 9;
controls.target.set(0, 0.1, 0);
controls.autoRotate = true;
controls.autoRotateSpeed = 2.4;

// lighting — soft studio rig
scene.add(new THREE.AmbientLight(0x9fb8ac, 0.55));
const key = new THREE.DirectionalLight(0xffffff, 2.1);
key.position.set(4, 6, 4);
scene.add(key);
const rim = new THREE.DirectionalLight(0xdce8e4, 1.1);
rim.position.set(-5, 2, -4);
scene.add(rim);
const fill = new THREE.PointLight(0xb9c9c5, 0.6, 12);
fill.position.set(-2, -1, 3);
scene.add(fill);

const pedestal = new THREE.Mesh(
  new THREE.CircleGeometry(2.6, 64),
  new THREE.MeshStandardMaterial({ color:0x0c1512, roughness:0.95, metalness:0, transparent:true, opacity:0.5 })
);
pedestal.rotation.x = -Math.PI/2;
pedestal.position.y = -0.92;
scene.add(pedestal);

const ring = new THREE.Mesh(
  new THREE.RingGeometry(1.55, 1.58, 90),
  new THREE.MeshBasicMaterial({ color:0xdce8e4, transparent:true, opacity:0.35, side:THREE.DoubleSide })
);
ring.rotation.x = -Math.PI/2;
ring.position.y = -0.915;
scene.add(ring);

/* ---------- procedural device ---------- */
const deviceMeshRefs = {};        // filled in buildProceduralDevice()
const explodeGroup = new THREE.Group(); // parts that move on "exploded view"
let deviceRoot = new THREE.Group();
scene.add(deviceRoot);

function makeLabelSprite(text){
  const c = document.createElement('canvas');
  c.width = 256; c.height = 64;
  const cx = c.getContext('2d');
  cx.fillStyle = 'rgba(10,20,16,0.85)';
  roundRectCtx(cx,2,2,252,60,10); cx.fill();
  cx.strokeStyle = 'rgba(61,220,143,0.6)'; cx.lineWidth=2;
  roundRectCtx(cx,2,2,252,60,10); cx.stroke();
  cx.fillStyle = '#d9fbe8';
  cx.font = '600 22px "IBM Plex Mono", monospace';
  cx.textAlign = 'center'; cx.textBaseline = 'middle';
  cx.fillText(text, 128, 33);
  const tex = new THREE.CanvasTexture(c);
  const mat = new THREE.SpriteMaterial({ map:tex, depthTest:false, transparent:true });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(0.9, 0.225, 1);
  sprite.renderOrder = 10;
  return sprite;
}
function roundRectCtx(ctx,x,y,w,h,r){
  ctx.beginPath();
  ctx.moveTo(x+r,y); ctx.arcTo(x+w,y,x+w,y+h,r); ctx.arcTo(x+w,y+h,x,y+h,r);
  ctx.arcTo(x,y+h,x,y,r); ctx.arcTo(x,y,x+w,y,r); ctx.closePath();
}

function buildProceduralDevice(){
  const group = new THREE.Group();

  const matBody = new THREE.MeshStandardMaterial({ color:0xd9e2df, roughness:0.48, metalness:0.35 });
  const matBezel = new THREE.MeshStandardMaterial({ color:0x8e9b98, roughness:0.48, metalness:0.5 });
  const matGlass = new THREE.MeshPhysicalMaterial({
    color:0x000000, roughness:0.05, metalness:0, transmission:0.55,
    thickness:0.05, clearcoat:1, clearcoatRoughness:0.05, ior:1.5, transparent:true, opacity:0.55,
  });
  const matRubber = new THREE.MeshStandardMaterial({ color:0xb6c2bf, roughness:0.88, metalness:0 });
  const matMetal = new THREE.MeshStandardMaterial({ color:0xc9d1cb, roughness:0.32, metalness:0.9 });
  const matAccent = new THREE.MeshStandardMaterial({ color:0x1a2b23, roughness:0.5, metalness:0.5, emissive:0x3ddc8f, emissiveIntensity:0.15 });
  const matSOS = new THREE.MeshStandardMaterial({ color:0x7a1119, roughness:0.4, metalness:0.35, emissive:0xff2233, emissiveIntensity:0.25 });
  const matPCB = new THREE.MeshStandardMaterial({ color:0x0c5c38, roughness:0.6, metalness:0.2 });
  const matChip = new THREE.MeshStandardMaterial({ color:0x1b1b1e, roughness:0.4, metalness:0.5 });
  const matBattery = new THREE.MeshStandardMaterial({ color:0x9aa39c, roughness:0.3, metalness:0.6 });
  const matGold = new THREE.MeshStandardMaterial({ color:0xd8b463, roughness:0.3, metalness:1 });

  // ----- main housing (rounded square, watch-like but chunkier/rugged) -----
  const BODY_W = 2.05, BODY_H = 2.05, BODY_D = 0.78;
  const body = new THREE.Mesh(new RoundedBoxGeometry(BODY_W, BODY_H, BODY_D, 6, 0.34), matBody);
  body.castShadow = true;
  group.add(body);

  // protective corner bumpers (rugged look)
  const bumperGeo = new THREE.CylinderGeometry(0.14, 0.14, BODY_D+0.04, 16);
  [[1,1],[1,-1],[-1,1],[-1,-1]].forEach(([sx,sy])=>{
    const b = new THREE.Mesh(bumperGeo, matBezel);
    b.rotation.x = Math.PI/2;
    b.position.set(sx*(BODY_W/2-0.16), sy*(BODY_H/2-0.16), 0);
    group.add(b);
  });

  // ----- front bezel + glass + screen -----
  const bezel = new THREE.Mesh(new RoundedBoxGeometry(BODY_W-0.16, BODY_H-0.16, 0.06, 6, 0.28), matBezel);
  bezel.position.z = BODY_D/2 + 0.02;
  group.add(bezel);

  const screenPlane = new THREE.Mesh(
    new THREE.PlaneGeometry(BODY_W-0.34, BODY_H-0.34),
    new THREE.MeshStandardMaterial({ map:screenTexture, emissive:0xffffff, emissiveMap:screenTexture, emissiveIntensity:0.85, roughness:0.4, metalness:0 })
  );
  screenPlane.position.z = BODY_D/2 + 0.052;
  group.add(screenPlane);
  deviceMeshRefs.screenMat = screenPlane.material;

  const glass = new THREE.Mesh(new RoundedBoxGeometry(BODY_W-0.16, BODY_H-0.16, 0.04, 6, 0.26), matGlass);
  glass.position.z = BODY_D/2 + 0.075;
  group.add(glass);

  // ----- side buttons (right edge: SOS on top, secondary below) -----
  const sos = new THREE.Mesh(new RoundedBoxGeometry(0.1, 0.34, 0.16, 3, 0.04), matSOS);
  sos.position.set(BODY_W/2+0.03, 0.34, 0.1);
  sos.rotation.z = Math.PI/2;
  group.add(sos);
  deviceMeshRefs.sosMat = sos.material;

  const sideBtn2 = new THREE.Mesh(new RoundedBoxGeometry(0.09, 0.26, 0.14, 3, 0.03), matMetal);
  sideBtn2.position.set(BODY_W/2+0.025, -0.18, 0.1);
  sideBtn2.rotation.z = Math.PI/2;
  group.add(sideBtn2);
  const btn2ring = new THREE.Mesh(new THREE.TorusGeometry(0.05,0.006,8,24), matAccent);
  btn2ring.position.set(BODY_W/2+0.075, -0.18, 0.1);
  btn2ring.rotation.y = Math.PI/2;
  group.add(btn2ring);

  // ----- crown / dial detail on left edge for rugged tech feel -----
  const crown = new THREE.Mesh(new THREE.CylinderGeometry(0.09,0.09,0.07,20), matMetal);
  crown.rotation.z = Math.PI/2;
  crown.position.set(-BODY_W/2-0.02, 0.05, 0.1);
  group.add(crown);

  // ----- strap (top and bottom) -----
  const strapMat = matRubber;
  const strapGeo = new THREE.BoxGeometry(0.9, 1.6, 0.32);
  const strapTop = new THREE.Mesh(strapGeo, strapMat);
  strapTop.position.set(0, BODY_H/2 + 0.78, -0.02);
  group.add(strapTop);
  const strapBottom = strapTop.clone();
  strapBottom.position.y = -(BODY_H/2 + 0.78);
  group.add(strapBottom);
  // strap perforations (ventilated rugged strap)
  for(let i=0;i<4;i++){
    const perfT = new THREE.Mesh(new THREE.BoxGeometry(0.5,0.05,0.34), matBezel);
    perfT.position.set(0, BODY_H/2 + 0.42 + i*0.24, -0.02);
    group.add(perfT);
    const perfB = perfT.clone();
    perfB.position.y = -(BODY_H/2 + 0.42 + i*0.24);
    group.add(perfB);
  }

  // ----- rear sensor cluster (PPG optical assembly) -----
  const rearGroup = new THREE.Group();
  rearGroup.position.z = -BODY_D/2 - 0.01;
  const rearHousing = new THREE.Mesh(new THREE.CylinderGeometry(0.34,0.34,0.05,32), matBezel);
  rearHousing.rotation.x = Math.PI/2;
  rearHousing.position.z = -0.01;
  rearGroup.add(rearHousing);

  const photodiode = new THREE.Mesh(new THREE.CylinderGeometry(0.09,0.09,0.02,24), matChip);
  photodiode.rotation.x = -Math.PI/2;
  photodiode.position.z = -0.03;
  rearGroup.add(photodiode);

  const ledColors = [0x24e07a, 0x24e07a, 0xff4d4d, 0x4d5bff];
  ledColors.forEach((c, i) => {
    const angle = (i / ledColors.length) * Math.PI * 2 + 0.4;
    const led = new THREE.Mesh(
      new THREE.CylinderGeometry(0.035,0.035,0.03,16),
      new THREE.MeshStandardMaterial({ color:c, emissive:c, emissiveIntensity:0.9, roughness:0.3 })
    );
    led.rotation.x = -Math.PI/2;
    led.position.set(Math.cos(angle)*0.2, Math.sin(angle)*0.2, -0.03);
    rearGroup.add(led);
    deviceMeshRefs['led'+i] = led.material;
  });

  // skin-temperature sensor (separate small metal pad, offset from PPG ring)
  const tempSensor = new THREE.Mesh(new THREE.CylinderGeometry(0.055,0.055,0.02,20), matMetal);
  tempSensor.rotation.x = -Math.PI/2;
  tempSensor.position.set(0.5, -0.5, -0.005);
  rearGroup.add(tempSensor);
  const tempRing = new THREE.Mesh(new THREE.TorusGeometry(0.07,0.006,8,24), matAccent);
  tempRing.position.set(0.5,-0.5,-0.005);
  rearGroup.add(tempRing);

  group.add(rearGroup);

  // charging contacts (row of 4 pads near bottom edge, rear face)
  const contactsGroup = new THREE.Group();
  contactsGroup.position.set(0, -BODY_H/2+0.22, -BODY_D/2-0.005);
  for(let i=0;i<4;i++){
    const pad = new THREE.Mesh(new THREE.CylinderGeometry(0.045,0.045,0.015,20), matGold);
    pad.rotation.x = -Math.PI/2;
    pad.position.x = (i-1.5)*0.14;
    contactsGroup.add(pad);
  }
  group.add(contactsGroup);

  // ----- internal components (hidden inside housing normally; revealed on "exploded view") -----
  const pcb = new THREE.Mesh(new THREE.BoxGeometry(BODY_W-0.5, BODY_H-0.5, 0.05), matPCB);
  pcb.position.set(0,0,0.08);
  explodeGroup.add(pcb);
  // chips on pcb
  const mcu = new THREE.Mesh(new THREE.BoxGeometry(0.34,0.34,0.06), matChip);
  mcu.position.set(-0.2,0.2,0.11); explodeGroup.add(mcu);
  const wireless = new THREE.Mesh(new THREE.BoxGeometry(0.22,0.16,0.05), matChip);
  wireless.position.set(0.3,0.25,0.105); explodeGroup.add(wireless);
  const battery = new THREE.Mesh(new THREE.BoxGeometry(BODY_W-0.7, BODY_H-0.9, 0.12), matBattery);
  battery.position.set(0,-0.15,-0.05);
  explodeGroup.add(battery);

  const labelMCU = makeLabelSprite('MCU / PCB');
  labelMCU.position.set(-0.2, 0.85, 0.11);
  const labelBatt = makeLabelSprite('BATTERY');
  labelBatt.position.set(0, -0.95, -0.05);
  const labelWireless = makeLabelSprite('WIRELESS MODULE');
  labelWireless.position.set(0.9, 0.55, 0.105);
  const labelPPG = makeLabelSprite('PPG SENSOR ARRAY');
  explodeGroup.add(labelMCU, labelBatt, labelWireless);

  group.add(explodeGroup);
  explodeGroup.userData.restZ = { pcb: pcb.position.z, battery: battery.position.z, mcu: mcu.position.z, wireless: wireless.position.z };
  explodeGroup.userData.parts = { pcb, mcu, wireless, battery, labelMCU, labelBatt, labelWireless };
  explodeGroup.traverse(o=>{ if(o.isSprite) o.visible=false; else if(o!==explodeGroup) o.userData.baseOpacity=1; });

  deviceMeshRefs.body = body;
  deviceMeshRefs.glass = glass;

  return group;
}

function loadDevice(){
  if (EXTERNAL_MODEL_URL){
    const loader = new GLTFLoader();
    loader.load(EXTERNAL_MODEL_URL, (gltf)=>{
      deviceRoot.add(gltf.scene);
      gltf.scene.traverse(o=>{
        if(o.isMesh && o.name === 'Screen'){
          o.material = new THREE.MeshStandardMaterial({ map:screenTexture, emissive:0xffffff, emissiveMap:screenTexture, emissiveIntensity:0.85 });
          deviceMeshRefs.screenMat = o.material;
        }
      });
      finishLoad();
    }, undefined, (err)=>{
      console.warn('GLB load failed, falling back to procedural model', err);
      deviceRoot.add(buildProceduralDevice());
      finishLoad();
    });
  } else {
    deviceRoot.add(buildProceduralDevice());
    finishLoad();
  }
}

function finishLoad(){
  updateWatchFace();
  loadingEl.style.opacity = '0';
  setTimeout(()=> loadingEl.style.display='none', 420);
  syncCanvasSize(els.graphHr);
  syncCanvasSize(els.graphResp);
  syncCanvasSize(els.graphEcg);
  bootstrap();
}

loadDevice();

window.addEventListener('resize', ()=>{
  syncCanvasSize(els.graphHr);
  syncCanvasSize(els.graphResp);
  syncCanvasSize(els.graphEcg);
  drawSparkline(els.graphHr, graphHistories.hr, '#3ddc8f');
  drawSparkline(els.graphResp, graphHistories.respiration, '#7bb8ff');
  drawSparkline(els.graphEcg, graphHistories.ecg, '#f2b84b');
});

/* ---------- resize ---------- */
function resize(){
  const w = mount.clientWidth, h = mount.clientHeight;
  camera.aspect = w/h;
  camera.updateProjectionMatrix();
  renderer.setSize(w,h,false);
}
new ResizeObserver(resize).observe(mount);
resize();

/* ---------- render loop ---------- */
const clock = new THREE.Clock();
let explodeState = false;
let explodeProgress = 0;
let statePulse = 0;
function animate(){
  requestAnimationFrame(animate);
  const dt = clock.getDelta();
  controls.update();

  // The watch body, strap, SOS button and secondary button never change
  // colour with health state — only the screen glow pulses, and only
  // for warning/emergency, so the alert reads on the display itself.
  if (deviceMeshRefs.screenMat){
    if (deviceState.status === 'normal'){
      deviceMeshRefs.screenMat.emissiveIntensity = 0.85;
    } else {
      statePulse += dt * (deviceState.status === 'emergency' ? 5 : 3);
      const p = (Math.sin(statePulse)+1)/2;
      deviceMeshRefs.screenMat.emissiveIntensity = 0.8 + p*0.45;
    }
  }

  // gentle PPG LED shimmer
  for(let i=0;i<4;i++){
    const m = deviceMeshRefs['led'+i];
    if(m) m.emissiveIntensity = 0.6 + Math.sin(clock.elapsedTime*3 + i)*0.3;
  }

  // exploded view lerp
  const target = explodeState ? 1 : 0;
  explodeProgress += (target - explodeProgress) * Math.min(1, dt*6);
  applyExplodeProgress(explodeProgress);

  renderer.render(scene, camera);
}
animate();

/* ---------------------------------------------------------------
   9. UI TOOLBAR — rotate toggle, exploded view, reset (unchanged)
   --------------------------------------------------------------- */
function applyExplodeProgress(t){
  const parts = explodeGroup.userData.parts;
  if(!parts) return;
  const rz = explodeGroup.userData.restZ;
  parts.pcb.position.z = rz.pcb + t*0.55;
  parts.mcu.position.z = rz.mcu + t*0.55;
  parts.wireless.position.z = rz.wireless + t*0.55;
  parts.battery.position.z = rz.battery - t*0.55;
  [parts.labelMCU, parts.labelBatt, parts.labelWireless].forEach(l=>{
    l.visible = t > 0.35;
    l.material.opacity = Math.min(1,(t-0.35)/0.4);
  });
  if (deviceMeshRefs.body) deviceMeshRefs.body.material.opacity = 1 - t*0.55;
  if (deviceMeshRefs.body) deviceMeshRefs.body.material.transparent = t > 0.02;
  if (deviceMeshRefs.glass) deviceMeshRefs.glass.visible = t < 0.5;
}

const rotateToggle = document.getElementById('toggle-rotate');
const explodeToggle = document.getElementById('toggle-explode');
const resetBtn = document.getElementById('toggle-reset');

rotateToggle.addEventListener('click', ()=>{
  controls.autoRotate = !controls.autoRotate;
  rotateToggle.classList.toggle('active', controls.autoRotate);
});
explodeToggle.addEventListener('click', ()=>{
  explodeState = !explodeState;
  explodeToggle.classList.toggle('active', explodeState);
});
resetBtn.addEventListener('click', ()=>{
  camera.position.set(3.4, 2.6, 5.2);
  controls.target.set(0,0.1,0);
});
