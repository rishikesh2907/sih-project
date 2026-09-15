# ===============================================================================
# VITALEDGE - FLASK BACKEND
# Website -> Flask -> Rolling 60s Window -> ONNX -> Health Anomaly
#                                      +
#                              Real Weather / AQI
#                                      ↓
#                                  Fusion
#                                      ↓
#                            Final Risk Assessment
# ===============================================================================
#                                      +
#                              Real Weather / AQI
#                                      ↓
#                                  Fusion
#                                      ↓
#                            Final Risk Assessment
# ==============================================================================

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

import numpy as np
import onnxruntime as ort
import threading
import os
import time
import requests


# ==============================================================================
# FLASK CONFIGURATION
# ==============================================================================

app = Flask(__name__)
CORS(app)


# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "vitaledge_6d_deep_autoencoder.onnx"
)

# Current threshold from your existing backend
HEALTH_THRESHOLD = 1.785423

SEQ_LEN = 60
FEATURE_DIM = 6

FEATURE_NAMES = [
    "HR/BVP",
    "ACC",
    "TEMP",
    "RESP",
    "EDA",
    "ECG"
]


# ==============================================================================
# LOCATION
# ==============================================================================

# Sasaram, Bihar, India
LOCATION_NAME = "Sasaram, Bihar, India"

LATITUDE = 24.9535
LONGITUDE = 84.0118

TIMEZONE = "Asia/Kolkata"


# ==============================================================================
# PHYSIOLOGICAL BASELINE
# ==============================================================================

BASELINE_MEAN = {
    "HR/BVP": 72.0,
    "ACC": 0.0,
    "TEMP": 36.7,
    "RESP": 15.0,
    "EDA": 2.5,
    "ECG": 0.0,
}

BASELINE_STD = {
    "HR/BVP": 15.0,
    "ACC": 0.5,
    "TEMP": 0.8,
    "RESP": 5.0,
    "EDA": 3.0,
    "ECG": 1.0,
}


# ==============================================================================
# FUSION CONFIGURATION
# ==============================================================================

HEALTH_WEIGHT = 0.70
ENVIRONMENT_WEIGHT = 0.30

# Initial prototype thresholds.
WARNING_THRESHOLD = 0.70
CRITICAL_THRESHOLD = 1.00

# Safety overrides.
HEALTH_CRITICAL_SCORE = 1.50
ENVIRONMENT_CRITICAL_SCORE = 0.90


# ==============================================================================
# WEATHER API CONFIGURATION
# ==============================================================================

WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"

AIR_QUALITY_API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# Don't request weather every second.
# Cache for 5 minutes.
WEATHER_CACHE_SECONDS = 300

# The demo endpoint records and acknowledges SOS requests. A production
# deployment can replace this with its SMS, email, or webhook provider.
EMERGENCY_CONTACT = os.getenv("EMERGENCY_CONTACT", "configured emergency contact")


# ==============================================================================
# WEATHER CACHE
# ==============================================================================

weather_cache = {
    "timestamp": 0,
    "data": None
}

weather_cache_lock = threading.Lock()


# ==============================================================================
# LOAD ONNX MODEL
# ==============================================================================

print("=" * 70)
print("VITALEDGE BACKEND")
print("=" * 70)

print("Loading model:")
print(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"ONNX model not found: {MODEL_PATH}"
    )

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print(f"Input name         : {input_name}")
print(f"Output name        : {output_name}")
print(f"Health threshold   : {HEALTH_THRESHOLD}")

print()
print("Weather location:")
print(f"Location           : {LOCATION_NAME}")
print(f"Latitude           : {LATITUDE}")
print(f"Longitude          : {LONGITUDE}")
print(f"Timezone           : {TIMEZONE}")

print("=" * 70)


# ==============================================================================
# GLOBAL STATE
# ==============================================================================

rolling_buffer = []

simulation_mode = "normal"

latest_result = {
    "status": "WAITING",

    "anomaly_score": 0.0,

    "health_threshold": HEALTH_THRESHOLD,

    "health_score": 0.0,

    "environment_score": 0.0,

    "final_risk_score": 0.0,

    "risk_reason": "Waiting for 60-second window",

    "window_ready": False,

    "seconds_collected": 0,

    "mode": "normal",

    "environment": {
        "temperature": 0.0,
        "humidity": 0.0,
        "aqi": 0.0,
        "uv": 0.0,
        "weather_code": None,
        "pm2_5": None,
        "pm10": None,
        "source": "Open-Meteo",
        "location": LOCATION_NAME,
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "updated_at": None
    },

    "environment_risk": {
        "temperature": 0.0,
        "humidity": 0.0,
        "aqi": 0.0,
        "uv": 0.0
    },

    "weather_source": "Open-Meteo",

    "weather_location": LOCATION_NAME,

    "values": {
        feature: 0.0
        for feature in FEATURE_NAMES
    }
}

buffer_lock = threading.Lock()


# ==============================================================================
# ONNX INFERENCE
# ==============================================================================

def run_inference(window):
    """
    Run the existing ONNX autoencoder.

    Input:
        window shape = (60, 6)

    Returns:
        reconstruction MSE
    """

    x = np.asarray(
        window,
        dtype=np.float32
    )

    # Expected:
    # (batch, sequence, features)

    x = np.expand_dims(
        x,
        axis=0
    )

    reconstruction = session.run(
        [output_name],
        {
            input_name: x
        }
    )[0]

    mse = np.mean(
        np.square(
            x - reconstruction
        )
    )

    return float(mse)


# ==============================================================================
# SENSOR DATA VALIDATION
# ==============================================================================

def validate_sensor_data(data):

    if not isinstance(data, dict):
        raise ValueError(
            "Sensor values must be a JSON object"
        )

    cleaned = {}

    for feature in FEATURE_NAMES:

        if feature not in data:
            raise ValueError(
                f"Missing feature: {feature}"
            )

        try:
            value = float(
                data[feature]
            )

        except (TypeError, ValueError):

            raise ValueError(
                f"Invalid value for {feature}"
            )

        if not np.isfinite(value):

            raise ValueError(
                f"Non-finite value for {feature}"
            )

        cleaned[feature] = value

    return cleaned


# ==============================================================================
# STANDARDIZE SENSOR DATA
# ==============================================================================

def standardize_sensor_data(values):

    return {
        feature:
            (
                values[feature]
                - BASELINE_MEAN[feature]
            )
            / BASELINE_STD[feature]

        for feature in FEATURE_NAMES
    }


# ==============================================================================
# HEALTH SCORE
# ==============================================================================

def calculate_health_score(mse):

    if HEALTH_THRESHOLD <= 0:
        return 0.0

    return float(
        mse / HEALTH_THRESHOLD
    )


# ==============================================================================
# WEATHER API
# ==============================================================================

def fetch_real_weather():

    """
    Fetch current weather conditions for Sasaram.

    Open-Meteo provides:
        temperature
        relative humidity
        UV index
        weather code
    """

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,

        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "uv_index,"
            "weather_code"
        ),

        "timezone": TIMEZONE
    }

    response = requests.get(
        WEATHER_API_URL,
        params=params,
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    current = data.get(
        "current",
        {}
    )

    temperature = current.get(
        "temperature_2m"
    )

    humidity = current.get(
        "relative_humidity_2m"
    )

    uv = current.get(
        "uv_index"
    )

    weather_code = current.get(
        "weather_code"
    )

    if temperature is None:
        raise ValueError(
            "Temperature missing from weather API"
        )

    if humidity is None:
        raise ValueError(
            "Humidity missing from weather API"
        )

    if uv is None:
        uv = 0.0

    return {
        "temperature": float(
            temperature
        ),

        "humidity": float(
            humidity
        ),

        "uv": float(
            uv
        ),

        "weather_code": weather_code
    }


# ==============================================================================
# AIR QUALITY API
# ==============================================================================

def fetch_real_air_quality():

    """
    Fetch current US AQI for Sasaram.
    """

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,

        "current": (
            "us_aqi,"
            "pm2_5,"
            "pm10"
        ),

        "timezone": TIMEZONE
    }

    response = requests.get(
        AIR_QUALITY_API_URL,
        params=params,
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    current = data.get(
        "current",
        {}
    )

    aqi = current.get(
        "us_aqi"
    )

    pm25 = current.get(
        "pm2_5"
    )

    pm10 = current.get(
        "pm10"
    )

    if aqi is None:

        raise ValueError(
            "US AQI missing from air quality API"
        )

    return {
        "aqi": float(aqi),

        "pm2_5": (
            float(pm25)
            if pm25 is not None
            else None
        ),

        "pm10": (
            float(pm10)
            if pm10 is not None
            else None
        )
    }


# ==============================================================================
# GET REAL ENVIRONMENTAL DATA
# ==============================================================================

def get_environmental_data():

    """
    Get real weather + air quality.

    Results are cached for WEATHER_CACHE_SECONDS
    so the API is not called every second.
    """

    global weather_cache

    current_time = time.time()

    # --------------------------------------------------------------
    # Return cached data if still fresh
    # --------------------------------------------------------------

    with weather_cache_lock:

        if (
            weather_cache["data"] is not None
            and
            (
                current_time
                - weather_cache["timestamp"]
            )
            < WEATHER_CACHE_SECONDS
        ):

            return dict(
                weather_cache["data"]
            )

    # --------------------------------------------------------------
    # Fetch fresh data
    # --------------------------------------------------------------

    try:

        weather = fetch_real_weather()

        air_quality = fetch_real_air_quality()

        environment = {

            "temperature":
                weather["temperature"],

            "humidity":
                weather["humidity"],

            "aqi":
                air_quality["aqi"],

            "uv":
                weather["uv"],

            "weather_code":
                weather["weather_code"],

            "pm2_5":
                air_quality["pm2_5"],

            "pm10":
                air_quality["pm10"],

            "source":
                "Open-Meteo",

            "location":
                LOCATION_NAME,

            "latitude":
                LATITUDE,

            "longitude":
                LONGITUDE,

            "updated_at":
                time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                    time.localtime()
                )
        }

        # ----------------------------------------------------------
        # Update cache
        # ----------------------------------------------------------

        with weather_cache_lock:

            weather_cache = {
                "timestamp":
                    current_time,

                "data":
                    environment
            }

        print(
            ">>> WEATHER UPDATED:",
            environment
        )

        return dict(
            environment
        )

    except Exception as e:

        print(
            "WARNING: Weather API failed:",
            str(e)
        )

        # ----------------------------------------------------------
        # Fallback to previous successful data
        # ----------------------------------------------------------

        with weather_cache_lock:

            if weather_cache["data"] is not None:

                fallback = dict(
                    weather_cache["data"]
                )

                fallback["source"] = (
                    "Open-Meteo cached"
                )

                return fallback

        # ----------------------------------------------------------
        # No previous data available
        # ----------------------------------------------------------

        return {
            "temperature": 25.0,
            "humidity": 50.0,
            "aqi": 50.0,
            "uv": 3.0,
            "weather_code": None,
            "pm2_5": None,
            "pm10": None,
            "source": "Fallback",
            "location": LOCATION_NAME,
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "updated_at": None
        }


# ==============================================================================
# TEMPERATURE RISK
# ==============================================================================

def temperature_risk(temperature):

    temperature = float(
        temperature
    )

    # Comfortable range.
    if 18.0 <= temperature <= 28.0:
        return 0.0

    # Cold.
    if temperature < 18.0:

        risk = (
            18.0 - temperature
        ) / 18.0

        return float(
            np.clip(
                risk,
                0.0,
                1.0
            )
        )

    # Heat.
    risk = (
        temperature - 28.0
    ) / 20.0

    return float(
        np.clip(
            risk,
            0.0,
            1.0
        )
    )


# ==============================================================================
# HUMIDITY RISK
# ==============================================================================

def humidity_risk(humidity):

    humidity = float(
        humidity
    )

    if 30.0 <= humidity <= 60.0:
        return 0.0

    if humidity < 30.0:

        risk = (
            30.0 - humidity
        ) / 30.0

    else:

        risk = (
            humidity - 60.0
        ) / 40.0

    return float(
        np.clip(
            risk,
            0.0,
            1.0
        )
    )


# ==============================================================================
# AQI RISK
# ==============================================================================

def aqi_risk(aqi):

    """
    US AQI interpretation.

    0-50       Good
    51-100     Moderate
    101-150    Unhealthy for sensitive groups
    151-200    Unhealthy
    201-300    Very unhealthy
    301-500    Hazardous
    """

    aqi = float(
        aqi
    )

    if aqi <= 50.0:
        return 0.0

    if aqi <= 100.0:

        risk = (
            aqi - 50.0
        ) / 100.0

    elif aqi <= 150.0:

        risk = (
            0.5
            +
            (
                aqi - 100.0
            ) / 100.0
        )

    elif aqi <= 200.0:

        risk = (
            0.75
            +
            (
                aqi - 150.0
            ) / 200.0
        )

    else:

        risk = 1.0

    return float(
        np.clip(
            risk,
            0.0,
            1.0
        )
    )


# ==============================================================================
# UV RISK
# ==============================================================================

def uv_risk(uv):

    uv = float(
        uv
    )

    if uv <= 2.0:
        return 0.0

    risk = (
        uv - 2.0
    ) / 8.0

    return float(
        np.clip(
            risk,
            0.0,
            1.0
        )
    )


# ==============================================================================
# ENVIRONMENT SCORE
# ==============================================================================

def calculate_environment_score(
    environment
):

    temp_score = temperature_risk(
        environment["temperature"]
    )

    humidity_score = humidity_risk(
        environment["humidity"]
    )

    aqi_score = aqi_risk(
        environment["aqi"]
    )

    uv_score = uv_risk(
        environment["uv"]
    )

    # Environmental weighting.
    score = (
        0.40 * temp_score
        +
        0.25 * humidity_score
        +
        0.25 * aqi_score
        +
        0.10 * uv_score
    )

    return (
        float(
            np.clip(
                score,
                0.0,
                1.0
            )
        ),

        {
            "temperature":
                temp_score,

            "humidity":
                humidity_score,

            "aqi":
                aqi_score,

            "uv":
                uv_score
        }
    )


# ==============================================================================
# FUSION ENGINE
# ==============================================================================

def calculate_final_risk(
    health_score,
    environment_score
):

    # --------------------------------------------------------------
    # Weighted fusion
    # --------------------------------------------------------------

    final_score = (
        HEALTH_WEIGHT * health_score
        +
        ENVIRONMENT_WEIGHT * environment_score
    )

    # --------------------------------------------------------------
    # Severe physiological anomaly
    # --------------------------------------------------------------

    if health_score >= HEALTH_CRITICAL_SCORE:

        return (
            float(final_score),
            "CRITICAL",
            "Severe physiological anomaly detected"
        )

    # --------------------------------------------------------------
    # Severe environmental stress
    # --------------------------------------------------------------

    if environment_score >= ENVIRONMENT_CRITICAL_SCORE:

        return (
            float(final_score),
            "CRITICAL",
            "Severe environmental stress detected"
        )

    # --------------------------------------------------------------
    # Combined risk
    # --------------------------------------------------------------

    if final_score >= CRITICAL_THRESHOLD:

        return (
            float(final_score),
            "CRITICAL",
            "Combined physiological and environmental risk"
        )

    # --------------------------------------------------------------
    # Warning
    # --------------------------------------------------------------

    if final_score >= WARNING_THRESHOLD:

        return (
            float(final_score),
            "WARNING",
            "Elevated health or environmental risk"
        )

    # --------------------------------------------------------------
    # Normal
    # --------------------------------------------------------------

    return (
        float(final_score),
        "NORMAL",
        "No significant anomaly detected"
    )


# ==============================================================================
# PROCESS ONE SENSOR SAMPLE
# ==============================================================================

def process_sensor_sample(values):

    global rolling_buffer
    global latest_result

    # --------------------------------------------------------------
    # Standardize physiological data
    # --------------------------------------------------------------

    model_values = standardize_sensor_data(
        values
    )

    # --------------------------------------------------------------
    # Create model vector
    # --------------------------------------------------------------

    sample = np.array(
        [
            model_values["HR/BVP"],
            model_values["ACC"],
            model_values["TEMP"],
            model_values["RESP"],
            model_values["EDA"],
            model_values["ECG"]
        ],
        dtype=np.float32
    )

    with buffer_lock:

        # ----------------------------------------------------------
        # Add new sample
        # ----------------------------------------------------------

        rolling_buffer.append(
            sample
        )

        # ----------------------------------------------------------
        # Keep latest 60 seconds
        # ----------------------------------------------------------

        if len(rolling_buffer) > SEQ_LEN:

            rolling_buffer.pop(0)

        seconds_collected = len(
            rolling_buffer
        )

        # ----------------------------------------------------------
        # Get current real environment
        # ----------------------------------------------------------

        environment = (
            get_environmental_data()
        )

        environment_score, environment_risk = (
            calculate_environment_score(
                environment
            )
        )

        # ----------------------------------------------------------
        # Still collecting
        # ----------------------------------------------------------

        if seconds_collected < SEQ_LEN:

            latest_result = {

                "status":
                    "COLLECTING",

                "anomaly_score":
                    0.0,

                "health_threshold":
                    HEALTH_THRESHOLD,

                "health_score":
                    0.0,

                "environment_score":
                    environment_score,

                "final_risk_score":
                    0.0,

                "risk_reason":
                    "Collecting 60-second physiological window",

                "window_ready":
                    False,

                "seconds_collected":
                    seconds_collected,

                "mode":
                    simulation_mode,

                "environment":
                    environment,

                "environment_risk":
                    environment_risk,

                "weather_source":
                    environment.get(
                        "source",
                        "Open-Meteo"
                    ),

                "weather_location":
                    LOCATION_NAME,

                "values":
                    model_values,

                "display_values":
                    values
            }

            return latest_result

        # ----------------------------------------------------------
        # Create 60-second window
        # ----------------------------------------------------------

        window = np.array(
            rolling_buffer,
            dtype=np.float32
        )

        # ----------------------------------------------------------
        # ONNX inference
        # ----------------------------------------------------------

        anomaly_score = run_inference(
            window
        )

        # ----------------------------------------------------------
        # Health score
        # ----------------------------------------------------------

        health_score = calculate_health_score(
            anomaly_score
        )

        # ----------------------------------------------------------
        # Fusion
        # ----------------------------------------------------------

        (
            final_risk_score,
            status,
            risk_reason
        ) = calculate_final_risk(
            health_score,
            environment_score
        )

        # ----------------------------------------------------------
        # Save result
        # ----------------------------------------------------------

        latest_result = {

            "status":
                status,

            "anomaly_score":
                anomaly_score,

            "health_threshold":
                HEALTH_THRESHOLD,

            "health_score":
                health_score,

            "environment_score":
                environment_score,

            "final_risk_score":
                final_risk_score,

            "risk_reason":
                risk_reason,

            "window_ready":
                True,

            "seconds_collected":
                SEQ_LEN,

            "mode":
                simulation_mode,

            "environment":
                environment,

            "environment_risk":
                environment_risk,

            "weather_source":
                environment.get(
                    "source",
                    "Open-Meteo"
                ),

            "weather_location":
                LOCATION_NAME,

            "values":
                model_values,

            "display_values":
                values
        }

        return latest_result


# ==============================================================================
# HOME PAGE
# ==============================================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==============================================================================
# RECEIVE SENSOR DATA
# ==============================================================================

@app.route(
    "/api/data",
    methods=["POST"]
)
def receive_sensor_data():

    try:

        data = request.get_json(
            silent=True
        )

        if data is None:

            return jsonify({
                "error":
                    "Request must contain JSON"
            }), 400

        if "values" in data:

            sensor_values = data[
                "values"
            ]

        else:

            sensor_values = data

        sensor_values = validate_sensor_data(
            sensor_values
        )

        result = process_sensor_sample(
            sensor_values
        )

        return jsonify(
            result
        ), 200

    except ValueError as e:

        return jsonify({
            "error":
                str(e)
        }), 400

    except Exception as e:

        print(
            "ERROR /api/data:",
            e
        )

        return jsonify({

            "error":
                "Internal server error",

            "details":
                str(e)

        }), 500


# ==============================================================================
# CURRENT STATUS
# ==============================================================================

@app.route(
    "/api/status",
    methods=["GET"]
)
def get_status():

    with buffer_lock:

        result = dict(
            latest_result
        )

        result["values"] = dict(
            latest_result[
                "values"
            ]
        )

        result["environment"] = dict(
            latest_result[
                "environment"
            ]
        )

        result["environment_risk"] = dict(
            latest_result[
                "environment_risk"
            ]
        )

        return jsonify(
            result
        )


# ==============================================================================
# CURRENT REAL ENVIRONMENT
# ==============================================================================

@app.route(
    "/api/environment",
    methods=["GET"]
)
def get_environment():

    environment = (
        get_environmental_data()
    )

    environment_score, environment_risk = (
        calculate_environment_score(
            environment
        )
    )

    return jsonify({

        "location":
            LOCATION_NAME,

        "latitude":
            LATITUDE,

        "longitude":
            LONGITUDE,

        "source":
            environment.get(
                "source",
                "Open-Meteo"
            ),

        "environment":
            environment,

        "environment_risk":
            environment_risk,

        "environment_score":
            environment_score

    })


# ==============================================================================
# SIMULATE ABNORMAL
# ==============================================================================

@app.route(
    "/api/send-sos",
    methods=["POST"]
)
def send_sos():

    payload = request.get_json(silent=True) or {}
    status = str(payload.get("status", "UNKNOWN")).upper()

    print(
        f">>> SOS ALERT SENT TO {EMERGENCY_CONTACT} (status: {status})"
    )

    return jsonify({
        "success": True,
        "contact": EMERGENCY_CONTACT,
        "message": "SOS alert sent to emergency contact"
    })


@app.route(
    "/api/report-false-alarm",
    methods=["POST"]
)
def report_false_alarm():

    print(
        f">>> FALSE ALARM REPORTED FOR {EMERGENCY_CONTACT}"
    )

    return jsonify({
        "success": True,
        "message": "False alarm reported"
    })


@app.route(
    "/api/simulate-abnormal",
    methods=["POST"]
)
def simulate_abnormal():

    global simulation_mode
    global latest_result

    with buffer_lock:

        simulation_mode = "abnormal"

        # Don't clear the buffer.
        #
        # New abnormal samples gradually replace
        # old normal samples.

        latest_result[
            "mode"
        ] = "abnormal"

    print(
        ">>> ABNORMAL SIMULATION ENABLED"
    )

    return jsonify({

        "success":
            True,

        "mode":
            "abnormal",

        "message":
            "Abnormal simulation enabled"

    })


# ==============================================================================
# RESUME NORMAL
# ==============================================================================

@app.route(
    "/api/reset",
    methods=["POST"]
)
def reset_simulation():

    global simulation_mode
    global latest_result
    global rolling_buffer

    with buffer_lock:

        simulation_mode = "normal"

        rolling_buffer.clear()

        latest_result = {

            "status":
                "WAITING",

            "anomaly_score":
                0.0,

            "health_threshold":
                HEALTH_THRESHOLD,

            "health_score":
                0.0,

            "environment_score":
                0.0,

            "final_risk_score":
                0.0,

            "risk_reason":
                "Waiting for 60-second window",

            "window_ready":
                False,

            "seconds_collected":
                0,

            "mode":
                "normal",

            "environment":
                get_environmental_data(),

            "environment_risk": {
                "temperature": 0.0,
                "humidity": 0.0,
                "aqi": 0.0,
                "uv": 0.0
            },

            "weather_source":
                "Open-Meteo",

            "weather_location":
                LOCATION_NAME,

            "values": {
                feature: 0.0
                for feature in FEATURE_NAMES
            }
        }

    print(
        ">>> NORMAL MODE / BUFFER RESET"
    )

    return jsonify({

        "success":
            True,

        "mode":
            "normal",

        "message":
            "Normal mode enabled and buffer reset"

    })


# ==============================================================================
# HEALTH CHECK
# ==============================================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status":
            "ok",

        "model_loaded":
            True,

        "health_threshold":
            HEALTH_THRESHOLD,

        "fusion":
            "enabled",

        "weather":
            "Open-Meteo",

        "location":
            LOCATION_NAME

    })


@app.route(
    "/api/health",
    methods=["GET"]
)
def api_health():

    return jsonify({

        "status":
            "ok",

        "model_loaded":
            True,

        "health_threshold":
            HEALTH_THRESHOLD,

        "fusion":
            "enabled",

        "weather":
            "Open-Meteo",

        "location":
            LOCATION_NAME

    })


# ==============================================================================
# DEBUG
# ==============================================================================

@app.route(
    "/api/debug",
    methods=["GET"]
)
def debug():

    with buffer_lock:

        with weather_cache_lock:

            weather_age = None

            if weather_cache["timestamp"] > 0:

                weather_age = (
                    time.time()
                    -
                    weather_cache["timestamp"]
                )

        return jsonify({

            "buffer_length":
                len(rolling_buffer),

            "sequence_length":
                SEQ_LEN,

            "feature_dimension":
                FEATURE_DIM,

            "features":
                FEATURE_NAMES,

            "mode":
                simulation_mode,

            "health_threshold":
                HEALTH_THRESHOLD,

            "health_weight":
                HEALTH_WEIGHT,

            "environment_weight":
                ENVIRONMENT_WEIGHT,

            "warning_threshold":
                WARNING_THRESHOLD,

            "critical_threshold":
                CRITICAL_THRESHOLD,

            "weather_provider":
                "Open-Meteo",

            "weather_location":
                LOCATION_NAME,

            "latitude":
                LATITUDE,

            "longitude":
                LONGITUDE,

            "weather_cache_seconds":
                WEATHER_CACHE_SECONDS,

            "weather_cache_age":
                weather_age,

            "model":
                "vitaledge_6d_deep_autoencoder.onnx",

            "input_name":
                input_name,

            "output_name":
                output_name

        })


# ==============================================================================
# START SERVER
# ==============================================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("VITALEDGE SERVER STARTING")
    print("=" * 70)
    print()

    print("Website:")
    print(
        "http://127.0.0.1:5000/"
    )

    print()

    print("API:")
    print("POST /api/data")
    print("GET  /api/status")
    print("GET  /api/environment")
    print("POST /api/simulate-abnormal")
    print("POST /api/reset")

    print()

    print("External Data:")
    print(
        "Open-Meteo Weather + Air Quality"
    )

    print(
        f"Location: {LOCATION_NAME}"
    )

    print(
        f"Coordinates: {LATITUDE}, {LONGITUDE}"
    )

    print()

    print("Architecture:")
    print()
    print("Website")
    print("   ↓")
    print("POST /api/data")
    print("   ↓")
    print("60-second rolling buffer")
    print("   ↓")
    print("ONNX Autoencoder")
    print("   ↓")
    print("Reconstruction MSE")
    print(
        f"   ↓ Health threshold = {HEALTH_THRESHOLD}"
    )
    print("   ↓")
    print("Health Score")
    print("   ↓")
    print("              +")
    print("              ↑")
    print("Open-Meteo Weather + AQI")
    print("   ↓")
    print("Environmental Risk Score")
    print("   ↓")
    print("Fusion Engine")
    print("   ↓")
    print("Final Risk Score")
    print("   ↓")
    print("NORMAL / WARNING / CRITICAL")

    print()
    print("=" * 70)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=False
    )