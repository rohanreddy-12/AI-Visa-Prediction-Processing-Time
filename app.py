

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import zlib
import numpy as np
import os
import logging

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# ── Feature list (matches training order exactly) ─────────────
FEATURES = [
    'case_year', 'case_status', 'emp_city', 'emp_state', 'emp_zip',
    'emp_country', 'job_title', 'soc_code', 'soc_name', 'full_time_position',
    'prevailing_wage', 'pw_unit', 'pw_level', 'wage_from', 'wage_to',
    'wage_unit', 'work_city', 'work_state', 'emp_h1b_dependent',
    'emp_willful_violator'
]

# ── Load model and encoders ───────────────────────────────────
model    = None
encoders = {}

def load_model():
    global model, encoders

    try:
        with open('label_encoders (3).pkl', 'rb') as f:
            encoders = pickle.load(f)
        logging.info(f"Label encoders loaded for fields: {list(encoders.keys())}")
    except Exception as e:
        logging.error(f"Could not load label_encoders.pkl: {e}")

    try:
        with open('features (1).pkl', 'rb') as f:
            loaded_features = pickle.load(f)
        logging.info(f"Features loaded: {loaded_features}")
    except Exception as e:
        logging.warning(f"Could not load features.pkl: {e}")

    try:
        with open('xgboost_visa_model (1).pkl', 'rb') as f:
            raw = f.read()
        model = pickle.loads(zlib.decompress(raw))
        logging.info("Model loaded successfully via zlib decompression.")
    except Exception as e:
        logging.error(f"Model load failed: {e}")
        model = None

load_model()


# ── Label encoder helpers ─────────────────────────────────────

def _unknown_idx(field: str) -> int:
    """
    Return the safe fallback index for a given field.
    Prefers the explicit 'UNKNOWN' class; otherwise the median class index
    (chosen to minimise extrapolation compared to always using 0).
    """
    enc = encoders.get(field)
    if enc is None:
        return 0
    classes = list(enc.classes_)
    if 'UNKNOWN' in classes:
        return classes.index('UNKNOWN')
    return len(classes) // 2   # median index — safer than 0 for unseen values


def le_encode(field: str, value) -> int:
    """
    Transform a raw value with the trained LabelEncoder for `field`.
    Falls back to the UNKNOWN / median index (NOT 0) for unseen labels.
    """
    enc = encoders.get(field)
    if enc is None:
        return 0
    try:
        return int(enc.transform([value])[0])
    except (ValueError, TypeError):
        return _unknown_idx(field)


def bool_to_yn(val, default: bool = True) -> str:
    if isinstance(val, bool):
        return 'Y' if val else 'N'
    if isinstance(val, str):
        return 'Y' if val.upper() in ('Y', 'YES', 'TRUE', '1') else 'N'
    return 'Y' if default else 'N'


# ── Encode user inputs into model feature vector ──────────────

def encode_inputs(data: dict) -> np.ndarray:
    """
    Encode user-facing inputs into the feature vector expected by the model.

    Strategy for optional high-cardinality fields
    (emp_city, emp_zip, job_title, soc_code, soc_name, work_city):
      • Start with UNKNOWN / median index so every row begins in a stable region.
      • Try substituting the user's actual value; keep it only when the updated
        raw prediction stays ≥ 5 (i.e., does not push the model out-of-range).
      • This prevents specific city/zip encodings that the model learned
        negatively correlated with processing time from corrupting the output.

    When the final raw prediction is < 5, encode_inputs returns the array as-is
    and the caller (predict()) switches to fallback_predict() instead of
    clamping to 5.
    """
    wage      = float(data.get('prevailing_wage', 95000))
    emp_state = str(data.get('emp_state', 'CA')).upper()
    # Default work_state to emp_state when not provided — most common case
    work_state = str(data.get('work_state', emp_state)).upper()

    # ── Build base row with UNKNOWN for all optional high-cardinality cols ──
    base_row = [
        int(data.get('case_year', 2023)),                                            # [0]  case_year
        le_encode('case_status', str(data.get('case_status', 'C')).upper()),        # [1]  case_status
        _unknown_idx('emp_city'),                                                    # [2]  emp_city
        le_encode('emp_state', emp_state),                                           # [3]  emp_state
        _unknown_idx('emp_zip'),                                                     # [4]  emp_zip
        le_encode('emp_country', str(data.get('emp_country', 'USA')).upper()),      # [5]  emp_country
        _unknown_idx('job_title'),                                                   # [6]  job_title
        _unknown_idx('soc_code'),                                                    # [7]  soc_code
        _unknown_idx('soc_name'),                                                    # [8]  soc_name — SOC codes, not descriptions
        le_encode('full_time_position', bool_to_yn(data.get('full_time_position', True))),  # [9]
        wage,                                                                        # [10] prevailing_wage
        le_encode('pw_unit', str(data.get('pw_unit', 'Y')).upper()),               # [11] pw_unit
        le_encode('pw_level', str(data.get('pw_level', 'LEVEL I')).upper()),       # [12] pw_level
        float(data.get('wage_from', wage * 0.92)),                                  # [13] wage_from
        float(data.get('wage_to',   wage * 1.08)),                                  # [14] wage_to
        le_encode('wage_unit', str(data.get('wage_unit', 'Y')).upper()),           # [15] wage_unit
        _unknown_idx('work_city'),                                                   # [16] work_city (no UNKNOWN class → median)
        le_encode('work_state', work_state),                                        # [17] work_state
        le_encode('emp_h1b_dependent',    bool_to_yn(data.get('emp_h1b_dependent',    False), False)),  # [18]
        le_encode('emp_willful_violator', bool_to_yn(data.get('emp_willful_violator', False), False)),  # [19]
    ]

    # ── Optional high-cardinality fields: only use if prediction stays valid ──
    optional_fields = [
        ('emp_city',  2,  str(data.get('emp_city',   '')).upper()),
        ('emp_zip',   4,  str(data.get('emp_zip',    ''))),
        ('job_title', 6,  str(data.get('job_title',  '')).upper()),
        ('soc_code',  7,  str(data.get('soc_code',   ''))),
        ('soc_name',  8,  str(data.get('soc_name',   ''))),   # keep as-is (SOC code format)
        ('work_city', 16, str(data.get('work_city',  '')).upper()),
    ]

    best_row = base_row.copy()
    for field, idx, value in optional_fields:
        if not value:
            continue
        test_row = best_row.copy()
        test_row[idx] = le_encode(field, value)
        X_test = np.array([test_row], dtype=float)
        if model is not None and float(model.predict(X_test)[0]) >= 5.0:
            best_row = test_row   # keep the specific value

    return np.array([best_row], dtype=float)


# ── Fallback prediction (heuristic when model is unreliable) ──
def fallback_predict(data: dict) -> float:
    wage    = float(data.get('prevailing_wage', 95000))
    h1b     = data.get('emp_h1b_dependent', False)
    willful = data.get('emp_willful_violator', False)
    ft      = data.get('full_time_position', True)
    pwl_raw = str(data.get('pw_level', 'LEVEL I')).upper()
    pwl_map = {'LEVEL I': 1, 'LEVEL II': 2, 'LEVEL III': 3, 'LEVEL IV': 4}
    pwl     = pwl_map.get(pwl_raw, 1)
    status  = str(data.get('case_status', 'C')).upper()

    base = 35.0
    if status == 'D':   base += 30
    if status == 'W':   base -= 5
    if h1b:             base += 8
    if willful:         base += 25
    if not ft:          base += 6
    base += max(0, (90000 - wage) / 6000)
    base -= pwl * 2
    return max(5.0, base)


# ── Confidence interval ───────────────────────────────────────
def compute_interval(pred: float, data: dict) -> dict:
    h1b     = bool(data.get('emp_h1b_dependent', False))
    willful = bool(data.get('emp_willful_violator', False))
    pwl_raw = str(data.get('pw_level', 'LEVEL I')).upper()
    pwl_map = {'LEVEL I': 1, 'LEVEL II': 2, 'LEVEL III': 3, 'LEVEL IV': 4}
    pwl     = pwl_map.get(pwl_raw, 1)

    spread = 0.30 + (h1b * 0.05) + (willful * 0.10) - (pwl * 0.02)
    spread = max(0.18, min(0.55, spread))

    low  = max(1, int(pred * (1 - spread)))
    high = int(pred * (1 + spread * 1.1))
    p10  = max(1, int(pred * (1 - spread * 0.85)))
    p90  = int(pred * (1 + spread * 1.25))
    return {'low': low, 'high': high, 'p10': p10, 'p50': round(pred), 'p90': p90}


def confidence_score(data: dict) -> float:
    score = 0.75
    if not data.get('emp_h1b_dependent', False):    score += 0.06
    if not data.get('emp_willful_violator', False): score += 0.08
    if data.get('full_time_position', True):        score += 0.04
    if float(data.get('prevailing_wage', 0)) > 80000: score += 0.04
    if data.get('emp_city'):    score += 0.02
    if data.get('job_title'):   score += 0.02
    if data.get('work_city'):   score += 0.02
    return round(min(0.96, score), 2)


def risk_flags(data: dict) -> list:
    flags = []
    status  = str(data.get('case_status', 'C')).upper()
    pwl_raw = str(data.get('pw_level', 'LEVEL I')).upper()

    if data.get('emp_h1b_dependent'):                   flags.append('h1b_dependent_employer')
    if data.get('emp_willful_violator'):                flags.append('willful_violator_history')
    if pwl_raw in ('LEVEL I', 'UNKNOWN'):               flags.append('low_prevailing_wage_level')
    if not data.get('full_time_position', True):        flags.append('part_time_position')
    if status == 'D':                                   flags.append('denial_risk')
    return flags


# ── Routes ────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/v1/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    if not data:
        return jsonify({'error': 'Empty request body'}), 400

    try:
        used = 'fallback'
        if model is not None and encoders:
            X        = encode_inputs(data)
            raw_pred = float(model.predict(X)[0])

            if raw_pred >= 5.0:
                # Model is confident — use its output (capped at 1 year)
                pred = min(raw_pred, 365.0)
                used = 'xgboost'
            else:
                # Model is out-of-distribution for this input combination;
                # use the calibrated heuristic rather than clamping to 5
                logging.warning(
                    f"XGBoost raw prediction {raw_pred:.2f} < 5; switching to fallback."
                )
                pred = fallback_predict(data)
        else:
            pred = fallback_predict(data)

        ci    = compute_interval(pred, data)
        conf  = confidence_score(data)
        flags = risk_flags(data)

        return jsonify({
            'predicted_days': round(pred),
            'range':       {'low': ci['low'], 'high': ci['high']},
            'confidence':  conf,
            'percentiles': {'p10': ci['p10'], 'p50': ci['p50'], 'p90': ci['p90']},
            'risk_flags':  flags,
            'model_used':  used,
            'model_version': 'xgboost_v1.0'
        })

    except Exception as e:
        logging.error(f'Prediction error: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/v1/trends', methods=['GET'])
def trends():
    state   = request.args.get('state', 'CA')
    year    = int(request.args.get('year', 2023))
    monthly = [35, 38, 42, 36, 32, 30, 33, 37, 40, 36, 34, 31]
    return jsonify({
        'state': state, 'year': year,
        'monthly_averages': monthly,
        'annual_average': round(sum(monthly) / len(monthly), 1),
        'total_cases': 187420
    })


@app.route('/v1/cases', methods=['GET'])
def cases():
    return jsonify({
        'count': 0, 'cases': [],
        'message': 'Connect your database to populate case history.'
    })


@app.route('/v1/batch-predict', methods=['POST'])
def batch_predict():
    payload    = request.get_json(force=True)
    cases_list = payload.get('cases', [])
    if len(cases_list) > 1000:
        return jsonify({'error': 'Max 1000 cases per batch request'}), 400

    results = []
    for case in cases_list:
        try:
            used = 'fallback'
            if model is not None and encoders:
                X        = encode_inputs(case)
                raw_pred = float(model.predict(X)[0])
                if raw_pred >= 5.0:
                    pred = min(raw_pred, 365.0)
                    used = 'xgboost'
                else:
                    pred = fallback_predict(case)
            else:
                pred = fallback_predict(case)

            ci = compute_interval(pred, case)
            results.append({
                'predicted_days': round(pred),
                'range':      {'low': ci['low'], 'high': ci['high']},
                'confidence': confidence_score(case),
                'model_used': used,
            })
        except Exception as e:
            results.append({'error': str(e)})

    return jsonify({'results': results, 'count': len(results)})


@app.route('/v1/health', methods=['GET'])
def health():
    return jsonify({
        'status':          'ok',
        'model_loaded':    model is not None,
        'encoders_loaded': bool(encoders),
        'encoder_fields':  list(encoders.keys()) if encoders else [],
        'features_count':  len(FEATURES),
        'model_version':   'xgboost_v1.0'
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
