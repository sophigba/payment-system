from flask import Flask, request, jsonify
from flask_cors import CORS
from models import db, Student, TransactionLog, SystemLog, Anomaly
from settings import Config
import joblib
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)
CORS(app)  # Allow future front-end integration

# Load trained anomaly detection model if available
MODEL_PATH = os.path.join(os.path.dirname(__file__), "anomaly_detector.pkl")
model = None
model_features = ["cpu_usage", "memory_usage", "wifi_signal", "reader_response", "error_rate"]

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        app.logger.info(f"Loaded anomaly model from {MODEL_PATH}")
    else:
        app.logger.warning(f"No model found at {MODEL_PATH}. /predict and anomaly detection disabled.")
except Exception as e:
    app.logger.exception("Failed to load anomaly model: %s", e)
    model = None

# ------------------------------------------------------
# Helpers for serializing objects into JSON
# ------------------------------------------------------
def student_to_dict(student):
    return {
        "uid": student.uid,
        "name": student.name,
        "phone": student.phone,
        "status": getattr(student, "status", "active"),
        "created_at": student.created_at.isoformat() if student.created_at else None
    }

def transaction_to_dict(tx):
    return {
        "tid": tx.tid,
        "uid": tx.uid,
        "amount": int(tx.amount),
        "timestamp": tx.timestamp.isoformat() if tx.timestamp else None
    }

def system_log_to_dict(log):
    return {
        "log_id": log.log_id,
        "timestamp": log.timestamp.isoformat() if log.timestamp else None,
        "cpu_usage": log.cpu_usage,
        "memory_usage": log.memory_usage,
        "wifi_signal": log.wifi_signal,
        "reader_response": log.reader_response,
        "error_rate": log.error_rate,
        "anomaly": log.anomaly,
    }

def anomaly_to_dict(anomaly):
    return {
        "anomaly_id": anomaly.anomaly_id,
        "type": anomaly.type,
        "source": anomaly.source,
        "details": anomaly.details,
        "severity": anomaly.severity,
        "timestamp": anomaly.timestamp.isoformat() if anomaly.timestamp else None
    }

def to_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

# ------------------------------------------------------
# STUDENT MANAGEMENT ENDPOINTS
# ------------------------------------------------------

# 1. Register student / card
@app.route("/register_student", methods=["POST"])
def register_student():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"status": "error", "message": "Invalid or missing JSON"}), 400

    uid = data.get("uid")
    name = data.get("name")
    phone = data.get("phone")

    if not uid or not name:
        return jsonify({"status": "error", "message": "UID and name are required"}), 400

    if Student.query.get(uid):
        return jsonify({"status": "error", "message": "Student already exists"}), 400

    student = Student(uid=uid, name=name, phone=phone)
    setattr(student, "status", "active")
    db.session.add(student)
    db.session.commit()

    return jsonify({"status": "success", "message": "Student registered", "student": student_to_dict(student)})

# 2. Get all students
@app.route("/students", methods=["GET"])
def get_all_students():
    students = Student.query.all()
    return jsonify({"status": "success", "students": [student_to_dict(s) for s in students]})

# 3. Update student status (block/unblock/unregister)
@app.route("/update_status", methods=["POST"])
def update_status():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"status": "error", "message": "Invalid or missing JSON"}), 400

    uid = data.get("uid")
    new_status = data.get("status")

    if not uid or new_status not in ["active", "blocked", "unregistered"]:
        return jsonify({
            "status": "error",
            "message": "UID and valid status (active/blocked/unregistered) are required"
        }), 400

    student = Student.query.get(uid)
    if not student:
        return jsonify({"status": "error", "message": "Student not found"}), 404

    setattr(student, "status", new_status)
    db.session.commit()

    return jsonify({
        "status": "success",
        "message": f"Student {uid} status updated to {new_status}",
        "student": student_to_dict(student)
    })

# 4. Shortcut endpoints for buttons
@app.route("/block_card", methods=["POST"])
def block_card():
    return update_status_wrapper("blocked")

@app.route("/unblock_card", methods=["POST"])
def unblock_card():
    return update_status_wrapper("active")

@app.route("/unregister_card", methods=["POST"])
def unregister_card():
    return update_status_wrapper("unregistered")

def update_status_wrapper(new_status):
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"status": "error", "message": "Invalid or missing JSON"}), 400

    uid = data.get("uid")
    if not uid:
        return jsonify({"status": "error", "message": "UID is required"}), 400

    student = Student.query.get(uid)
    if not student:
        return jsonify({"status": "error", "message": "Student not found"}), 404

    setattr(student, "status", new_status)
    db.session.commit()

    return jsonify({
        "status": "success",
        "message": f"Student {uid} status updated to {new_status}",
        "student": student_to_dict(student)
    })

# 5. Recharge card
@app.route("/recharge_card", methods=["POST"])
def recharge_card():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"status": "error", "message": "Invalid or missing JSON"}), 400

    uid = data.get("uid")
    amount = data.get("amount")

    if not uid or amount is None:
        return jsonify({"status": "error", "message": "UID and amount are required"}), 400

    student = Student.query.get(uid)
    if not student:
        return jsonify({"status": "error", "message": "Student not found"}), 404

    if getattr(student, "status", "active") != "active":
        return jsonify({"status": "error", "message": "Recharge not allowed. Student is not active."}), 403

    # ✅ Update balance instead of logging a transaction
    try:
        student.balance = int(student.balance or 0) + int(amount)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": f"Recharge failed: {str(e)}"}), 500

    return jsonify({
        "status": "success",
        "message": "Balance updated successfully",                                                                                                 
        "uid": student.uid,
        "new_balance": int(student.balance)
    })

# ------------------------------------------------------
# ANOMALIES DASHBOARD ENDPOINT
# ------------------------------------------------------
@app.route("/anomalies_dashboard", methods=["GET"])
def anomalies_dashboard():
    total_anomalies = Anomaly.query.count()
    high = Anomaly.query.filter_by(severity="High").count()
    medium = Anomaly.query.filter_by(severity="Medium").count()
    low = Anomaly.query.filter_by(severity="Low").count()

    system_logs = SystemLog.query.count()
    anomaly_logs = SystemLog.query.filter_by(anomaly=1).count()
    anomaly_rate = round((anomaly_logs / system_logs) * 100, 2) if system_logs else 0

    return jsonify({
        "status": "success",
        "total_anomalies": total_anomalies,
        "severity_distribution": {"High": high, "Medium": medium, "Low": low},
        "system_logs": system_logs,
        "anomaly_rate": anomaly_rate
    })

# ------------------------------------------------------
# SYSTEM MANAGEMENT ENDPOINTS
# ------------------------------------------------------
@app.route("/reset_system", methods=["POST"])
def reset_system():
    try:
        num_tx = TransactionLog.query.delete()
        num_anomalies = Anomaly.query.delete()
        db.session.commit()
        return jsonify({"status": "success", "message": f"System reset: {num_tx} transactions, {num_anomalies} anomalies cleared"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/recent_transactions", methods=["GET"])
def recent_transactions():
    txs = TransactionLog.query.order_by(TransactionLog.timestamp.desc()).limit(10).all()
    return jsonify([transaction_to_dict(tx) for tx in txs])

@app.route("/system_logs", methods=["GET", "POST"])
def system_logs():
    if request.method == "GET":
        logs = SystemLog.query.order_by(SystemLog.timestamp.desc()).limit(10).all()
        return jsonify([system_log_to_dict(log) for log in logs])

    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"status": "error", "message": "Invalid or missing JSON"}), 400

    ts = datetime.utcnow()
    try:
        ts = datetime.fromisoformat(data.get("timestamp")) if data.get("timestamp") else datetime.utcnow()
    except Exception:
        ts = datetime.utcnow()

    telemetry = {
        "cpu_usage": to_float(data.get("cpu_usage")),
        "memory_usage": to_float(data.get("memory_usage")),
        "wifi_signal": to_float(data.get("wifi_signal")),
        "reader_response": to_float(data.get("reader_response")),
        "error_rate": to_float(data.get("error_rate")),
    }

    log = SystemLog(
        cpu_usage=telemetry["cpu_usage"],
        memory_usage=telemetry["memory_usage"],
        wifi_signal=telemetry["wifi_signal"],
        reader_response=telemetry["reader_response"],
        error_rate=telemetry["error_rate"],
        anomaly=0,
        timestamp=ts
    )
    db.session.add(log)
    db.session.commit()

    if model and all(not np.isnan(v) for v in telemetry.values()):
        try:
            features = np.array([[telemetry[f] for f in model_features]], dtype=float)
            pred = model.predict(features)[0]
            if pred == -1:
                log.anomaly = 1
                new_anomaly = Anomaly(
                    type="System",
                    source="Sensor",
                    details=str(telemetry),
                    severity="High",
                    timestamp=datetime.utcnow()
                )
                db.session.add(new_anomaly)
                db.session.commit()
                return jsonify({"status": "success", "message": "Log saved, anomaly detected", "log": system_log_to_dict(log)}), 201
        except Exception as e:
            app.logger.exception("Prediction failed: %s", e)

    return jsonify({"status": "success", "message": "System log saved", "log": system_log_to_dict(log)}), 201

@app.route("/anomalies", methods=["GET"])
def anomalies():
    anomalies_list = Anomaly.query.order_by(Anomaly.timestamp.desc()).limit(10).all()
    return jsonify([anomaly_to_dict(a) for a in anomalies_list])

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"status": "error", "message": "No model loaded"}), 503

    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"status": "error", "message": "Invalid or missing JSON"}), 400

    features = np.array([[to_float(data.get(f)) for f in model_features]], dtype=float)
    if np.any(np.isnan(features)):
        return jsonify({"status": "error", "message": "All features required and must be numeric"}), 400

    pred = model.predict(features)[0]
    result = "Anomaly" if pred == -1 else "Normal"
    return jsonify({"status": "success", "prediction": result})

@app.route("/log_transaction", methods=["POST"])
def log_transaction():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"status": "error", "message": "Invalid or missing JSON"}), 400

    uid = data.get("uid")
    amount = data.get("amount")
    ts = datetime.utcnow()

    if not uid or amount is None:
        return jsonify({"status": "error", "message": "UID and amount are required"}), 400

    student = Student.query.get(uid)
    if not student:
        return jsonify({"status": "error", "message": "Student not found"}), 404

    # Check if student is active
    if getattr(student, "status", "active") != "active":
        return jsonify({"status": "error", "message": "Transaction denied. Card inactive."}), 403

    try:
        # Deduct amount from student balance
        student.balance = int(student.balance or 0) - int(amount)
        tx = TransactionLog(uid=uid, amount=int(amount), timestamp=ts)
        db.session.add(tx)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": f"Transaction failed: {str(e)}"}), 500

    return jsonify({
        "status": "success",
        "message": "Transaction logged successfully",
        "transaction": transaction_to_dict(tx)
    }), 201

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        print("Tables created successfully!")
    app.run(debug=True)
