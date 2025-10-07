from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Student(db.Model):
    __tablename__ = 'students'
    uid = db.Column(db.String(50), primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(15))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default="active", nullable=False)
    balance = db.Column(db.Integer, default=0)

    def is_blocked(self):
        return self.status == "blocked"


class TransactionLog(db.Model):
    __tablename__ = 'transaction_logs'
    tid = db.Column(db.Integer, primary_key=True, autoincrement=True)
    uid = db.Column(db.String(50), db.ForeignKey('students.uid'), nullable=False)
    amount = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


class SystemLog(db.Model):
    __tablename__ = 'system_logs'

    log_id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp(), nullable=False)
    memory_usage = db.Column(db.Float, nullable=False)    # Memory usage in %
    wifi_signal = db.Column(db.Float, nullable=False)     # Wi-Fi signal strength in dBm
    reader_response = db.Column(db.Float, nullable=False) # NFC reader response time in ms
    error_rate = db.Column(db.Float, nullable=False)      # Error rate in %
    anomaly = db.Column(db.Integer, nullable=False)       # 0 = normal, 1 = anomaly


class Anomaly(db.Model):
    __tablename__ = 'anomalies'
    anomaly_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    type = db.Column(db.String(50))
    source = db.Column(db.String(50))
    details = db.Column(db.Text)
    severity = db.Column(db.String(20))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
