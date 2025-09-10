import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SQLALCHEMY_DATABASE_URI = "postgresql://logs_ip0q_user:pJX2IZ1ZBUQU8lshtY5Hj6HVViEigjGP@dpg-d2t1696r433s73cusr3g-a.frankfurt-postgres.render.com:5432/logs_ip0q"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = "supersecretkey"
