"""
An Interactive Web Portal for Keeping Records of Diabetes-Affected Patients in Pakistan
=======================================================================================
Single-file Flask backend  +  SQLite database  +  scikit-learn ML module.

Run:
    pip install -r requirements.txt
    python app.py
Open http://127.0.0.1:5000

Search this file for  "# >>> UPDATE"  to find every place you should change.

Sections
    1. CONFIG ............ secret key, DB path, admin invite code      (UPDATE)
    2. DATABASE .......... schema + helpers + demo seed data           (UPDATE)
    3. SECURITY .......... CSRF, password hashing, role decorators
    4. ML MODULE ......... diabetes risk model (Random Forest)         (UPDATE)
    5. ROUTES ............ End-user / Doctor / Admin
    6. TEMPLATES ......... Bootstrap 5 pages (inline, no extra files)
"""
import os
import re
import secrets
import sqlite3
from datetime import date, datetime
from functools import wraps

import joblib
import numpy as np
import pandas as pd
from flask import (Flask, abort, flash, g, jsonify, redirect, render_template,
                   request, session, url_for)
from jinja2 import DictLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from werkzeug.security import check_password_hash, generate_password_hash

# =============================================================================
# 1. CONFIG
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# >>> UPDATE: set a long random SECRET_KEY in your environment before deployment.
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-only-change-me")

# >>> UPDATE: invite code required to create an Admin account (stops random people
#     from registering as admin). Change it and keep it private.
ADMIN_INVITE_CODE = os.environ.get("ADMIN_INVITE_CODE", "ADMIN-2026")

DB_PATH = os.path.join(BASE_DIR, "data", "portal.db")
MODEL_PATH = os.path.join(BASE_DIR, "data", "risk_model.joblib")

# >>> UPDATE (ML): put the Pima Indians Diabetes CSV here to train on real data.
#     Columns: Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,
#              DiabetesPedigreeFunction,Age,Outcome
DATASET_PATH = os.path.join(BASE_DIR, "data", "diabetes.csv")

REGIONS = ["Punjab", "Sindh", "Khyber Pakhtunkhwa", "Balochistan",
           "Islamabad Capital Territory", "Gilgit-Baltistan", "Azad Jammu & Kashmir"]
DIABETES_TYPES = ["Type 1", "Type 2", "Gestational", "Pre-diabetic", "Other"]

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")
# >>> UPDATE: when you host it over HTTPS also set SESSION_COOKIE_SECURE=True


# =============================================================================
# 2. DATABASE
# =============================================================================
SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL CHECK(role IN ('patient','doctor','admin')),
    full_name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    phone TEXT,
    password_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'approved' CHECK(status IN ('pending','approved','rejected')),
    -- doctor / healthcare-unit only
    unit_name TEXT, license_no TEXT, speciality TEXT, city TEXT, region TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uid TEXT UNIQUE,                       -- unique ID used for searching, e.g. PK-DM-00001
    user_id INTEGER UNIQUE REFERENCES users(id),
    full_name TEXT NOT NULL,
    age INTEGER, gender TEXT, region TEXT, city TEXT, contact TEXT,
    diabetes_type TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL REFERENCES patients(id),
    doctor_id  INTEGER NOT NULL REFERENCES users(id),
    visit_date TEXT NOT NULL,
    medical_history TEXT,
    fasting_glucose REAL, hba1c REAL, bmi REAL, systolic_bp INTEGER,
    treatment_plan TEXT, care_precautions TEXT, progress_remarks TEXT,
    next_appointment TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sender_id INTEGER NOT NULL REFERENCES users(id),
    receiver_id INTEGER NOT NULL REFERENCES users(id),
    message TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS content (               -- admin-managed announcements
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL, body TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS access_log (            -- who viewed which record (privacy audit)
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER, patient_uid TEXT, action TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def q(sql, args=(), one=False):
    cur = get_db().execute(sql, args)
    rows = cur.fetchall()
    return (rows[0] if rows else None) if one else rows


def run(sql, args=()):
    db = get_db()
    cur = db.execute(sql, args)
    db.commit()
    return cur.lastrowid


def create_patient_row(full_name, user_id=None, age=None, gender=None, region=None,
                       city=None, contact=None, diabetes_type=None):
    pid = run("INSERT INTO patients(user_id,full_name,age,gender,region,city,contact,diabetes_type)"
              " VALUES(?,?,?,?,?,?,?,?)",
              (user_id, full_name, age, gender, region, city, contact, diabetes_type))
    uid = f"PK-DM-{pid:05d}"
    run("UPDATE patients SET uid=? WHERE id=?", (uid, pid))
    return pid, uid


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.executescript(SCHEMA)
    db.commit()
    db.close()


def seed_demo_data():
    """Creates demo accounts so your presentation works immediately.
    >>> UPDATE: delete this function (and its call at the bottom) before real use."""
    with app.app_context():
        if q("SELECT 1 FROM users LIMIT 1"):
            return
        import random
        random.seed(7)
        pw = generate_password_hash("Password@123")
        admin = run("INSERT INTO users(role,full_name,email,password_hash) VALUES('admin','System Admin','admin@diabetes.pk',?)",
                    (generate_password_hash("Admin@123"),))
        d1 = run("INSERT INTO users(role,full_name,email,phone,password_hash,status,unit_name,license_no,speciality,city,region)"
                 " VALUES('doctor','Dr. Ayesha Khan','doctor@diabetes.pk','0300-1111111',?,'approved','Mayo Diabetes Clinic','PMDC-12345','Endocrinologist','Lahore','Punjab')", (pw,))
        run("INSERT INTO users(role,full_name,email,phone,password_hash,status,unit_name,license_no,speciality,city,region)"
            " VALUES('doctor','Dr. Imran Shah','pending@diabetes.pk','0300-2222222',?,'pending','Shah Medical Centre','PMDC-67890','General Physician','Karachi','Sindh')", (pw,))
        names = ["Ali Raza", "Fatima Bibi", "Usman Tariq", "Sana Malik", "Bilal Ahmed", "Hina Siddiqui",
                 "Kamran Yousaf", "Zainab Noor", "Farhan Iqbal", "Maryam Aslam", "Hamza Butt", "Nadia Gul"]
        cities = {"Punjab": "Lahore", "Sindh": "Karachi", "Khyber Pakhtunkhwa": "Peshawar",
                  "Balochistan": "Quetta", "Islamabad Capital Territory": "Islamabad"}
        for i, n in enumerate(names):
            region = random.choice(list(cities))
            uid_user = None
            if i == 0:  # one demo patient who can log in
                uid_user = run("INSERT INTO users(role,full_name,email,password_hash,region) VALUES('patient',?,?,?,?)",
                               (n, "patient@diabetes.pk", pw, region))
            pid, _ = create_patient_row(n, uid_user, random.randint(25, 70), random.choice(["Male", "Female"]),
                                        region, cities[region], f"03{random.randint(0,4)}{random.randint(1000000,9999999)}",
                                        random.choice(DIABETES_TYPES[:3]))
            hb = random.uniform(6.5, 11)
            for m in range(1, 5):  # 4 visits per patient
                run("INSERT INTO records(patient_id,doctor_id,visit_date,medical_history,fasting_glucose,hba1c,bmi,systolic_bp,"
                    "treatment_plan,care_precautions,progress_remarks,next_appointment) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (pid, d1, f"2026-{m+4:02d}-{random.randint(1,25):02d}", "Family history of diabetes; hypertension.",
                     round(hb * 18 + random.uniform(-10, 10), 0), round(hb - m * 0.15, 1), round(random.uniform(22, 35), 1),
                     random.randint(110, 150), "Metformin 500mg twice daily; diet plan.",
                     "Avoid sugary drinks, 30 min walk daily, check feet daily.",
                     "Improving." if m > 1 else "Initial visit.", f"2026-{m+5:02d}-15"))
        run("INSERT INTO content(title,body) VALUES(?,?)",
            ("Free diabetes screening camp", "Healthcare units can register on the portal to share patient records securely."))


# =============================================================================
# 3. SECURITY  (CSRF, auth, roles)
# =============================================================================
@app.before_request
def csrf_protect():
    if request.method == "POST":
        token = session.get("_csrf")
        if not token or token != request.form.get("_csrf"):
            abort(400, "Invalid or missing CSRF token")


def csrf_token():
    if "_csrf" not in session:
        session["_csrf"] = secrets.token_hex(16)
    return session["_csrf"]


app.jinja_env.globals["csrf_token"] = csrf_token


def current_user():
    if "uid" not in session:
        return None
    if "user" not in g:
        g.user = q("SELECT * FROM users WHERE id=?", (session["uid"],), one=True)
    return g.user


app.jinja_env.globals["current_user"] = current_user


def login_required(*roles):
    def deco(fn):
        @wraps(fn)
        def wrapper(*a, **kw):
            u = current_user()
            if not u:
                flash("Please sign in first.", "warning")
                return redirect(url_for("login"))
            if roles and u["role"] not in roles:
                abort(403)
            if u["role"] == "doctor" and u["status"] != "approved":
                flash("Your account is awaiting admin approval.", "warning")
                session.clear()
                return redirect(url_for("login"))
            return fn(*a, **kw)
        return wrapper
    return deco


def valid_email(e):
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", e or ""))


def log_access(patient_uid, action):
    u = current_user()
    run("INSERT INTO access_log(user_id,patient_uid,action) VALUES(?,?,?)", (u["id"], patient_uid, action))


def num(v, cast=float):
    try:
        return cast(v) if v not in (None, "") else None
    except ValueError:
        return None


# =============================================================================
# 4. ML MODULE  - diabetes risk prediction
# =============================================================================
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
_model_cache = {}


def load_training_data():
    """>>> UPDATE (ML): best practice is to place the real Pima Indians Diabetes CSV at
    data/diabetes.csv. If it is missing we generate SYNTHETIC data so the demo still
    runs - say this clearly in your presentation."""
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        # Pima data stores missing values as 0 in these columns
        for c in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
            df[c] = df[c].replace(0, np.nan)
            df[c] = df[c].fillna(df[c].median())
        return df[FEATURES + ["Outcome"]], "Pima Indians Diabetes (real data)"
    rng = np.random.default_rng(42)
    n = 1500
    df = pd.DataFrame({
        "Pregnancies": rng.integers(0, 10, n),
        "Glucose": rng.normal(120, 32, n).clip(50, 250),
        "BloodPressure": rng.normal(72, 12, n).clip(40, 120),
        "SkinThickness": rng.normal(29, 9, n).clip(5, 60),
        "Insulin": rng.normal(110, 60, n).clip(10, 400),
        "BMI": rng.normal(32, 6.5, n).clip(16, 60),
        "DiabetesPedigreeFunction": rng.gamma(2, 0.25, n).clip(0.08, 2.4),
        "Age": rng.integers(21, 80, n),
    })
    logit = (0.035 * (df.Glucose - 120) + 0.07 * (df.BMI - 32) + 0.025 * (df.Age - 40)
             + 0.9 * (df.DiabetesPedigreeFunction - 0.5) + 0.01 * (df.BloodPressure - 72) - 0.3)
    df["Outcome"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df, "SYNTHETIC demo data (replace with data/diabetes.csv)"


def get_model():
    """Train once, cache on disk + memory. Delete data/risk_model.joblib to retrain."""
    if "m" in _model_cache:
        return _model_cache["m"]
    if os.path.exists(MODEL_PATH):
        bundle = joblib.load(MODEL_PATH)
    else:
        df, source = load_training_data()
        X_tr, X_te, y_tr, y_te = train_test_split(df[FEATURES], df["Outcome"], test_size=0.2,
                                                  random_state=42, stratify=df["Outcome"])
        # >>> UPDATE (ML): try other models / tune hyper-parameters (GridSearchCV),
        #     e.g. LogisticRegression, XGBoost, GradientBoosting - compare in your slides.
        clf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5,
                                     class_weight="balanced", random_state=42)
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)[:, 1]
        bundle = {"clf": clf, "source": source, "n_rows": len(df),
                  "accuracy": round(accuracy_score(y_te, proba > 0.5), 3),
                  "auc": round(roc_auc_score(y_te, proba), 3),
                  "importance": dict(zip(FEATURES, map(float, clf.feature_importances_))),
                  "medians": df[FEATURES].median().to_dict()}
        joblib.dump(bundle, MODEL_PATH)
    _model_cache["m"] = bundle
    return bundle


def predict_risk(values: dict):
    b = get_model()
    row = pd.DataFrame([{f: values.get(f) if values.get(f) is not None else b["medians"][f] for f in FEATURES}])
    p = float(b["clf"].predict_proba(row)[0, 1])
    level = "High" if p >= 0.66 else "Moderate" if p >= 0.33 else "Low"
    return p, level


# =============================================================================
# 5. ROUTES
# =============================================================================
# ---------- public ----------
@app.route("/")
def home():
    stats = {
        "patients": q("SELECT COUNT(*) c FROM patients", one=True)["c"],
        "doctors": q("SELECT COUNT(*) c FROM users WHERE role='doctor' AND status='approved'", one=True)["c"],
        "records": q("SELECT COUNT(*) c FROM records", one=True)["c"],
    }
    news = q("SELECT * FROM content ORDER BY id DESC LIMIT 5")
    return render_template("home.html", stats=stats, news=news)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    """End-user (patient) sign-up."""
    if request.method == "POST":
        f = request.form
        name, email, pw = f.get("full_name", "").strip(), f.get("email", "").strip().lower(), f.get("password", "")
        if not name or not valid_email(email) or len(pw) < 8:
            flash("Enter a name, valid email and a password of at least 8 characters.", "danger")
        elif q("SELECT 1 FROM users WHERE email=?", (email,)):
            flash("This email is already registered.", "danger")
        else:
            uid_user = run("INSERT INTO users(role,full_name,email,phone,password_hash,region) VALUES('patient',?,?,?,?,?)",
                           (name, email, f.get("phone"), generate_password_hash(pw), f.get("region")))
            _, puid = create_patient_row(name, uid_user, num(f.get("age"), int), f.get("gender"),
                                         f.get("region"), f.get("city"), f.get("phone"), f.get("diabetes_type"))
            flash(f"Account created! Your unique Patient ID is {puid}. Please sign in.", "success")
            return redirect(url_for("login"))
    return render_template("signup.html", regions=REGIONS, types=DIABETES_TYPES)


@app.route("/signup/doctor", methods=["GET", "POST"])
def signup_doctor():
    """Doctor / healthcare unit registration - needs admin approval."""
    if request.method == "POST":
        f = request.form
        email, pw = f.get("email", "").strip().lower(), f.get("password", "")
        if not f.get("full_name") or not f.get("license_no") or not valid_email(email) or len(pw) < 8:
            flash("Name, licence number, valid email and 8+ character password are required.", "danger")
        elif q("SELECT 1 FROM users WHERE email=?", (email,)):
            flash("This email is already registered.", "danger")
        else:
            run("INSERT INTO users(role,full_name,email,phone,password_hash,status,unit_name,license_no,speciality,city,region)"
                " VALUES('doctor',?,?,?,?,'pending',?,?,?,?,?)",
                (f["full_name"], email, f.get("phone"), generate_password_hash(pw), f.get("unit_name"),
                 f["license_no"], f.get("speciality"), f.get("city"), f.get("region")))
            flash("Registration submitted. You can sign in after the admin approves your profile.", "info")
            return redirect(url_for("login"))
    return render_template("signup_doctor.html", regions=REGIONS)


@app.route("/signup/admin", methods=["GET", "POST"])
def signup_admin():
    if request.method == "POST":
        f = request.form
        email, pw = f.get("email", "").strip().lower(), f.get("password", "")
        if f.get("invite") != ADMIN_INVITE_CODE:
            flash("Wrong invite code.", "danger")
        elif not f.get("full_name") or not valid_email(email) or len(pw) < 8:
            flash("Fill all fields (password 8+ characters).", "danger")
        elif q("SELECT 1 FROM users WHERE email=?", (email,)):
            flash("This email is already registered.", "danger")
        else:
            run("INSERT INTO users(role,full_name,email,phone,password_hash) VALUES('admin',?,?,?,?)",
                (f["full_name"], email, f.get("phone"), generate_password_hash(pw)))
            flash("Admin account created. Please sign in.", "success")
            return redirect(url_for("login"))
    return render_template("signup_admin.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email, pw = request.form.get("email", "").strip().lower(), request.form.get("password", "")
        u = q("SELECT * FROM users WHERE email=?", (email,), one=True)
        if not u or not check_password_hash(u["password_hash"], pw):
            flash("Invalid email or password.", "danger")
        elif u["role"] == "doctor" and u["status"] == "pending":
            flash("Your profile is still waiting for admin approval.", "warning")
        elif u["role"] == "doctor" and u["status"] == "rejected":
            flash("Your registration was rejected. Please contact the admin.", "danger")
        else:
            session.clear()
            session["uid"] = u["id"]
            return redirect(url_for("dashboard"))
    return render_template("login.html")


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("home"))


@app.route("/dashboard")
@login_required()
def dashboard():
    return redirect(url_for({"patient": "patient_home", "doctor": "doctor_home", "admin": "admin_home"}[current_user()["role"]]))


# ---------- search by unique ID (all roles) ----------
@app.route("/search")
@login_required()
def search():
    uid = request.args.get("uid", "").strip().upper()
    if not uid:
        return render_template("search.html", result=None, searched=False)
    p = q("SELECT * FROM patients WHERE uid=?", (uid,), one=True)
    u = current_user()
    if p and u["role"] == "patient" and p["user_id"] != u["id"]:
        p = None                       # a patient can only find their own record
    return render_template("search.html", result=p, searched=True, uid=uid)


# ---------- END-USER (patient) ----------
@app.route("/patient")
@login_required("patient")
def patient_home():
    p = q("SELECT * FROM patients WHERE user_id=?", (current_user()["id"],), one=True)
    recs = q("SELECT r.*, d.full_name doctor_name, d.unit_name FROM records r JOIN users d ON d.id=r.doctor_id "
             "WHERE patient_id=? ORDER BY visit_date DESC", (p["id"],))
    doctors = q("SELECT DISTINCT d.id, d.full_name, d.unit_name FROM records r JOIN users d ON d.id=r.doctor_id "
                "WHERE r.patient_id=?", (p["id"],))
    sent = q("SELECT f.*, u.full_name receiver FROM feedback f JOIN users u ON u.id=f.receiver_id "
             "WHERE sender_id=? ORDER BY f.id DESC", (current_user()["id"],))
    return render_template("patient_home.html", p=p, recs=recs, doctors=doctors, sent=sent,
                           latest=recs[0] if recs else None)


@app.route("/feedback", methods=["POST"])
@login_required("patient")
def send_feedback():
    doc = q("SELECT id FROM users WHERE id=? AND role='doctor'", (request.form.get("doctor_id"),), one=True)
    msg = request.form.get("message", "").strip()
    if doc and msg:
        run("INSERT INTO feedback(sender_id,receiver_id,message) VALUES(?,?,?)", (current_user()["id"], doc["id"], msg))
        flash("Feedback sent to your doctor.", "success")
    else:
        flash("Choose a doctor and write a message.", "danger")
    return redirect(url_for("patient_home"))


# ---------- DOCTOR / HEALTHCARE UNIT ----------
@app.route("/doctor")
@login_required("doctor")
def doctor_home():
    me = current_user()
    mine = q("SELECT p.*, MAX(r.visit_date) last_visit, MIN(NULLIF(r.next_appointment,'')) nxt "
             "FROM patients p JOIN records r ON r.patient_id=p.id WHERE r.doctor_id=? GROUP BY p.id ORDER BY last_visit DESC",
             (me["id"],))
    inbox = q("SELECT f.*, u.full_name sender, u.role sender_role FROM feedback f JOIN users u ON u.id=f.sender_id "
              "WHERE receiver_id=? ORDER BY f.id DESC LIMIT 20", (me["id"],))
    return render_template("doctor_home.html", patients=mine, inbox=inbox)


@app.route("/doctor/patient/new", methods=["GET", "POST"])
@login_required("doctor")
def new_patient():
    if request.method == "POST":
        f = request.form
        if not f.get("full_name"):
            flash("Patient name is required.", "danger")
        else:
            _, uid = create_patient_row(f["full_name"], None, num(f.get("age"), int), f.get("gender"),
                                        f.get("region"), f.get("city"), f.get("contact"), f.get("diabetes_type"))
            flash(f"Patient created. Unique ID: {uid}", "success")
            return redirect(url_for("patient_detail", uid=uid))
    return render_template("new_patient.html", regions=REGIONS, types=DIABETES_TYPES)


def get_patient_or_404(uid):
    p = q("SELECT * FROM patients WHERE uid=?", (uid,), one=True)
    if not p:
        abort(404)
    return p


@app.route("/patient/<uid>")
@login_required("doctor", "admin")
def patient_detail(uid):
    p = get_patient_or_404(uid)
    recs = q("SELECT r.*, d.full_name doctor_name, d.unit_name FROM records r JOIN users d ON d.id=r.doctor_id "
             "WHERE patient_id=? ORDER BY visit_date DESC", (p["id"],))
    log_access(uid, "view")
    chart = [{"date": r["visit_date"], "hba1c": r["hba1c"], "glucose": r["fasting_glucose"]} for r in reversed(recs)]
    return render_template("patient_detail.html", p=p, recs=recs, chart=chart)


def _record_form_values(f):
    return (f.get("visit_date") or date.today().isoformat(), f.get("medical_history"),
            num(f.get("fasting_glucose")), num(f.get("hba1c")), num(f.get("bmi")), num(f.get("systolic_bp"), int),
            f.get("treatment_plan"), f.get("care_precautions"), f.get("progress_remarks"), f.get("next_appointment"))


@app.route("/patient/<uid>/record/new", methods=["GET", "POST"])
@login_required("doctor")
def new_record(uid):
    p = get_patient_or_404(uid)
    if request.method == "POST":
        run("INSERT INTO records(patient_id,doctor_id,visit_date,medical_history,fasting_glucose,hba1c,bmi,systolic_bp,"
            "treatment_plan,care_precautions,progress_remarks,next_appointment) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (p["id"], current_user()["id"]) + _record_form_values(request.form))
        log_access(uid, "add_record")
        flash("Record saved.", "success")
        return redirect(url_for("patient_detail", uid=uid))
    return render_template("record_form.html", p=p, r=None, today=date.today().isoformat())


@app.route("/record/<int:rid>/edit", methods=["GET", "POST"])
@login_required("doctor")
def edit_record(rid):
    r = q("SELECT * FROM records WHERE id=?", (rid,), one=True)
    if not r:
        abort(404)
    if r["doctor_id"] != current_user()["id"]:
        abort(403)                         # doctors can only edit their own entries
    p = q("SELECT * FROM patients WHERE id=?", (r["patient_id"],), one=True)
    if request.method == "POST":
        run("UPDATE records SET visit_date=?,medical_history=?,fasting_glucose=?,hba1c=?,bmi=?,systolic_bp=?,"
            "treatment_plan=?,care_precautions=?,progress_remarks=?,next_appointment=? WHERE id=?",
            _record_form_values(request.form) + (rid,))
        log_access(p["uid"], "edit_record")
        flash("Record updated.", "success")
        return redirect(url_for("patient_detail", uid=p["uid"]))
    return render_template("record_form.html", p=p, r=r, today=date.today().isoformat())


@app.route("/patient/<uid>/predict", methods=["GET", "POST"])
@login_required("doctor")
def predict(uid):
    """AI/ML feature: diabetes risk score. Pre-fills from the patient's latest record."""
    p = get_patient_or_404(uid)
    last = q("SELECT * FROM records WHERE patient_id=? ORDER BY visit_date DESC LIMIT 1", (p["id"],), one=True)
    vals = {"Pregnancies": 0, "Glucose": last["fasting_glucose"] if last else None,
            "BloodPressure": None, "SkinThickness": None, "Insulin": None,
            "BMI": last["bmi"] if last else None, "DiabetesPedigreeFunction": None, "Age": p["age"]}
    # >>> UPDATE: the model uses Pima-style features. Map more portal fields here
    #     (e.g. diastolic BP, insulin) once you collect them in the records table.
    result = None
    if request.method == "POST":
        vals = {f: num(request.form.get(f)) for f in FEATURES}
        prob, level = predict_risk(vals)
        result = {"prob": round(prob * 100, 1), "level": level}
        log_access(uid, "ml_predict")
    b = get_model()
    return render_template("predict.html", p=p, vals=vals, features=FEATURES, result=result, model=b)


# ---------- ADMIN ----------
@app.route("/admin")
@login_required("admin")
def admin_home():
    pending = q("SELECT * FROM users WHERE role='doctor' AND status='pending' ORDER BY id")
    doctors = q("SELECT * FROM users WHERE role='doctor' AND status!='pending' ORDER BY id DESC")
    counts = {
        "patients": q("SELECT COUNT(*) c FROM patients", one=True)["c"],
        "doctors": q("SELECT COUNT(*) c FROM users WHERE role='doctor' AND status='approved'", one=True)["c"],
        "pending": len(pending),
        "records": q("SELECT COUNT(*) c FROM records", one=True)["c"],
    }
    return render_template("admin_home.html", pending=pending, doctors=doctors, counts=counts)


@app.route("/admin/doctor/<int:did>/<action>", methods=["POST"])
@login_required("admin")
def review_doctor(did, action):
    if action not in ("approve", "reject"):
        abort(400)
    run("UPDATE users SET status=? WHERE id=? AND role='doctor'", ("approved" if action == "approve" else "rejected", did))
    flash(f"Doctor profile {action}d.", "success")
    return redirect(url_for("admin_home"))


@app.route("/admin/feedback", methods=["POST"])
@login_required("admin")
def admin_feedback():
    doc = q("SELECT id FROM users WHERE id=? AND role='doctor'", (request.form.get("doctor_id"),), one=True)
    msg = request.form.get("message", "").strip()
    if doc and msg:
        run("INSERT INTO feedback(sender_id,receiver_id,message) VALUES(?,?,?)", (current_user()["id"], doc["id"], msg))
        flash("Feedback sent.", "success")
    return redirect(url_for("admin_home"))


@app.route("/admin/content", methods=["GET", "POST"])
@login_required("admin")
def admin_content():
    if request.method == "POST":
        act = request.form.get("action")
        if act == "add" and request.form.get("title") and request.form.get("body"):
            run("INSERT INTO content(title,body) VALUES(?,?)", (request.form["title"], request.form["body"]))
        elif act == "delete":
            run("DELETE FROM content WHERE id=?", (request.form.get("id"),))
        elif act == "edit":
            run("UPDATE content SET title=?, body=? WHERE id=?",
                (request.form.get("title"), request.form.get("body"), request.form.get("id")))
        return redirect(url_for("admin_content"))
    return render_template("admin_content.html", items=q("SELECT * FROM content ORDER BY id DESC"))


@app.route("/admin/reports")
@login_required("admin")
def admin_reports():
    return render_template("admin_reports.html", regions=REGIONS)


@app.route("/admin/api/report")
@login_required("admin")
def report_data():
    """JSON for the charts. Optional ?region=Punjab filter (country vs region level)."""
    region = request.args.get("region") or None
    where, args = ("WHERE p.region=?", (region,)) if region else ("", ())
    by_region = q("SELECT COALESCE(region,'Unknown') k, COUNT(*) v FROM patients GROUP BY region ORDER BY v DESC")
    by_type = q(f"SELECT COALESCE(diabetes_type,'Unknown') k, COUNT(*) v FROM patients p {where} GROUP BY diabetes_type", args)
    avg_hba1c = q("SELECT COALESCE(p.region,'Unknown') k, ROUND(AVG(r.hba1c),2) v FROM records r "
                  "JOIN patients p ON p.id=r.patient_id GROUP BY p.region")
    monthly = q(f"SELECT substr(r.visit_date,1,7) k, COUNT(*) v FROM records r JOIN patients p ON p.id=r.patient_id {where} "
                "GROUP BY k ORDER BY k", args)
    # glycaemic control bands based on each patient's latest HbA1c
    latest = q(f"""SELECT r.hba1c h FROM records r JOIN patients p ON p.id=r.patient_id {where + (' AND' if where else 'WHERE')}
                   r.visit_date = (SELECT MAX(visit_date) FROM records WHERE patient_id=r.patient_id) AND r.hba1c IS NOT NULL""", args)
    bands = {"Controlled (<7%)": 0, "Moderate (7-9%)": 0, "Poor (>9%)": 0}
    for row in latest:
        h = row["h"]
        bands["Controlled (<7%)" if h < 7 else "Moderate (7-9%)" if h <= 9 else "Poor (>9%)"] += 1
    f = lambda rows: {"labels": [r["k"] for r in rows], "values": [r["v"] for r in rows]}
    return jsonify(by_region=f(by_region), by_type=f(by_type), avg_hba1c=f(avg_hba1c), monthly=f(monthly),
                   control={"labels": list(bands), "values": list(bands.values())})


@app.route("/admin/audit")
@login_required("admin")
def admin_audit():
    rows = q("SELECT a.*, u.full_name, u.role FROM access_log a LEFT JOIN users u ON u.id=a.user_id ORDER BY a.id DESC LIMIT 100")
    return render_template("admin_audit.html", rows=rows)


@app.errorhandler(403)
def e403(_e):
    return render_template("error.html", code=403, msg="You do not have permission to view this page."), 403


@app.errorhandler(404)
def e404(_e):
    return render_template("error.html", code=404, msg="Page or record not found."), 404


@app.errorhandler(400)
def e400(e):
    return render_template("error.html", code=400, msg=str(getattr(e, "description", "Bad request"))), 400


# =============================================================================
# 6. TEMPLATES  (Bootstrap 5, kept inline so the whole project is ONE file)
# =============================================================================
BASE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{% block title %}Diabetes Records Portal{% endblock %}</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
 body{background:#f4f7fb} .navbar{background:#0b5d7a} .hero{background:linear-gradient(135deg,#0b5d7a,#1a9bb5);color:#fff;border-radius:1rem}
 .stat{font-size:2.2rem;font-weight:700} .card{border:0;box-shadow:0 2px 8px rgba(0,0,0,.06)}
</style></head><body>
<nav class="navbar navbar-expand-lg navbar-dark mb-4"><div class="container">
 <a class="navbar-brand fw-bold" href="{{ url_for('home') }}">&#129656; DiabetesCare PK</a>
 <div class="ms-auto d-flex align-items-center gap-2">
 {% if current_user() %}
   <a class="btn btn-sm btn-light" href="{{ url_for('dashboard') }}">Dashboard</a>
   <a class="btn btn-sm btn-outline-light" href="{{ url_for('search') }}">Search ID</a>
   {% if current_user().role=='admin' %}<a class="btn btn-sm btn-outline-light" href="{{ url_for('admin_reports') }}">Reports</a>{% endif %}
   <span class="text-white-50 small d-none d-md-inline">{{ current_user().full_name }} ({{ current_user().role }})</span>
   <form method="post" action="{{ url_for('logout') }}" class="m-0"><input type="hidden" name="_csrf" value="{{ csrf_token() }}">
     <button class="btn btn-sm btn-warning">Sign out</button></form>
 {% else %}
   <a class="btn btn-sm btn-outline-light" href="{{ url_for('login') }}">Sign in</a>
   <a class="btn btn-sm btn-light" href="{{ url_for('signup') }}">Sign up</a>
 {% endif %}</div></div></nav>
<div class="container pb-5">
{% for cat,msg in get_flashed_messages(with_categories=true) %}<div class="alert alert-{{cat}} alert-dismissible fade show">{{msg}}<button class="btn-close" data-bs-dismiss="alert"></button></div>{% endfor %}
{% block body %}{% endblock %}</div>
<footer class="text-center text-muted small pb-4">Diabetes Patient Records Portal &middot; Pakistan &middot; Academic project</footer>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
{% block scripts %}{% endblock %}</body></html>"""

REGION_OPTS = "{% for r in regions %}<option>{{ r }}</option>{% endfor %}"

TEMPLATES = {
"base.html": BASE,

"home.html": """{% extends 'base.html' %}{% block body %}
<div class="hero p-5 mb-4"><h1 class="fw-bold">Diabetes Patient Records Portal</h1>
<p class="lead mb-4">A secure, centralised platform to store, track and analyse diabetes patient data across Pakistan.</p>
{% if not current_user() %}<a class="btn btn-light btn-lg me-2" href="{{ url_for('signup') }}">I am a patient</a>
<a class="btn btn-outline-light btn-lg" href="{{ url_for('signup_doctor') }}">I am a doctor / clinic</a>{% endif %}</div>
<div class="row g-3 mb-4 text-center">
 <div class="col-md-4"><div class="card p-3"><div class="stat text-primary">{{ stats.patients }}</div>Patients registered</div></div>
 <div class="col-md-4"><div class="card p-3"><div class="stat text-success">{{ stats.doctors }}</div>Approved doctors / units</div></div>
 <div class="col-md-4"><div class="card p-3"><div class="stat text-danger">{{ stats.records }}</div>Medical records</div></div></div>
<h4>Announcements</h4>{% for n in news %}<div class="card p-3 mb-2"><b>{{ n.title }}</b><div>{{ n.body }}</div><small class="text-muted">{{ n.created_at[:10] }}</small></div>
{% else %}<p class="text-muted">No announcements.</p>{% endfor %}{% endblock %}""",

"login.html": """{% extends 'base.html' %}{% block body %}
<div class="row justify-content-center"><div class="col-md-5"><div class="card p-4"><h4>Sign in</h4>
<form method="post"><input type="hidden" name="_csrf" value="{{ csrf_token() }}">
<div class="mb-3"><label class="form-label">Email</label><input class="form-control" type="email" name="email" required></div>
<div class="mb-3"><label class="form-label">Password</label><input class="form-control" type="password" name="password" required></div>
<button class="btn btn-primary w-100">Sign in</button></form><hr>
<small>New? <a href="{{ url_for('signup') }}">Patient sign-up</a> &middot; <a href="{{ url_for('signup_doctor') }}">Doctor / unit</a> &middot; <a href="{{ url_for('signup_admin') }}">Admin</a></small></div></div></div>{% endblock %}""",

"signup.html": """{% extends 'base.html' %}{% block body %}
<div class="row justify-content-center"><div class="col-md-7"><div class="card p-4"><h4>Patient sign-up</h4>
<form method="post" class="row g-3"><input type="hidden" name="_csrf" value="{{ csrf_token() }}">
<div class="col-md-6"><label class="form-label">Full name</label><input class="form-control" name="full_name" required></div>
<div class="col-md-6"><label class="form-label">Email</label><input class="form-control" type="email" name="email" required></div>
<div class="col-md-6"><label class="form-label">Password (8+ chars)</label><input class="form-control" type="password" name="password" minlength="8" required></div>
<div class="col-md-6"><label class="form-label">Phone</label><input class="form-control" name="phone"></div>
<div class="col-md-3"><label class="form-label">Age</label><input class="form-control" type="number" name="age" min="0" max="120"></div>
<div class="col-md-3"><label class="form-label">Gender</label><select class="form-select" name="gender"><option>Male</option><option>Female</option><option>Other</option></select></div>
<div class="col-md-6"><label class="form-label">Region</label><select class="form-select" name="region">""" + REGION_OPTS + """</select></div>
<div class="col-md-6"><label class="form-label">City</label><input class="form-control" name="city"></div>
<div class="col-md-6"><label class="form-label">Diabetes type</label><select class="form-select" name="diabetes_type">{% for t in types %}<option>{{ t }}</option>{% endfor %}</select></div>
<div class="col-12"><button class="btn btn-primary">Create account</button></div></form></div></div></div>{% endblock %}""",

"signup_doctor.html": """{% extends 'base.html' %}{% block body %}
<div class="row justify-content-center"><div class="col-md-8"><div class="card p-4"><h4>Doctor / Healthcare unit registration</h4>
<p class="text-muted">Your profile will be reviewed by an admin before you can sign in.</p>
<form method="post" class="row g-3"><input type="hidden" name="_csrf" value="{{ csrf_token() }}">
<div class="col-md-6"><label class="form-label">Doctor / contact name</label><input class="form-control" name="full_name" required></div>
<div class="col-md-6"><label class="form-label">Clinic / hospital name</label><input class="form-control" name="unit_name"></div>
<div class="col-md-6"><label class="form-label">PMDC / licence number</label><input class="form-control" name="license_no" required></div>
<div class="col-md-6"><label class="form-label">Speciality</label><input class="form-control" name="speciality"></div>
<div class="col-md-6"><label class="form-label">Email</label><input class="form-control" type="email" name="email" required></div>
<div class="col-md-6"><label class="form-label">Phone (for appointments)</label><input class="form-control" name="phone"></div>
<div class="col-md-4"><label class="form-label">City</label><input class="form-control" name="city"></div>
<div class="col-md-4"><label class="form-label">Region</label><select class="form-select" name="region">""" + REGION_OPTS + """</select></div>
<div class="col-md-4"><label class="form-label">Password (8+ chars)</label><input class="form-control" type="password" name="password" minlength="8" required></div>
<div class="col-12"><button class="btn btn-primary">Submit for approval</button></div></form></div></div></div>{% endblock %}""",

"signup_admin.html": """{% extends 'base.html' %}{% block body %}
<div class="row justify-content-center"><div class="col-md-5"><div class="card p-4"><h4>Admin registration</h4>
<form method="post"><input type="hidden" name="_csrf" value="{{ csrf_token() }}">
{% for n,l,t in [('full_name','Full name','text'),('email','Email','email'),('phone','Phone','text'),('password','Password','password'),('invite','Admin invite code','password')] %}
<div class="mb-3"><label class="form-label">{{ l }}</label><input class="form-control" type="{{ t }}" name="{{ n }}" required></div>{% endfor %}
<button class="btn btn-primary w-100">Create admin</button></form></div></div></div>{% endblock %}""",

"search.html": """{% extends 'base.html' %}{% block body %}
<div class="card p-4"><h4>Search a record by unique Patient ID</h4>
<form class="d-flex gap-2" method="get"><input class="form-control" name="uid" placeholder="e.g. PK-DM-00001" value="{{ uid or '' }}" required>
<button class="btn btn-primary">Search</button></form>
{% if searched %}<hr>{% if result %}
 <div><b>{{ result.full_name }}</b> &middot; {{ result.uid }} &middot; {{ result.region }} &middot; {{ result.diabetes_type }}</div>
 {% if current_user().role=='patient' %}<a class="btn btn-sm btn-primary mt-2" href="{{ url_for('patient_home') }}">Open my record</a>
 {% else %}<a class="btn btn-sm btn-primary mt-2" href="{{ url_for('patient_detail', uid=result.uid) }}">Open record</a>{% endif %}
{% else %}<div class="text-danger">No record found for that ID.</div>{% endif %}{% endif %}</div>{% endblock %}""",

"patient_home.html": """{% extends 'base.html' %}{% block body %}
<div class="row g-3 mb-3"><div class="col-md-5"><div class="card p-3"><h5>My profile</h5>
<div>Unique ID: <span class="badge bg-primary fs-6">{{ p.uid }}</span></div>
<div>Name: {{ p.full_name }} &middot; Age: {{ p.age or '-' }} &middot; {{ p.gender or '' }}</div>
<div>Region: {{ p.region }} ({{ p.city or '-' }})</div><div>Diabetes type: {{ p.diabetes_type }}</div></div></div>
<div class="col-md-7"><div class="card p-3 h-100"><h5>Treatment plan &amp; next appointment</h5>
{% if latest %}<div><b>Next appointment:</b> <span class="text-success fw-bold">{{ latest.next_appointment or 'Not scheduled' }}</span></div>
<div><b>Treatment plan:</b> {{ latest.treatment_plan }}</div><div><b>Care precautions:</b> {{ latest.care_precautions }}</div>
{% else %}<span class="text-muted">No records yet. Share your ID ({{ p.uid }}) with your doctor.</span>{% endif %}</div></div></div>
<div class="card p-3 mb-3"><h5>Medical history</h5>
<table class="table table-sm"><thead><tr><th>Date</th><th>Doctor</th><th>HbA1c</th><th>Fasting glucose</th><th>BMI</th><th>BP</th><th>History / remarks</th></tr></thead><tbody>
{% for r in recs %}<tr><td>{{ r.visit_date }}</td><td>{{ r.doctor_name }}<br><small class="text-muted">{{ r.unit_name }}</small></td><td>{{ r.hba1c }}</td><td>{{ r.fasting_glucose }}</td><td>{{ r.bmi }}</td><td>{{ r.systolic_bp }}</td><td>{{ r.medical_history }}<br><i>{{ r.progress_remarks }}</i></td></tr>
{% else %}<tr><td colspan="7" class="text-muted">Nothing recorded.</td></tr>{% endfor %}</tbody></table></div>
<div class="card p-3"><h5>Leave feedback for your doctor</h5>
{% if doctors %}<form method="post" action="{{ url_for('send_feedback') }}" class="row g-2"><input type="hidden" name="_csrf" value="{{ csrf_token() }}">
<div class="col-md-4"><select class="form-select" name="doctor_id">{% for d in doctors %}<option value="{{ d.id }}">{{ d.full_name }} - {{ d.unit_name }}</option>{% endfor %}</select></div>
<div class="col-md-6"><input class="form-control" name="message" placeholder="Your feedback" required></div><div class="col-md-2"><button class="btn btn-primary w-100">Send</button></div></form>
{% else %}<span class="text-muted">You can send feedback once a doctor has added a record for you.</span>{% endif %}
{% for f in sent %}<div class="border-top mt-2 pt-2 small">To <b>{{ f.receiver }}</b>: {{ f.message }} <span class="text-muted">({{ f.created_at[:10] }})</span></div>{% endfor %}</div>{% endblock %}""",

"doctor_home.html": """{% extends 'base.html' %}{% block body %}
<div class="d-flex justify-content-between align-items-center mb-3"><h4>My patients</h4>
<a class="btn btn-primary" href="{{ url_for('new_patient') }}">+ Register new patient</a></div>
<div class="card p-3 mb-3"><table class="table table-hover"><thead><tr><th>ID</th><th>Name</th><th>Region</th><th>Type</th><th>Last visit</th><th></th></tr></thead><tbody>
{% for p in patients %}<tr><td>{{ p.uid }}</td><td>{{ p.full_name }}</td><td>{{ p.region }}</td><td>{{ p.diabetes_type }}</td><td>{{ p.last_visit }}</td>
<td><a class="btn btn-sm btn-outline-primary" href="{{ url_for('patient_detail', uid=p.uid) }}">Open</a></td></tr>
{% else %}<tr><td colspan="6" class="text-muted">No patients yet. Register one or search by ID.</td></tr>{% endfor %}</tbody></table></div>
<div class="card p-3"><h5>Feedback inbox</h5>{% for f in inbox %}<div class="border-bottom py-2"><b>{{ f.sender }}</b> <span class="badge bg-secondary">{{ f.sender_role }}</span>: {{ f.message }} <small class="text-muted">{{ f.created_at[:10] }}</small></div>
{% else %}<span class="text-muted">No feedback yet.</span>{% endfor %}</div>{% endblock %}""",

"new_patient.html": """{% extends 'base.html' %}{% block body %}
<div class="card p-4 col-md-8"><h4>Register a new patient</h4><form method="post" class="row g-3"><input type="hidden" name="_csrf" value="{{ csrf_token() }}">
<div class="col-md-6"><label class="form-label">Full name</label><input class="form-control" name="full_name" required></div>
<div class="col-md-3"><label class="form-label">Age</label><input class="form-control" type="number" name="age"></div>
<div class="col-md-3"><label class="form-label">Gender</label><select class="form-select" name="gender"><option>Male</option><option>Female</option><option>Other</option></select></div>
<div class="col-md-4"><label class="form-label">Region</label><select class="form-select" name="region">""" + REGION_OPTS + """</select></div>
<div class="col-md-4"><label class="form-label">City</label><input class="form-control" name="city"></div>
<div class="col-md-4"><label class="form-label">Contact</label><input class="form-control" name="contact"></div>
<div class="col-md-6"><label class="form-label">Diabetes type</label><select class="form-select" name="diabetes_type">{% for t in types %}<option>{{ t }}</option>{% endfor %}</select></div>
<div class="col-12"><button class="btn btn-primary">Create &amp; get unique ID</button></div></form></div>{% endblock %}""",

"patient_detail.html": """{% extends 'base.html' %}{% block body %}
<div class="card p-3 mb-3"><div class="d-flex justify-content-between flex-wrap gap-2"><div>
<h4 class="mb-0">{{ p.full_name }} <span class="badge bg-primary">{{ p.uid }}</span></h4>
<small>{{ p.age }} yrs &middot; {{ p.gender }} &middot; {{ p.city }}, {{ p.region }} &middot; {{ p.diabetes_type }} &middot; {{ p.contact }}</small></div>
{% if current_user().role=='doctor' %}<div><a class="btn btn-success" href="{{ url_for('new_record', uid=p.uid) }}">+ Add record</a>
<a class="btn btn-dark" href="{{ url_for('predict', uid=p.uid) }}">&#129504; AI risk check</a></div>{% endif %}</div></div>
{% if chart|length > 1 %}<div class="card p-3 mb-3"><h6>HbA1c &amp; glucose trend</h6><canvas id="trend" height="90"></canvas></div>{% endif %}
{% for r in recs %}<div class="card p-3 mb-2"><div class="d-flex justify-content-between"><b>{{ r.visit_date }}</b>
<span class="text-muted small">{{ r.doctor_name }} - {{ r.unit_name }}</span></div>
<div class="row small mt-2"><div class="col-6 col-md-3">HbA1c: <b>{{ r.hba1c }}</b></div><div class="col-6 col-md-3">Glucose: <b>{{ r.fasting_glucose }}</b></div><div class="col-6 col-md-3">BMI: <b>{{ r.bmi }}</b></div><div class="col-6 col-md-3">BP: <b>{{ r.systolic_bp }}</b></div></div>
<div class="mt-2"><b>History:</b> {{ r.medical_history }}</div><div><b>Treatment plan:</b> {{ r.treatment_plan }}</div>
<div><b>Care precautions:</b> {{ r.care_precautions }}</div><div><b>Progress:</b> {{ r.progress_remarks }}</div>
<div><b>Next appointment:</b> {{ r.next_appointment or '-' }}</div>
{% if current_user().role=='doctor' and r.doctor_id==current_user().id %}<div class="mt-2"><a class="btn btn-sm btn-outline-primary" href="{{ url_for('edit_record', rid=r.id) }}">Edit</a></div>{% endif %}</div>
{% else %}<div class="text-muted">No records yet.</div>{% endfor %}{% endblock %}
{% block scripts %}{% if chart|length > 1 %}<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script><script>
const d={{ chart|tojson }};new Chart(document.getElementById('trend'),{type:'line',data:{labels:d.map(x=>x.date),datasets:[
{label:'HbA1c (%)',data:d.map(x=>x.hba1c),borderColor:'#c0392b',yAxisID:'y'},{label:'Fasting glucose (mg/dL)',data:d.map(x=>x.glucose),borderColor:'#0b5d7a',yAxisID:'y1'}]},
options:{scales:{y:{position:'left'},y1:{position:'right',grid:{drawOnChartArea:false}}}}});</script>{% endif %}{% endblock %}""",

"record_form.html": """{% extends 'base.html' %}{% block body %}
<div class="card p-4"><h4>{{ 'Edit' if r else 'New' }} record for {{ p.full_name }} ({{ p.uid }})</h4>
<form method="post" class="row g-3"><input type="hidden" name="_csrf" value="{{ csrf_token() }}">
<div class="col-md-3"><label class="form-label">Visit date</label><input class="form-control" type="date" name="visit_date" value="{{ r.visit_date if r else today }}"></div>
<div class="col-md-3"><label class="form-label">HbA1c (%)</label><input class="form-control" type="number" step="0.1" name="hba1c" value="{{ r.hba1c if r and r.hba1c is not none else '' }}"></div>
<div class="col-md-3"><label class="form-label">Fasting glucose (mg/dL)</label><input class="form-control" type="number" step="1" name="fasting_glucose" value="{{ r.fasting_glucose if r and r.fasting_glucose is not none else '' }}"></div>
<div class="col-md-3"><label class="form-label">BMI</label><input class="form-control" type="number" step="0.1" name="bmi" value="{{ r.bmi if r and r.bmi is not none else '' }}"></div>
<div class="col-md-3"><label class="form-label">Systolic BP</label><input class="form-control" type="number" name="systolic_bp" value="{{ r.systolic_bp if r and r.systolic_bp is not none else '' }}"></div>
<div class="col-md-3"><label class="form-label">Next appointment</label><input class="form-control" type="date" name="next_appointment" value="{{ r.next_appointment if r else '' }}"></div>
<div class="col-12"><label class="form-label">Medical history</label><textarea class="form-control" name="medical_history" rows="2">{{ r.medical_history if r else '' }}</textarea></div>
<div class="col-12"><label class="form-label">Treatment plan</label><textarea class="form-control" name="treatment_plan" rows="2">{{ r.treatment_plan if r else '' }}</textarea></div>
<div class="col-12"><label class="form-label">Care precautions</label><textarea class="form-control" name="care_precautions" rows="2">{{ r.care_precautions if r else '' }}</textarea></div>
<div class="col-12"><label class="form-label">Progress remarks</label><textarea class="form-control" name="progress_remarks" rows="2">{{ r.progress_remarks if r else '' }}</textarea></div>
<div class="col-12"><button class="btn btn-success">Save</button> <a class="btn btn-light" href="{{ url_for('patient_detail', uid=p.uid) }}">Cancel</a></div></form></div>{% endblock %}""",

"predict.html": """{% extends 'base.html' %}{% block body %}
<div class="row g-3"><div class="col-md-7"><div class="card p-4"><h4>AI diabetes-risk check &mdash; {{ p.full_name }}</h4>
<p class="text-muted small">Pre-filled from the latest record. Blank fields use the training-set median.</p>
<form method="post" class="row g-3"><input type="hidden" name="_csrf" value="{{ csrf_token() }}">
{% for f in features %}<div class="col-md-6"><label class="form-label">{{ f }}</label><input class="form-control" type="number" step="any" name="{{ f }}" value="{{ vals[f] if vals[f] is not none else '' }}"></div>{% endfor %}
<div class="col-12"><button class="btn btn-dark">Predict</button></div></form>
{% if result %}<div class="alert mt-3 alert-{{ 'danger' if result.level=='High' else 'warning' if result.level=='Moderate' else 'success' }}">
<h5 class="mb-1">{{ result.level }} risk &mdash; {{ result.prob }}%</h5><small>Decision-support only. Not a diagnosis.</small></div>{% endif %}</div></div>
<div class="col-md-5"><div class="card p-4"><h6>Model information</h6><ul class="small mb-2">
<li>Algorithm: Random Forest (300 trees)</li><li>Data: {{ model.source }} ({{ model.n_rows }} rows)</li>
<li>Test accuracy: <b>{{ model.accuracy }}</b> &middot; ROC-AUC: <b>{{ model.auc }}</b></li></ul>
<h6>Feature importance</h6>{% for k,v in model.importance|dictsort(by='value', reverse=true) %}
<div class="small">{{ k }}</div><div class="progress mb-1" style="height:8px"><div class="progress-bar" style="width:{{ (v*100)|round(1) }}%"></div></div>{% endfor %}</div></div></div>{% endblock %}""",

"admin_home.html": """{% extends 'base.html' %}{% block body %}
<div class="row g-3 mb-3 text-center">{% for k,v in counts.items() %}<div class="col-6 col-md-3"><div class="card p-3"><div class="stat">{{ v }}</div><span class="text-capitalize">{{ k }}</span></div></div>{% endfor %}</div>
<div class="card p-3 mb-3"><h5>Pending doctor / unit approvals</h5>{% for d in pending %}<div class="border-bottom py-2 d-flex justify-content-between flex-wrap gap-2">
<div><b>{{ d.full_name }}</b> &middot; {{ d.unit_name }} &middot; Licence: <code>{{ d.license_no }}</code> &middot; {{ d.speciality }} &middot; {{ d.city }}, {{ d.region }}<br><small>{{ d.email }} &middot; {{ d.phone }}</small></div>
<div class="d-flex gap-1"><form method="post" action="{{ url_for('review_doctor', did=d.id, action='approve') }}"><input type="hidden" name="_csrf" value="{{ csrf_token() }}"><button class="btn btn-sm btn-success">Approve</button></form>
<form method="post" action="{{ url_for('review_doctor', did=d.id, action='reject') }}"><input type="hidden" name="_csrf" value="{{ csrf_token() }}"><button class="btn btn-sm btn-danger">Reject</button></form></div></div>
{% else %}<span class="text-muted">No pending requests.</span>{% endfor %}</div>
<div class="card p-3 mb-3"><h5>Send feedback to a doctor / unit</h5><form method="post" action="{{ url_for('admin_feedback') }}" class="row g-2"><input type="hidden" name="_csrf" value="{{ csrf_token() }}">
<div class="col-md-4"><select class="form-select" name="doctor_id">{% for d in doctors if d.status=='approved' %}<option value="{{ d.id }}">{{ d.full_name }} - {{ d.unit_name }}</option>{% endfor %}</select></div>
<div class="col-md-6"><input class="form-control" name="message" placeholder="Message" required></div><div class="col-md-2"><button class="btn btn-primary w-100">Send</button></div></form></div>
<div class="d-flex gap-2"><a class="btn btn-outline-primary" href="{{ url_for('admin_reports') }}">Reports &amp; trends</a>
<a class="btn btn-outline-primary" href="{{ url_for('admin_content') }}">Manage content</a><a class="btn btn-outline-secondary" href="{{ url_for('admin_audit') }}">Access log</a></div>{% endblock %}""",

"admin_content.html": """{% extends 'base.html' %}{% block body %}
<div class="card p-3 mb-3"><h5>Add announcement</h5><form method="post" class="row g-2"><input type="hidden" name="_csrf" value="{{ csrf_token() }}"><input type="hidden" name="action" value="add">
<div class="col-md-4"><input class="form-control" name="title" placeholder="Title" required></div><div class="col-md-6"><input class="form-control" name="body" placeholder="Text" required></div><div class="col-md-2"><button class="btn btn-primary w-100">Publish</button></div></form></div>
{% for i in items %}<div class="card p-3 mb-2"><form method="post" class="row g-2"><input type="hidden" name="_csrf" value="{{ csrf_token() }}"><input type="hidden" name="id" value="{{ i.id }}">
<div class="col-md-3"><input class="form-control" name="title" value="{{ i.title }}"></div><div class="col-md-6"><input class="form-control" name="body" value="{{ i.body }}"></div>
<div class="col-md-3"><button class="btn btn-sm btn-success" name="action" value="edit">Update</button> <button class="btn btn-sm btn-danger" name="action" value="delete">Delete</button></div></form></div>{% endfor %}{% endblock %}""",

"admin_reports.html": """{% extends 'base.html' %}{% block body %}
<div class="d-flex justify-content-between align-items-center mb-3"><h4>Diabetes trends &amp; patterns</h4>
<select id="region" class="form-select w-auto"><option value="">Whole country</option>""" + REGION_OPTS + """</select></div>
<div class="row g-3">
<div class="col-md-6"><div class="card p-3"><h6>Patients by region</h6><canvas id="c1"></canvas></div></div>
<div class="col-md-6"><div class="card p-3"><h6>Diabetes type distribution</h6><canvas id="c2"></canvas></div></div>
<div class="col-md-6"><div class="card p-3"><h6>Average HbA1c by region</h6><canvas id="c3"></canvas></div></div>
<div class="col-md-6"><div class="card p-3"><h6>Glycaemic control (latest HbA1c)</h6><canvas id="c4"></canvas></div></div>
<div class="col-12"><div class="card p-3"><h6>Clinic visits per month</h6><canvas id="c5" height="70"></canvas></div></div></div>{% endblock %}
{% block scripts %}<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script><script>
const charts={};const pal=['#0b5d7a','#1a9bb5','#e67e22','#c0392b','#27ae60','#8e44ad','#7f8c8d'];
function draw(id,type,d,label){if(charts[id])charts[id].destroy();
 charts[id]=new Chart(document.getElementById(id),{type:type,data:{labels:d.labels,datasets:[{label:label,data:d.values,backgroundColor:type==='line'?'#0b5d7a':pal,borderColor:'#0b5d7a'}]},options:{plugins:{legend:{display:type==='pie'||type==='doughnut'}}}});}
async function load(){const r=document.getElementById('region').value;const d=await (await fetch('{{ url_for("report_data") }}?region='+encodeURIComponent(r))).json();
 draw('c1','bar',d.by_region,'Patients');draw('c2','doughnut',d.by_type,'Patients');draw('c3','bar',d.avg_hba1c,'Avg HbA1c');draw('c4','pie',d.control,'Patients');draw('c5','line',d.monthly,'Visits');}
document.getElementById('region').onchange=load;load();</script>{% endblock %}""",

"admin_audit.html": """{% extends 'base.html' %}{% block body %}<div class="card p-3"><h5>Record access log (last 100)</h5>
<table class="table table-sm"><thead><tr><th>Time</th><th>User</th><th>Role</th><th>Patient ID</th><th>Action</th></tr></thead><tbody>
{% for r in rows %}<tr><td>{{ r.created_at }}</td><td>{{ r.full_name }}</td><td>{{ r.role }}</td><td>{{ r.patient_uid }}</td><td>{{ r.action }}</td></tr>{% endfor %}</tbody></table></div>{% endblock %}""",

"error.html": """{% extends 'base.html' %}{% block body %}<div class="text-center py-5"><h1 class="display-3">{{ code }}</h1><p class="lead">{{ msg }}</p><a class="btn btn-primary" href="{{ url_for('home') }}">Back to home</a></div>{% endblock %}""",
}
app.jinja_loader = DictLoader(TEMPLATES)

# =============================================================================
# ENTRY POINT
# =============================================================================
init_db()
seed_demo_data()   # >>> UPDATE: remove for production

if __name__ == "__main__":
    get_model()    # train/load ML model at start-up
    # >>> UPDATE: debug=False for any real deployment
    app.run(debug=True, port=5000, use_reloader=False)
