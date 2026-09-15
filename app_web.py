from flask import Flask, request, render_template, send_file
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import io
import os

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

# ============================================================
# LOAD & TRAIN MODEL
# ============================================================

try:
    df = pd.read_csv("student_performance_dataset.csv")

    required_columns = ["studytime", "absences", "G1", "G2", "G3"]

    if not all(col in df.columns for col in required_columns):
        raise ValueError("Dataset missing required columns")

    df = df[required_columns].dropna()

    # 🔥 FEATURE ENGINEERING
    df["avg_score"] = (df["G1"] + df["G2"]) / 2
    df["improvement"] = df["G2"] - df["G1"]

    X = df[["studytime", "absences", "G1", "G2", "avg_score", "improvement"]]
    y = df["G3"]

    # 🔥 TRAIN TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 🔥 OPTIMIZED MODEL
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )

    model.fit(X_train, y_train)

    # 🔥 ACCURACY (for logs + viva)
    y_pred = model.predict(X_test)
    accuracy = r2_score(y_test, y_pred)

    print("Model trained successfully")
    print("Accuracy (R2 Score):", round(accuracy * 100, 2), "%")

except Exception as e:
    print("ERROR:", e)
    model = None


# ============================================================
# GLOBAL STORAGE
# ============================================================

students = []


# ============================================================
# PERFORMANCE CLASSIFICATION (IMPROVED)
# ============================================================

def get_performance(score):
    if score >= 17:
        return "Outstanding"
    elif score >= 15:
        return "Excellent"
    elif score >= 13:
        return "Good"
    elif score >= 11:
        return "Average"
    elif score >= 9:
        return "Below Average"
    else:
        return "Poor"


# ============================================================
# PREPARE LEADERBOARD
# ============================================================

def prepare_students():
    global students
    students = sorted(students, key=lambda x: x["score"], reverse=True)
    for i, s in enumerate(students):
        s["rank"] = i + 1


# ============================================================
# DASHBOARD
# ============================================================

def get_dashboard_data():
    if not students:
        return {
            "total": 0,
            "average": 0,
            "highest": 0,
            "lowest": 0,
            "outstanding": 0,
            "excellent": 0,
            "good": 0,
            "average_count": 0,
            "below_average": 0,
            "poor": 0
        }

    scores = [s["score"] for s in students]

    counts = {
        "Outstanding": 0,
        "Excellent": 0,
        "Good": 0,
        "Average": 0,
        "Below Average": 0,
        "Poor": 0
    }

    for s in students:
        counts[s["performance"]] += 1

    return {
        "total": len(students),
        "average": round(sum(scores)/len(scores), 2),
        "highest": round(max(scores), 2),
        "lowest": round(min(scores), 2),
        "outstanding": counts["Outstanding"],
        "excellent": counts["Excellent"],
        "good": counts["Good"],
        "average_count": counts["Average"],
        "below_average": counts["Below Average"],
        "poor": counts["Poor"]
    }


# ============================================================
# HOME
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():
    global students

    prediction = None
    performance = None
    message = None
    message_type = None

    if request.method == "POST":
        try:
            if model is None:
                raise ValueError("Model not available")

            name = request.form.get("name", "").strip()
            if not name:
                name = f"Student {len(students)+1}"

            studytime = float(request.form.get("studytime", 0))
            absences = float(request.form.get("absences", 0))
            g1 = float(request.form.get("g1", 0))
            g2 = float(request.form.get("g2", 0))

            # validation
            if not (0 <= studytime <= 12):
                raise ValueError("Studytime must be 0–12")
            if not (0 <= absences <= 22):
                raise ValueError("Absences must be 0–22")
            if not (0 <= g1 <= 20):
                raise ValueError("G1 must be 0–20")
            if not (0 <= g2 <= 20):
                raise ValueError("G2 must be 0–20")

            # 🔥 NEW FEATURES
            avg_score = (g1 + g2) / 2
            improvement = g2 - g1

            input_df = pd.DataFrame([[studytime, absences, g1, g2, avg_score, improvement]],
                                    columns=["studytime", "absences", "G1", "G2", "avg_score", "improvement"])

            pred = model.predict(input_df)[0]

            prediction = round(max(0, min(20, pred)), 2)
            performance = get_performance(prediction)

            students.append({
                "name": name,
                "score": prediction,
                "performance": performance
            })

            prepare_students()

            message = "Prediction successful"
            message_type = "success"

        except Exception as e:
            message = str(e)
            message_type = "error"

    return render_template(
        "index.html",
        prediction=prediction,
        performance=performance,
        students=students,
        dashboard=get_dashboard_data(),
        message=message,
        message_type=message_type
    )


# ============================================================
# UPLOAD CSV
# ============================================================

@app.route("/upload", methods=["POST"])
def upload():
    global students

    try:
        if model is None:
            raise ValueError("Model not loaded")

        file = request.files.get("file")
        if not file or not file.filename.endswith(".csv"):
            raise ValueError("Upload valid CSV")

        df = pd.read_csv(file)

        required = ["studytime", "absences", "G1", "G2"]
        if not all(col in df.columns for col in required):
            raise ValueError("Missing required columns")

        df = df.dropna(subset=required)
        students = []

        for i, row in df.iterrows():
            studytime = float(row["studytime"])
            absences = float(row["absences"])
            g1 = float(row["G1"])
            g2 = float(row["G2"])

            avg_score = (g1 + g2) / 2
            improvement = g2 - g1

            input_df = pd.DataFrame([[studytime, absences, g1, g2, avg_score, improvement]],
                                    columns=["studytime", "absences", "G1", "G2", "avg_score", "improvement"])

            pred = model.predict(input_df)[0]
            score = round(max(0, min(20, pred)), 2)

            name = row["name"] if "name" in df.columns else f"Student {i+1}"

            students.append({
                "name": name,
                "score": score,
                "performance": get_performance(score)
            })

        prepare_students()

        message = f"Predicted {len(students)} students"
        message_type = "success"

    except Exception as e:
        message = str(e)
        message_type = "error"

    return render_template(
        "index.html",
        prediction=None,
        performance=None,
        students=students,
        dashboard=get_dashboard_data(),
        message=message,
        message_type=message_type
    )


# ============================================================
# DOWNLOAD
# ============================================================

@app.route("/download")
def download():
    if not students:
        return "No data"

    df = pd.DataFrame(students)
    df = df[["rank", "name", "score", "performance"]]

    output = io.StringIO()
    df.to_csv(output, index=False)

    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="results.csv"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=PORT)
