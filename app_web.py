from flask import Flask, request, render_template, send_file
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import io
import os

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

# ============================================================
# LOAD TRAINING DATASET
# ============================================================

try:
    df = pd.read_csv("student_performance_dataset.csv")

    required_cols = ["studytime", "absences", "G1", "G2", "G3"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df = df[required_cols].dropna()

    X = df[["studytime", "absences", "G1", "G2"]]
    y = df["G3"]

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)

    print("Model trained successfully")

except Exception as e:
    print("ERROR LOADING DATASET:", e)
    model = None

# ============================================================
# GLOBAL STORAGE
# ============================================================

students = []

# ============================================================
# PERFORMANCE CATEGORY
# ============================================================

def get_performance(score):
    if score >= 18:
        return "Outstanding"
    elif score >= 16:
        return "Excellent"
    elif score >= 14:
        return "Good"
    elif score >= 12:
        return "Average"
    elif score >= 10:
        return "Below Average"
    else:
        return "Poor"

# ============================================================
# SORT + RANK
# ============================================================

def prepare_students():
    global students
    students = sorted(students, key=lambda x: x["score"], reverse=True)

    for i, s in enumerate(students):
        s["rank"] = i + 1

# ============================================================
# DASHBOARD STATS
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
# HOME ROUTE
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
                raise ValueError("Model not loaded")

            name = request.form.get("name", "").strip()
            if not name:
                name = f"Student {len(students)+1}"

            studytime = float(request.form.get("studytime", 0))
            absences = float(request.form.get("absences", 0))
            g1 = float(request.form.get("g1", 0))
            g2 = float(request.form.get("g2", 0))

            input_df = pd.DataFrame(
                [[studytime, absences, g1, g2]],
                columns=["studytime", "absences", "G1", "G2"]
            )

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

    dashboard = get_dashboard_data()

    return render_template(
        "index.html",
        prediction=prediction,
        performance=performance,
        students=students,
        dashboard=dashboard,
        message=message,
        message_type=message_type
    )

# ============================================================
# DATASET UPLOAD
# ============================================================

@app.route("/upload", methods=["POST"])
def upload():
    global students

    try:
        file = request.files.get("file")

        if not file:
            raise ValueError("No file uploaded")

        df = pd.read_csv(file)

        required_cols = ["studytime", "absences", "G1", "G2"]

        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        students = []

        for i, row in df.iterrows():
            input_df = pd.DataFrame(
                [[row["studytime"], row["absences"], row["G1"], row["G2"]]],
                columns=["studytime", "absences", "G1", "G2"]
            )

            pred = model.predict(input_df)[0]
            score = round(max(0, min(20, pred)), 2)

            students.append({
                "name": row.get("name", f"Student {i+1}"),
                "score": score,
                "performance": get_performance(score)
            })

        prepare_students()

        message = f"{len(students)} students processed"
        message_type = "success"

    except Exception as e:
        message = str(e)
        message_type = "error"

    return render_template(
        "index.html",
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
