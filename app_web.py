from flask import Flask, request, render_template, send_file
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import io

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

# ============================================================
# LOAD TRAINING DATASET
# ============================================================

try:
    df = pd.read_csv("student_performance_dataset.csv")

    required_training_columns = [
        "studytime",
        "absences",
        "G1",
        "G2",
        "G3"
    ]

    missing_columns = [
        col for col in required_training_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "maths.csv is missing columns: "
            + ", ".join(missing_columns)
        )

    # Keep only required columns
    df = df[required_training_columns].copy()

    # Remove rows with missing values
    df = df.dropna()

    # G3 in the original dataset is out of 100.
    # Convert it to a score out of 20.
    

    # Features
    X = df[["studytime", "absences", "G1", "G2"]]

    # Target
    y = df["G3"]

    # ========================================================
    # TRAIN RANDOM FOREST MODEL
    # ========================================================

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(X, y)

    print("Training dataset loaded successfully.")
    print("Number of training records:", len(df))
    print("Model trained successfully.")

except Exception as e:
    print("ERROR LOADING TRAINING DATASET:", e)
    model = None


# ============================================================
# GLOBAL STORAGE
# ============================================================

students = []


# ============================================================
# PERFORMANCE CLASSIFICATION
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
# PREPARE LEADERBOARD
# ============================================================

def prepare_students():

    global students

    # Sort by score
    students = sorted(
        students,
        key=lambda x: x["score"],
        reverse=True
    )

    # Assign ranks
    for index, student in enumerate(students):
        student["rank"] = index + 1


# ============================================================
# DASHBOARD DATA
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

    scores = [student["score"] for student in students]

    performance_counts = {
        "Outstanding": 0,
        "Excellent": 0,
        "Good": 0,
        "Average": 0,
        "Below Average": 0,
        "Poor": 0
    }

    for student in students:
        performance_counts[student["performance"]] += 1

    return {
        "total": len(students),
        "average": round(sum(scores) / len(scores), 2),
        "highest": round(max(scores), 2),
        "lowest": round(min(scores), 2),

        "outstanding": performance_counts["Outstanding"],
        "excellent": performance_counts["Excellent"],
        "good": performance_counts["Good"],
        "average_count": performance_counts["Average"],
        "below_average": performance_counts["Below Average"],
        "poor": performance_counts["Poor"]
    }


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    global students

    prediction = None
    performance = None
    message = None
    message_type = None

    # ========================================================
    # MANUAL PREDICTION
    # ========================================================

    if request.method == "POST":

        try:

            if model is None:
                raise ValueError(
                    "Machine learning model could not be loaded."
                )

            name = request.form.get("name", "").strip()

            if not name:
                name = f"Student {len(students) + 1}"

            studytime = float(request.form.get("studytime", 0))
            absences = float(request.form.get("absences", 0))
            g1 = float(request.form.get("g1", 0))
            g2 = float(request.form.get("g2", 0))

            # ------------------------------------------------
            # VALIDATION
            # ------------------------------------------------

            if studytime < 0 or studytime > 12:
                raise ValueError(
                    "Study time must be between 0 and 12."
                )

            if absences < 0 or absences > 22:
                raise ValueError(
                    "Absences must be between 0 and 22."
                )

            if g1 < 0 or g1 > 20:
                raise ValueError(
                    "G1 must be between 0 and 20."
                )

            if g2 < 0 or g2 > 20:
                raise ValueError(
                    "G2 must be between 0 and 20."
                )

            # ------------------------------------------------
            # CREATE INPUT DATAFRAME
            # ------------------------------------------------

            input_data = pd.DataFrame(
                [[
                    studytime,
                    absences,
                    g1,
                    g2
                ]],
                columns=[
                    "studytime",
                    "absences",
                    "G1",
                    "G2"
                ]
            )

            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            pred = model.predict(input_data)[0]

            # Keep score between 0 and 20
            prediction = round(
                max(0, min(20, float(pred))),
                2
            )

            performance = get_performance(prediction)

            # ------------------------------------------------
            # UPDATE EXISTING STUDENT OR ADD NEW STUDENT
            # ------------------------------------------------

            existing_student = next(
                (
                    student
                    for student in students
                    if student["name"].lower() == name.lower()
                ),
                None
            )

            if existing_student:

                existing_student["score"] = prediction
                existing_student["performance"] = performance

            else:

                students.append({
                    "name": name,
                    "score": prediction,
                    "performance": performance
                })

            prepare_students()

            message = "Prediction completed successfully."
            message_type = "success"

        except Exception as e:

            prediction = None
            performance = None

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

    message = None
    message_type = None

    try:

        if model is None:
            raise ValueError(
                "Machine learning model is not available."
            )

        uploaded_file = request.files.get("file")

        if uploaded_file is None:
            raise ValueError(
                "Please select a CSV file."
            )

        if uploaded_file.filename == "":
            raise ValueError(
                "No file was selected."
            )

        if not uploaded_file.filename.lower().endswith(".csv"):
            raise ValueError(
                "Only CSV files are supported."
            )

        # ----------------------------------------------------
        # READ UPLOADED DATASET
        # ----------------------------------------------------

        uploaded_df = pd.read_csv(uploaded_file)

        required_columns = [
            "studytime",
            "absences",
            "G1",
            "G2"
        ]

        missing_columns = [
            col
            for col in required_columns
            if col not in uploaded_df.columns
        ]

        if missing_columns:

            raise ValueError(
                "Your dataset is missing these columns: "
                + ", ".join(missing_columns)
            )

        # Remove rows where required values are missing
        uploaded_df = uploaded_df.dropna(
            subset=required_columns
        )

        if uploaded_df.empty:

            raise ValueError(
                "The uploaded dataset contains no valid rows."
            )

        # ----------------------------------------------------
        # RESET CURRENT LEADERBOARD
        # ----------------------------------------------------

        students = []

        # ----------------------------------------------------
        # PREDICT ALL STUDENTS
        # ----------------------------------------------------

        for index, row in uploaded_df.iterrows():

            try:

                studytime = float(row["studytime"])
                absences = float(row["absences"])
                g1 = float(row["G1"])
                g2 = float(row["G2"])

                # Keep values within allowed limits
                studytime = max(
                    0,
                    min(12, studytime)
                )

                absences = max(
                    0,
                    min(22, absences)
                )

                g1 = max(
                    0,
                    min(20, g1)
                )

                g2 = max(
                    0,
                    min(20, g2)
                )

                input_data = pd.DataFrame(
                    [[
                        studytime,
                        absences,
                        g1,
                        g2
                    ]],
                    columns=[
                        "studytime",
                        "absences",
                        "G1",
                        "G2"
                    ]
                )

                pred = model.predict(input_data)[0]

                score = round(
                    max(0, min(20, float(pred))),
                    2
                )

                performance = get_performance(score)

                # ------------------------------------------------
                # NAME
                # ------------------------------------------------

                if "name" in uploaded_df.columns:

                    name = str(row["name"]).strip()

                    if not name or name.lower() == "nan":
                        name = f"Student {index + 1}"

                else:

                    name = f"Student {index + 1}"

                students.append({
                    "name": name,
                    "score": score,
                    "performance": performance
                })

            except Exception as row_error:

                print(
                    f"Skipping row {index + 1}:",
                    row_error
                )

        if not students:

            raise ValueError(
                "No valid student records could be predicted."
            )

        # Sort and rank
        prepare_students()

        message = (
            f"Successfully predicted performance "
            f"for {len(students)} students."
        )

        message_type = "success"

    except Exception as e:

        message = str(e)
        message_type = "error"

    dashboard = get_dashboard_data()

    return render_template(
        "index.html",
        prediction=None,
        performance=None,
        students=students,
        dashboard=dashboard,
        message=message,
        message_type=message_type
    )


# ============================================================
# DOWNLOAD RESULTS
# ============================================================

@app.route("/download")
def download():

    if not students:
        return "No prediction results available."

    result_df = pd.DataFrame(students)

    # Reorder columns
    result_df = result_df[
        [
            "rank",
            "name",
            "score",
            "performance"
        ]
    ]

    result_df.columns = [
        "Rank",
        "Student Name",
        "Predicted G3",
        "Performance"
    ]

    # Create CSV in memory
    output = io.StringIO()

    result_df.to_csv(
        output,
        index=False
    )

    output.seek(0)

    return send_file(
        io.BytesIO(
            output.getvalue().encode("utf-8")
        ),
        mimetype="text/csv",
        as_attachment=True,
        download_name="student_prediction_results.csv"
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
=======
from flask import Flask, request, render_template, send_file
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import io

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

# ============================================================
# LOAD TRAINING DATASET
# ============================================================

try:
    df = pd.read_csv("student_performance_dataset.csv")

    required_training_columns = [
        "studytime",
        "absences",
        "G1",
        "G2",
        "G3"
    ]

    missing_columns = [
        col for col in required_training_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "maths.csv is missing columns: "
            + ", ".join(missing_columns)
        )

    # Keep only required columns
    df = df[required_training_columns].copy()

    # Remove rows with missing values
    df = df.dropna()

    # G3 in the original dataset is out of 100.
    # Convert it to a score out of 20.
    

    # Features
    X = df[["studytime", "absences", "G1", "G2"]]

    # Target
    y = df["G3"]

    # ========================================================
    # TRAIN RANDOM FOREST MODEL
    # ========================================================

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(X, y)

    print("Training dataset loaded successfully.")
    print("Number of training records:", len(df))
    print("Model trained successfully.")

except Exception as e:
    print("ERROR LOADING TRAINING DATASET:", e)
    model = None


# ============================================================
# GLOBAL STORAGE
# ============================================================

students = []


# ============================================================
# PERFORMANCE CLASSIFICATION
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
# PREPARE LEADERBOARD
# ============================================================

def prepare_students():

    global students

    # Sort by score
    students = sorted(
        students,
        key=lambda x: x["score"],
        reverse=True
    )

    # Assign ranks
    for index, student in enumerate(students):
        student["rank"] = index + 1


# ============================================================
# DASHBOARD DATA
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

    scores = [student["score"] for student in students]

    performance_counts = {
        "Outstanding": 0,
        "Excellent": 0,
        "Good": 0,
        "Average": 0,
        "Below Average": 0,
        "Poor": 0
    }

    for student in students:
        performance_counts[student["performance"]] += 1

    return {
        "total": len(students),
        "average": round(sum(scores) / len(scores), 2),
        "highest": round(max(scores), 2),
        "lowest": round(min(scores), 2),

        "outstanding": performance_counts["Outstanding"],
        "excellent": performance_counts["Excellent"],
        "good": performance_counts["Good"],
        "average_count": performance_counts["Average"],
        "below_average": performance_counts["Below Average"],
        "poor": performance_counts["Poor"]
    }


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    global students

    prediction = None
    performance = None
    message = None
    message_type = None

    # ========================================================
    # MANUAL PREDICTION
    # ========================================================

    if request.method == "POST":

        try:

            if model is None:
                raise ValueError(
                    "Machine learning model could not be loaded."
                )

            name = request.form.get("name", "").strip()

            if not name:
                name = f"Student {len(students) + 1}"

            studytime = float(request.form.get("studytime", 0))
            absences = float(request.form.get("absences", 0))
            g1 = float(request.form.get("g1", 0))
            g2 = float(request.form.get("g2", 0))

            # ------------------------------------------------
            # VALIDATION
            # ------------------------------------------------

            if studytime < 0 or studytime > 12:
                raise ValueError(
                    "Study time must be between 0 and 12."
                )

            if absences < 0 or absences > 22:
                raise ValueError(
                    "Absences must be between 0 and 22."
                )

            if g1 < 0 or g1 > 20:
                raise ValueError(
                    "G1 must be between 0 and 20."
                )

            if g2 < 0 or g2 > 20:
                raise ValueError(
                    "G2 must be between 0 and 20."
                )

            # ------------------------------------------------
            # CREATE INPUT DATAFRAME
            # ------------------------------------------------

            input_data = pd.DataFrame(
                [[
                    studytime,
                    absences,
                    g1,
                    g2
                ]],
                columns=[
                    "studytime",
                    "absences",
                    "G1",
                    "G2"
                ]
            )

            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            pred = model.predict(input_data)[0]

            # Keep score between 0 and 20
            prediction = round(
                max(0, min(20, float(pred))),
                2
            )

            performance = get_performance(prediction)

            # ------------------------------------------------
            # UPDATE EXISTING STUDENT OR ADD NEW STUDENT
            # ------------------------------------------------

            existing_student = next(
                (
                    student
                    for student in students
                    if student["name"].lower() == name.lower()
                ),
                None
            )

            if existing_student:

                existing_student["score"] = prediction
                existing_student["performance"] = performance

            else:

                students.append({
                    "name": name,
                    "score": prediction,
                    "performance": performance
                })

            prepare_students()

            message = "Prediction completed successfully."
            message_type = "success"

        except Exception as e:

            prediction = None
            performance = None

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

    message = None
    message_type = None

    try:

        if model is None:
            raise ValueError(
                "Machine learning model is not available."
            )

        uploaded_file = request.files.get("file")

        if uploaded_file is None:
            raise ValueError(
                "Please select a CSV file."
            )

        if uploaded_file.filename == "":
            raise ValueError(
                "No file was selected."
            )

        if not uploaded_file.filename.lower().endswith(".csv"):
            raise ValueError(
                "Only CSV files are supported."
            )

        # ----------------------------------------------------
        # READ UPLOADED DATASET
        # ----------------------------------------------------

        uploaded_df = pd.read_csv(uploaded_file)

        required_columns = [
            "studytime",
            "absences",
            "G1",
            "G2"
        ]

        missing_columns = [
            col
            for col in required_columns
            if col not in uploaded_df.columns
        ]

        if missing_columns:

            raise ValueError(
                "Your dataset is missing these columns: "
                + ", ".join(missing_columns)
            )

        # Remove rows where required values are missing
        uploaded_df = uploaded_df.dropna(
            subset=required_columns
        )

        if uploaded_df.empty:

            raise ValueError(
                "The uploaded dataset contains no valid rows."
            )

        # ----------------------------------------------------
        # RESET CURRENT LEADERBOARD
        # ----------------------------------------------------

        students = []

        # ----------------------------------------------------
        # PREDICT ALL STUDENTS
        # ----------------------------------------------------

        for index, row in uploaded_df.iterrows():

            try:

                studytime = float(row["studytime"])
                absences = float(row["absences"])
                g1 = float(row["G1"])
                g2 = float(row["G2"])

                # Keep values within allowed limits
                studytime = max(
                    0,
                    min(12, studytime)
                )

                absences = max(
                    0,
                    min(22, absences)
                )

                g1 = max(
                    0,
                    min(20, g1)
                )

                g2 = max(
                    0,
                    min(20, g2)
                )

                input_data = pd.DataFrame(
                    [[
                        studytime,
                        absences,
                        g1,
                        g2
                    ]],
                    columns=[
                        "studytime",
                        "absences",
                        "G1",
                        "G2"
                    ]
                )

                pred = model.predict(input_data)[0]

                score = round(
                    max(0, min(20, float(pred))),
                    2
                )

                performance = get_performance(score)

                # ------------------------------------------------
                # NAME
                # ------------------------------------------------

                if "name" in uploaded_df.columns:

                    name = str(row["name"]).strip()

                    if not name or name.lower() == "nan":
                        name = f"Student {index + 1}"

                else:

                    name = f"Student {index + 1}"

                students.append({
                    "name": name,
                    "score": score,
                    "performance": performance
                })

            except Exception as row_error:

                print(
                    f"Skipping row {index + 1}:",
                    row_error
                )

        if not students:

            raise ValueError(
                "No valid student records could be predicted."
            )

        # Sort and rank
        prepare_students()

        message = (
            f"Successfully predicted performance "
            f"for {len(students)} students."
        )

        message_type = "success"

    except Exception as e:

        message = str(e)
        message_type = "error"

    dashboard = get_dashboard_data()

    return render_template(
        "index.html",
        prediction=None,
        performance=None,
        students=students,
        dashboard=dashboard,
        message=message,
        message_type=message_type
    )


# ============================================================
# DOWNLOAD RESULTS
# ============================================================

@app.route("/download")
def download():

    if not students:
        return "No prediction results available."

    result_df = pd.DataFrame(students)

    # Reorder columns
    result_df = result_df[
        [
            "rank",
            "name",
            "score",
            "performance"
        ]
    ]

    result_df.columns = [
        "Rank",
        "Student Name",
        "Predicted G3",
        "Performance"
    ]

    # Create CSV in memory
    output = io.StringIO()

    result_df.to_csv(
        output,
        index=False
    )

    output.seek(0)

    return send_file(
        io.BytesIO(
            output.getvalue().encode("utf-8")
        ),
        mimetype="text/csv",
        as_attachment=True,
        download_name="student_prediction_results.csv"
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000)
