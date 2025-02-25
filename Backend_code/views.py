from django.shortcuts import render
from django.http import HttpRequest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def bmi(request):
    value = ""
    if request.method == "POST":
        gender = float(request.POST["gender"])
        height = float(request.POST["height"])
        weight = float(request.POST["weight"])

        bmiv = (weight * 10000.0) / (height * height)
        if bmiv < 18.5:
            value = "Underweight:("
        elif (bmiv >= 18.5) and (bmiv < 24.9):
            value = "Normal:)"
        elif (bmiv >= 25) and (bmiv < 29.9):
            value = "Overweight:("
        elif bmiv >= 30:
            value = "Obese:("
        else:
            value = "Please enter a valid input."

    return render(
        request,
        "bmi.html",
        {
            "context": value,
            "title": "Body Mass Index (Health Status)",
            "active": "btn btn-success peach-gradient text-violet",
            "bmi": True,
            "background": "bg-primary text-dark",
        },
    )


def heart(request):
    df = pd.read_csv("Disease_Datasets/Heart_train.csv")
    data = df.values
    X = data[:, :-1]
    Y = data[:, -1]

    value = ""

    if request.method == "POST":
        age = float(request.POST["age"])
        sex = float(request.POST["sex"])
        cp = float(request.POST["cp"])
        trestbps = float(request.POST["trestbps"])
        chol = float(request.POST["chol"])
        fbs = float(request.POST["fbs"])
        restecg = float(request.POST["restecg"])
        thalach = float(request.POST["thalach"])
        exang = float(request.POST["exang"])
        oldpeak = float(request.POST["oldpeak"])
        slope = float(request.POST["slope"])
        ca = float(request.POST["ca"])
        thal = float(request.POST["thal"])

        user_data = np.array(
            [
                age,
                sex,
                cp,
                trestbps,
                chol,
                fbs,
                restecg,
                thalach,
                exang,
                oldpeak,
                slope,
                ca,
                thal,
            ]
        ).reshape(1, -1)

        rf = RandomForestClassifier(n_estimators=16, criterion="entropy", max_depth=9)
        rf.fit(np.nan_to_num(X), Y)
        predictions = rf.predict(user_data)

        if int(predictions[0]) == 1:
            value = "have:("
        elif int(predictions[0]) == 0:
            value = "don't have:)"

    return render(
        request,
        "heart.html",
        {
            "context": value,
            "title": "Heart Disease Prediction",
            "active": "btn btn-success peach-gradient text-white",
            "heart": True,
            "background": "bg-danger text-white",
        },
    )


def diabetes(request):
    dfx = pd.read_csv("Disease_Datasets/Diabetes_XTrain.csv")
    dfy = pd.read_csv("Disease_Datasets/Diabetes_YTrain.csv")
    X = dfx.values
    Y = dfy.values
    Y = Y.reshape((-1,))

    value = ""
    if request.method == "POST":
        pregnancies = float(request.POST["pregnancies"])
        glucose = float(request.POST["glucose"])
        bloodpressure = float(request.POST["bloodpressure"])
        skinthickness = float(request.POST["skinthickness"])
        bmi = float(request.POST["bmi"])
        insulin = float(request.POST["insulin"])
        pedigree = float(request.POST["pedigree"])
        age = float(request.POST["age"])

        user_data = np.array(
            [
                pregnancies,
                glucose,
                bloodpressure,
                skinthickness,
                bmi,
                insulin,
                pedigree,
                age,
            ]
        ).reshape(1, -1)

        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X, Y)
        predictions = knn.predict(user_data)

        if int(predictions[0]) == 1:
            value = "have diabetes:("
        elif int(predictions[0]) == 0:
            value = "don't have:)"

    return render(
        request,
        "diabetes.html",
        {
            "context": value,
            "title": "Diabetes Disease Prediction",
            "active": "btn btn-success peach-gradient text-white",
            "diabetes": True,
            "background": "bg-dark text-white",
        },
    )


def breast(request):
    df = pd.read_csv("Disease_Datasets/Breast_train.csv")
    data = df.values
    X = data[:, :-1]
    Y = data[:, -1]

    value = ""
    if request.method == "POST":
        radius = float(request.POST["radius"])
        texture = float(request.POST["texture"])
        perimeter = float(request.POST["perimeter"])
        area = float(request.POST["area"])
        smoothness = float(request.POST["smoothness"])

        rf = RandomForestClassifier(n_estimators=16, criterion="entropy", max_depth=5)
        rf.fit(np.nan_to_num(X), Y)

        user_data = np.array((radius, texture, perimeter, area, smoothness)).reshape(1,-1)
        predictions = rf.predict(user_data)

        if int(predictions[0]) == 1:
            value = "have :("
        elif int(predictions[0]) == 0:
            value = "don't have :)"

    return render(
        request,
        "breast_cancer.html",
        {
            "context": value,
            "title": "Breast Cancer Prediction",
            "active": "btn btn-success peach-gradient text-white",
            "breast": True,
            "background": "bg-warning text-dark",
        },
    )


def blood_pressure(request: HttpRequest):
    value = ""
    if request.method == "POST":
        age = request.POST.get("age")
        weight = request.POST.get("weight")
        gender = request.POST.get("gender")
        num_medications = request.POST.get("num_medications")
        num_lab_procedures = request.POST.get("num_lab_procedures")
        diag_1 = request.POST.get("diag_1")
        medical_specialty = request.POST.get("medical_specialty")
        max_glu_serum = request.POST.get("max_glu_serum")
        A1Cresult = request.POST.get("A1Cresult")
        admission_typeid = request.POST.get("admission_typeid")
        time_in_hospital = request.POST.get("time_in_hospital")

        # Validate and process the data
        try:
            age = float(age)
            weight = float(weight)
            num_medications = int(num_medications)
            num_lab_procedures = int(num_lab_procedures)
            admission_typeid = int(admission_typeid)
            time_in_hospital = int(time_in_hospital)

            # Here you would typically call your prediction model
            # For demonstration, we'll use a simple logic
            if age < 30 and weight < 70:
                value = "Normal Blood Pressure"
            else:
                value = "High Blood Pressure"

        except ValueError:
            value = "Please enter valid numeric values."

    return render(
        request,
        "blood_pressure.html",
        {
            "context": value,
            "title": "Blood Pressure Prediction",
            "active": "btn btn-success peach-gradient text-white",
            "blood_pressure": True,
            "background": "bg-info text-dark",
        },
    )


def covid_19(request: HttpRequest):
    context = ""
    if request.method == "POST":
        fever = request.POST.get("fever")
        body_pain = request.POST.get("bodyPain")
        age = request.POST.get("age")
        runny_nose = request.POST.get("runnyNose")
        diff_breath = request.POST.get("diffBreath")
        infection_prob = request.POST.get("infectionProb")

        try:
            # Convert input values to appropriate types
            fever = float(fever)
            body_pain = float(body_pain)
            age = float(age)
            runny_nose = float(runny_nose)
            diff_breath = float(diff_breath)
            infection_prob = float(infection_prob)

            # Placeholder for the prediction logic
            # For example, we use a simple threshold-based approach
            if fever > 0 and diff_breath > 0 and infection_prob > 0.5:
                context = "High risk of COVID-19"
            else:
                context = "Low risk of COVID-19"

        except ValueError:
            context = "Invalid input. Please enter valid numbers."

    return render(
        request,
        "covid_19.html",
        {
            "context": context,
            "title": "COVID-19 Symptoms Checker",
        },
    )

def home(request):
    return render(request, "home.html")
