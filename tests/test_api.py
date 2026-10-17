from src.api import Patient


def test_patient_schema_accepts_valid_payload():
    item = Patient(
        age=55,
        sex=1,
        cp=2,
        trestbps=130,
        chol=240,
        fbs=0,
        restecg=1,
        thalach=150,
        exang=0,
        oldpeak=1.2,
        slope=1,
        ca=0,
        thal=2,
    )
    assert item.age == 55
