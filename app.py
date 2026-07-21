import os
import joblib
import gradio as gr


# =====================================
# Load Model Safely
# =====================================

MODEL_PATH = "loan_prediction_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")


# =====================================
# Validation Function
# =====================================

def validate_inputs(
    Dependents,
    Education,
    SelfEmployed,
    Income,
    LoanAmount,
    LoanTerm,
    CIBIL,
    ResidentialAssets,
    CommercialAssets,
    LuxuryAssets,
    BankAssets,
):

    values = [
        Dependents,
        Income,
        LoanAmount,
        LoanTerm,
        CIBIL,
        ResidentialAssets,
        CommercialAssets,
        LuxuryAssets,
        BankAssets,
    ]

    names = [
        "Dependents",
        "Annual Income",
        "Loan Amount",
        "Loan Term",
        "CIBIL Score",
        "Residential Assets",
        "Commercial Assets",
        "Luxury Assets",
        "Bank Assets",
    ]

    for value, name in zip(values, names):
        if value is None:
            return False, f"❌ {name} cannot be empty."

    if Dependents < 0 or Dependents > 20:
        return False, "❌ Dependents must be between 0 and 20."

    if Income <= 0:
        return False, "❌ Annual Income must be greater than 0."

    if LoanAmount <= 0:
        return False, "❌ Loan Amount must be greater than 0."

    if LoanTerm <= 0:
        return False, "❌ Loan Term must be greater than 0."

    if CIBIL < 300 or CIBIL > 900:
        return False, "❌ CIBIL Score must be between 300 and 900."

    return True, ""


# =====================================
# Prediction Function
# =====================================

def predict_loan(
    Dependents,
    Education,
    SelfEmployed,
    Income,
    LoanAmount,
    LoanTerm,
    CIBIL,
    ResidentialAssets,
    CommercialAssets,
    LuxuryAssets,
    BankAssets,
):

    valid, message = validate_inputs(
        Dependents,
        Education,
        SelfEmployed,
        Income,
        LoanAmount,
        LoanTerm,
        CIBIL,
        ResidentialAssets,
        CommercialAssets,
        LuxuryAssets,
        BankAssets,
    )

    if not valid:
        return message

    try:

        education = 0 if Education == "Graduate" else 1
        self_employed = 0 if SelfEmployed == "Yes" else 1

        input_data = [[
            int(Dependents),
            education,
            self_employed,
            float(Income),
            float(LoanAmount),
            float(LoanTerm),
            float(CIBIL),
            float(ResidentialAssets),
            float(CommercialAssets),
            float(LuxuryAssets),
            float(BankAssets),
        ]]

        prediction = model.predict(input_data)[0]

        if prediction == 0:
            return (
                "✅ LOAN APPROVED\n\n"
                "Prediction: Approved\n\n"
                "Congratulations! Your loan is likely to be approved."
            )
        else:
            return (
                "❌ LOAN REJECTED\n\n"
                "Prediction: Rejected\n\n"
                "The loan application is likely to be rejected."
            )

    except Exception as e:
        return f"❌ Prediction Error:\n\n{e}"    
 # =====================================
# Gradio Interface
# =====================================

interface = gr.Interface(
    fn=predict_loan,

    inputs=[
        gr.Number(label="Dependents"),
        gr.Dropdown(
            choices=["Graduate", "Not Graduate"],
            value="Graduate",
            label="Education"
        ),
        gr.Dropdown(
            choices=["Yes", "No"],
            value="No",
            label="Self Employed"
        ),
        gr.Number(label="Annual Income"),
        gr.Number(label="Loan Amount"),
        gr.Number(label="Loan Term"),
        gr.Slider(
            minimum=300,
            maximum=900,
            value=750,
            step=1,
            label="CIBIL Score"
        ),
        gr.Number(label="Residential Assets"),
        gr.Number(label="Commercial Assets"),
        gr.Number(label="Luxury Assets"),
        gr.Number(label="Bank Assets"),
    ],

    outputs=gr.Textbox(
        label="Prediction Result",
        lines=6
    ),

    title="🏦 AI Loan Approval Prediction System",

    description="""
This web application predicts whether a loan is likely to be approved or rejected using a trained Machine Learning model.

📌 Developed by: **Prachi Valecha**

🏫 **Panipat Institute of Engineering and Technology (PIET), Panipat**
""",

    article="""
### Instructions

• Enter all applicant details.

• Click **Submit**.

• The prediction is generated using the trained Machine Learning model.

⚠️ This application is for educational purposes only and should not be used as an official banking decision.
""",

    examples=[
        [2, "Graduate", "No", 5000000, 1000000, 240, 780, 2000000, 500000, 300000, 1000000],
        [4, "Not Graduate", "Yes", 2500000, 1500000, 360, 620, 500000, 300000, 100000, 200000],
        [1, "Graduate", "No", 8000000, 2000000, 180, 850, 4000000, 1000000, 700000, 3000000],
    ]
)

# =====================================
# Launch
# =====================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 7860))

    interface.launch(
        server_name="0.0.0.0",
        server_port=port
    )
