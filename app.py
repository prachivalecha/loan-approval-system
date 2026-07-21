import os
import gradio as gr
import joblib
import numpy as np

# ==========================
# Load Model
# ==========================

try:
    model = joblib.load("loan_prediction_model.pkl")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# ==========================
# Prediction Function
# ==========================

def predict_loan(
    dependents,
    education,
    self_employed,
    income,
    loan_amount,
    loan_term,
    cibil,
    residential_assets,
    commercial_assets,
    luxury_assets,
    bank_assets,
):

    try:

        # ---------------- Validation ----------------

        if income <= 0:
            return "❌ Annual Income must be greater than 0."

        if loan_amount <= 0:
            return "❌ Loan Amount must be greater than 0."

        if loan_term <= 0:
            return "❌ Loan Term must be greater than 0."

        if not (300 <= cibil <= 900):
            return "❌ CIBIL Score must be between 300 and 900."

        # -------- Encoding (same as training notebook) --------

        education = 0 if education == "Graduate" else 1
        self_employed = 0 if self_employed == "Yes" else 1

        input_data = np.array([[
            dependents,
            education,
            self_employed,
            income,
            loan_amount,
            loan_term,
            cibil,
            residential_assets,
            commercial_assets,
            luxury_assets,
            bank_assets
        ]])

        prediction = model.predict(input_data)[0]

        confidence = None

        if hasattr(model, "predict_proba"):
            confidence = np.max(model.predict_proba(input_data)) * 100

        # 0 = Approved
        # 1 = Rejected

        if int(prediction) == 0:
            result = """
# 💖 Loan Approved

### 🎉 Congratulations!

Your loan application is likely to be **APPROVED**.
"""
        else:
            result = """
# ❌ Loan Rejected

Unfortunately, your loan application is likely to be **REJECTED**.
"""

        if confidence is not None:
            result += f"\n\n### Confidence : **{confidence:.2f}%**"

        return result

    except Exception as e:
        return f"❌ Error : {str(e)}"


# ==========================
# Pink Theme
# ==========================

pink = gr.themes.Soft(
    primary_hue="pink",
    secondary_hue="rose",
)

css = """

body{
background:linear-gradient(135deg,#fff0f6,#ffe6f2);
}

.gradio-container{

max-width:1050px !important;

margin:auto;

background:white;

padding:25px;

border-radius:25px;

box-shadow:0px 0px 25px rgba(255,105,180,.25);

}

h1{

text-align:center;

color:#e91e63;

font-size:40px;

}

h2,h3{

text-align:center;

color:#d81b60;

}

button{

background:#ff69b4 !important;

border:none !important;

color:white !important;

font-size:18px !important;

font-weight:bold !important;

}

button:hover{

background:#ec4899 !important;

}

footer{

display:none !important;

}

"""
# ==========================
# Interface
# ==========================

with gr.Blocks(theme=pink, css=css) as demo:

    gr.Markdown("""
# 🏦 AI Loan Approval Prediction System

### 💖 Smart Machine Learning Based Loan Approval Prediction

---

### 👩‍💻 Developed By
## **Prachi Valecha**

### 🎓 Panipat Institute of Engineering and Technology

---
""")

    with gr.Row():

        dependents = gr.Slider(
            minimum=0,
            maximum=10,
            value=0,
            step=1,
            label="👨‍👩‍👧 Number of Dependents"
        )

        education = gr.Radio(
            ["Graduate", "Not Graduate"],
            value="Graduate",
            label="🎓 Education"
        )

        self_employed = gr.Radio(
            ["Yes", "No"],
            value="No",
            label="💼 Self Employed"
        )

    with gr.Row():

        income = gr.Number(
            label="💰 Annual Income"
        )

        loan_amount = gr.Number(
            label="🏦 Loan Amount"
        )

        loan_term = gr.Number(
            label="📅 Loan Term (Months)"
        )

    cibil = gr.Slider(
        minimum=300,
        maximum=900,
        value=700,
        step=1,
        label="📈 CIBIL Score"
    )

    with gr.Row():

        residential_assets = gr.Number(
            label="🏠 Residential Assets"
        )

        commercial_assets = gr.Number(
            label="🏢 Commercial Assets"
        )

    with gr.Row():

        luxury_assets = gr.Number(
            label="🚗 Luxury Assets"
        )

        bank_assets = gr.Number(
            label="🏦 Bank Assets"
        )

    predict_btn = gr.Button(
        "💖 Predict Loan Status",
        variant="primary",
        size="lg"
    )

    output = gr.Markdown()

    predict_btn.click(
        fn=predict_loan,
        inputs=[
            dependents,
            education,
            self_employed,
            income,
            loan_amount,
            loan_term,
            cibil,
            residential_assets,
            commercial_assets,
            luxury_assets,
            bank_assets
        ],
        outputs=output
    )

    gr.Markdown("""
---

## 🌸 Tips

✔ Higher CIBIL Score generally improves approval chances.

✔ Higher Annual Income strengthens your profile.

✔ Larger Assets increase eligibility.

✔ Lower Loan Amount compared to Income improves approval chances.

---

💖 **Made with Python • Gradio • Scikit-Learn**

""")

# ==========================
# Launch
# ==========================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 7860))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False
    )
