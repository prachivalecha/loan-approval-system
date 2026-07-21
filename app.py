import os
import gradio as gr
import joblib
import numpy as np

# =====================================
# Load Model
# =====================================
try:
    model = joblib.load("loan_prediction_model.pkl")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")


# =====================================
# Prediction Function
# =====================================
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

        # Validation
        if income <= 0:
            return "❌ Annual Income must be greater than 0."

        if loan_amount <= 0:
            return "❌ Loan Amount must be greater than 0."

        if loan_term <= 0:
            return "❌ Loan Term must be greater than 0."

        if not (300 <= cibil <= 900):
            return "❌ CIBIL Score must be between 300 and 900."

        # Encode categorical variables
        education = 1 if education == "Graduate" else 0
        self_employed = 1 if self_employed == "Yes" else 0

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

        if prediction in [1, "1", "Approved", "approved", "Yes", "yes"]:
            result = "## ✅ Loan Approved 🎉"
        else:
            result = "## ❌ Loan Rejected"

        if confidence is not None:
            result += f"\n\n### Confidence: **{confidence:.2f}%**"

        return result

    except Exception as e:
        return f"❌ Error: {str(e)}"


# =====================================
# Custom CSS
# =====================================

css = """
body{
    background:#0d1117;
}

.gradio-container{
    max-width:1000px !important;
    margin:auto;
}

h1,h2,h3{
    text-align:center;
}

footer{
    display:none !important;
}
"""


# =====================================
# Interface
# =====================================

with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:

    gr.Markdown("""
# 🏦 AI Loan Approval Prediction System

### Predict whether a loan will be **Approved** or **Rejected**

---

### 👩‍💻 Developed By
**Prachi Valecha**

### 🎓 Panipat Institute of Engineering and Technology

---
""")

    with gr.Row():

        dependents = gr.Slider(
            0, 10,
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

        income = gr.Number(label="💰 Annual Income")

        loan_amount = gr.Number(label="🏦 Loan Amount")

        loan_term = gr.Number(label="📅 Loan Term (Months)")

    cibil = gr.Slider(
        300,
        900,
        value=700,
        step=1,
        label="📈 CIBIL Score"
    )

    with gr.Row():

        residential_assets = gr.Number(label="🏠 Residential Assets")

        commercial_assets = gr.Number(label="🏢 Commercial Assets")

    with gr.Row():

        luxury_assets = gr.Number(label="🚗 Luxury Assets")

        bank_assets = gr.Number(label="🏦 Bank Assets")

    predict = gr.Button(
        "🔍 Predict Loan Status",
        variant="primary"
    )

    output = gr.Markdown()

    predict.click(
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
            bank_assets,
        ],
        outputs=output,
    )

    gr.Markdown("""
---
### 💡 Tips

✅ Higher CIBIL Score generally improves approval chances.

✅ Stable income increases eligibility.

✅ Higher assets strengthen your loan profile.

---
Made with ❤️ using **Python | Gradio | Scikit-Learn**
""")


# =====================================
# Launch for Render
# =====================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False
    )
