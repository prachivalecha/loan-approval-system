import os
import joblib
import numpy as np
import gradio as gr

MODEL_PATH = "loan_prediction_model.pkl"
model = joblib.load(MODEL_PATH)

def predict_loan(dependents, education, self_employed, income, loan_amount, loan_term, cibil,
                 residential_assets, commercial_assets, luxury_assets, bank_assets):

    education = 0 if education == "Graduate" else 1
    self_employed = 0 if self_employed == "Yes" else 1

    X = np.array([[
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

    p = int(model.predict(X)[0])

    if p == 0:
        return """
        <div style='text-align:center;'>
        <h2 style='color:#E91E63;'>🎉 Loan Approved</h2>
        </div>
        """
    else:
        return """
        <div style='text-align:center;'>
        <h2 style='color:#D32F2F;'>❌ Loan Rejected</h2>
        </div>
        """

css = """
body,.gradio-container{
    background:#FFF8FC;
}

.gradio-container{
    max-width:900px !important;
    margin:auto;
    padding:20px;
}

footer{
    display:none !important;
}
"""

with gr.Blocks(css=css) as demo:

    gr.Markdown("""
# 🏦 AI Loan Approval Prediction

### Predict Loan Approval using Machine Learning
""")

    dep = gr.Number(label="Dependents")
    edu = gr.Dropdown(["Graduate", "Not Graduate"], value="Graduate", label="Education")
    se = gr.Dropdown(["Yes", "No"], value="No", label="Self Employed")
    inc = gr.Number(label="Annual Income")
    la = gr.Number(label="Loan Amount")
    lt = gr.Number(label="Loan Term")
    cs = gr.Slider(300, 900, value=750, label="CIBIL Score")
    ra = gr.Number(label="Residential Assets")
    ca = gr.Number(label="Commercial Assets")
    lux = gr.Number(label="Luxury Assets")
    ba = gr.Number(label="Bank Assets")

    out = gr.HTML()

    gr.Button("🔍 Predict Loan").click(
        predict_loan,
        [dep, edu, se, inc, la, lt, cs, ra, ca, lux, ba],
        out
    )

    gr.Markdown("""
---

## 👩‍💻 Developed By

# 🌸 Prachi Valecha

### 🎓 Panipat Institute of Engineering and Technology

💖 **AI Loan Approval Prediction using Machine Learning**
""")

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
