import os
import joblib
import numpy as np
import gradio as gr

# =====================================================
# Load Model
# =====================================================

try:
    model = joblib.load("loan_prediction_model.pkl")
except Exception as e:
    raise RuntimeError(f"Model Loading Error : {e}")


# =====================================================
# Prediction Function
# =====================================================

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

        if income <= 0:
            return "❌ Annual Income should be greater than 0."

        if loan_amount <= 0:
            return "❌ Loan Amount should be greater than 0."

        if loan_term <= 0:
            return "❌ Loan Term should be greater than 0."

        if cibil < 300 or cibil > 900:
            return "❌ CIBIL Score must be between 300 and 900."

        # Encoding exactly like training

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
            confidence = np.max(model.predict_proba(input_data))*100

        # 0 = Approved
        # 1 = Rejected

        if int(prediction)==0:

            result=f"""

<div style="background:#FFE7F2;
padding:30px;
border-radius:25px;
border:3px solid #FF6FB0;
text-align:center;
box-shadow:0px 10px 25px rgba(255,105,180,.25);">

<h1 style="color:#E91E63;">💖 Loan Approved</h1>

<h2 style="color:#C2185B;">🎉 Congratulations!</h2>

<p style="font-size:20px;color:#333;">
Your Loan Application has been
<b style="color:#E91E63;">APPROVED</b>.
</p>

</div>

"""

        else:

            result=f"""

<div style="background:#FFF0F0;
padding:30px;
border-radius:25px;
border:3px solid #FF8A80;
text-align:center;
box-shadow:0px 10px 25px rgba(255,0,0,.15);">

<h1 style="color:#D32F2F;">❌ Loan Rejected</h1>

<h2 style="color:#E53935;">Please Try Again</h2>

<p style="font-size:20px;color:#333;">
Your Loan Application has been
<b style="color:#D32F2F;">REJECTED</b>.
</p>

</div>

"""

        if confidence is not None:

            result += f"""

<br>

<div style="background:white;
padding:18px;
border-radius:15px;
border:2px solid #FFD4E8;
text-align:center;">

<h3 style="color:#E91E63;">
Prediction Confidence : {confidence:.2f}%
</h3>

</div>

"""

        return result

    except Exception as e:

        return f"❌ {e}"


# =====================================================
# PREMIUM CSS
# =====================================================

css="""

body{

background:linear-gradient(135deg,#FFF6FB,#FFE6F1);

font-family:'Segoe UI';

}

.gradio-container{

max-width:1100px !important;

margin:25px auto !important;

background:white !important;

border-radius:25px !important;

padding:35px !important;

box-shadow:0px 15px 40px rgba(255,105,180,.18);

}

/* Headings */

h1{

color:#E91E63 !important;

font-weight:800 !important;

}

h2{

color:#EC407A !important;

}

h3{

color:#F06292 !important;

}

/* ALL TEXT */

p,
span,
div,
label,
li{

color:#333333 !important;

}

/* Labels */

label{

font-weight:bold !important;

color:#C2185B !important;

}

/* Inputs */

input,
textarea{

border-radius:12px !important;

background:#FFF9FC !important;

border:2px solid #FFD3E8 !important;

}

/* Slider */

input[type=range]{

accent-color:#FF4FA2 !important;

}

/* Radio */

input[type=radio]{

accent-color:#FF4FA2 !important;

}

/* Button */

button{

background:linear-gradient(90deg,#FF80BF,#FF4FA2) !important;

color:white !important;

font-size:18px !important;

font-weight:bold !important;

border-radius:15px !important;

border:none !important;

box-shadow:0px 10px 20px rgba(255,105,180,.25);

}

button:hover{

background:linear-gradient(90deg,#FF5CA8,#FF2D8B)!important;

transform:scale(1.02);

}

footer{

display:none !important;

}

"""
# =====================================================
# Theme
# =====================================================

pink_theme = gr.themes.Soft(
    primary_hue="pink",
    secondary_hue="rose",
    neutral_hue="slate",
)

# =====================================================
# Interface
# =====================================================

with gr.Blocks(
    theme=pink_theme,
    css=css,
    title="AI Loan Approval Prediction"
) as demo:

    gr.Markdown("""

# 🏦 AI Loan Approval Prediction System

### 🌸 Smart Machine Learning Based Loan Approval

---

### 👩‍💻 **Developed By**

## **Prachi Valecha**

### 🎓 Panipat Institute of Engineering and Technology

---

""")

    with gr.Group():

        gr.Markdown("## 👤 Applicant Information")

        with gr.Row():

            dependents = gr.Slider(
                minimum=0,
                maximum=10,
                value=0,
                step=1,
                label="👨‍👩‍👧 Number of Dependents"
            )

            education = gr.Radio(
                choices=["Graduate","Not Graduate"],
                value="Graduate",
                label="🎓 Education"
            )

            self_employed = gr.Radio(
                choices=["Yes","No"],
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

    with gr.Group():

        gr.Markdown("## 🏠 Asset Information")

        with gr.Row():

            residential_assets = gr.Number(
                label="🏡 Residential Assets"
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

    gr.Markdown("<br>")

    predict_btn = gr.Button(
        "💖 Predict Loan Approval",
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

## 🌸 Tips for Better Loan Approval

✅ Maintain a Higher **CIBIL Score**

✅ Keep a Stable Annual Income

✅ Maintain Good Bank Assets

✅ Lower Loan Amount improves approval chances

✅ Keep Financial History Clean

---

""")
# =====================================================
# Footer
# =====================================================

gr.Markdown("""

---

<div style="text-align:center;padding:15px;">

### 💖 Thank You for Using the AI Loan Approval System

Made with ❤️ using **Python • Scikit-Learn • Gradio**

**Developed By**

### 👩‍💻 Prachi Valecha

**Panipat Institute of Engineering and Technology**

</div>

""")

# =====================================================
# Launch App
# =====================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))

    demo.launch(

        server_name="0.0.0.0",

        server_port=port,

        share=False,

        show_error=True

    )
