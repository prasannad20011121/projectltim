# AI Medical Report Generator

An intelligent application that automates the diagnosis of medical scans (MRI/X-ray) and generates professional, multi-modal medical reports using deep learning and Large Language Models (LLMs).

## 🚀 Key Features

- **Automated Organ Detection**: Automatically identifies the organ in the uploaded scan (Brain, Bone, Breast, or Kidney).
- **Specialized Classification**:
  - **Brain**: Glioma, Meningioma, Pituitary, or No Tumor.
  - **Bone**: Fracture detection (Fractured vs. Not Fractured).
  - **Breast**: Tumor classification (Benign vs. Malignant).
  - **Kidney**: Detection of Cysts, Stones, Tumors, or Normal.
- **AI-Powered Reports**: Integrated with **Google Gemini Pro** to generate detailed clinical reports.
- **Dual Reporting Modes**:
  - **Doctor Mode**: Technical, clinical language for medical professionals.
  - **Patient Mode**: Simplified, empathetic explanations for patients.
- **Professional PDF Export**: Generate and download a formatted PDF report including the scan image and AI findings.

## 🛠️ Technology Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Deep Learning**: [TensorFlow](https://www.tensorflow.org/) / Keras
- **LLM**: [Google Generative AI (Gemini Pro)](https://ai.google.dev/)
- **PDF Generation**: [ReportLab](https://www.reportlab.com/)
- **Image Processing**: [Pillow (PIL)](https://python-pillow.org/)

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Clinical_MRI_Report_Generator_Multi
```

### 2. Install Dependencies
Ensure you have Python 3.8+ installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Configure API Key
The application uses Google Gemini for report generation. You need a Gemini API key.
- Get your key from [Google AI Studio](https://aistudio.google.com/).
- Set it as an environment variable:
  ```bash
  export GEMINI_API_KEY="your_api_key_here"
  ```
  *(On Windows PowerShell: `$env:GEMINI_API_KEY="your_api_key_here"`) *

### 4. Run the Application
```bash
streamlit run app.py
```
*Note: The application will automatically download the required `.keras` models from Google Drive on the first run if they are not present in the `models/` directory.*

## 📖 Usage Guide

1. **Upload**: Drag and drop a medical scan (JPG, JPEG, or PNG).
2. **Select Mode**: Choose between "Doctor Mode" or "Patient Mode".
3. **Analyze**: Click "Generate Report" to start the AI analysis.
4. **Review**: Read the AI-generated findings and the detailed LLM report.
5. **Download**: Click "Download PDF" to save the professional report.

## ⚠️ Medical Disclaimer

**This software is for educational and research purposes only.** It is not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

---
© 2024 AI Medical Report Generator
