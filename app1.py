import os
import time
from flask import Flask, render_template, request, jsonify, send_file
from dotenv import load_dotenv
from PIL import Image
import requests
from langchain_groq import ChatGroq
from fpdf import FPDF

load_dotenv()
app = Flask(__name__)

UPLOAD_FOLDER = '/tmp/uploads'
OUTPUT_FOLDER = '/tmp/outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

API_NINJAS_KEY = os.getenv("API_NINJAS_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    api_key=GROQ_API_KEY
)
def cleanup_old_files():
    try:
        now = time.time()
        for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
            for f in os.listdir(folder):
                f_path = os.path.join(folder, f)
                if os.stat(f_path).st_mtime < now - 600: 
                    os.remove(f_path)
    except Exception as e:
        print(f"Cleanup Error: {e}")

@app.route('/health')
def health_check():
    ocr_key = os.getenv("API_NINJAS_KEY")
    llm_key = os.getenv("GROQ_API_KEY")
    
    status = {
        "status": "online",
        "ocr_key_present": bool(ocr_key),
        "llm_key_present": bool(llm_key),
        "upload_folder_exists": os.path.exists(UPLOAD_FOLDER)
    }
    return jsonify(status)

def optimize_image(image_path, max_size_kb=200):
    file_size = os.path.getsize(image_path) / 1024
    
    if file_size <= max_size_kb:
        return image_path

    print(f"Compressing {file_size:.2f}KB image...")
    img = Image.open(image_path)

    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    if img.width > 1024:
        ratio = 1024 / float(img.width)
        new_height = int(float(img.height) * float(ratio))
        img = img.resize((1024, new_height), Image.Resampling.LANCZOS)

    optimized_filename = f"compressed_{os.path.basename(image_path)}"
    optimized_path = os.path.join(UPLOAD_FOLDER, optimized_filename)
    
    img.save(optimized_path, optimize=True, quality=40)
    
    return optimized_path

def get_ocr_text(image_path):
    """
    Sends image to API Ninjas for OCR.
    """
    url = "https://api.api-ninjas.com/v1/imagetotext"
    with open(image_path, "rb") as img_file:
        files = {"image": img_file}
        headers = {"X-Api-Key": API_NINJAS_KEY}
        resp = requests.post(url, files=files, headers=headers)
        
        if resp.status_code != 200:
            raise Exception(f"OCR API Error: {resp.text}")
            
        data = resp.json()
        text = " ".join([block["text"].strip() for block in data if block["text"].strip()])
        return text

def generate_pdf(text_content, filename):
    """
    Generates a PDF from text.
    """
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(0, 70, 140)
    pdf.cell(0, 10, "Medical Report Analysis", ln=True, align="C")
    
    pdf.ln(5)
    pdf.set_font("Arial", size=11)
    pdf.set_text_color(0, 0, 0)
    
    safe_text = text_content.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 6, safe_text)
    
    filepath = os.path.join(OUTPUT_FOLDER, filename)
    pdf.output(filepath)
    return filepath


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    cleanup_old_files() 

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    filename = f"{int(time.time())}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        final_path = optimize_image(filepath)
        raw_text = get_ocr_text(final_path)
        
        if not raw_text:
            return jsonify({"error": "Could not read text. Image might be blurry or not a medical report."}), 400

        prompt = f"""
        You are a Medical Report Interpretation AI. You will analyze ANY type of medical document (blood test, scan report, radiology, ECG, discharge summary, PCR, microbiology, biopsy, or any hospital report) and produce two sections:

SECTION 1: PATIENT EXPLANATION
SECTION 2: DOCTOR SUMMARY

Follow these rules exactly:

GENERAL RULES:

NO markdown. Pure text only.
Do not add facts not found in the report.
Do not guess diagnoses, severities, or abnormal values.
Expand medical abbreviations when needed.
Keep the tone calm, neutral, and medically safe.
If reference ranges are missing, explicitly say so.
If meaning is unclear, say “cannot be determined from this report.”
Never dump a long list of numbers without grouping.
Always separate raw values (findings) and value changes (trends).

SECTION 1: PATIENT EXPLANATION (simple, friendly, non-technical)

Write in short paragraphs. Include:
What kind of test/report this is.
What this type of test usually checks for.
A simple explanation of the important values and trends in the report.
Only interpret what is clearly supported by the given data.
Mention when results cannot be judged because reference ranges or clinical context are missing.
End with: “Only your doctor can confirm what these results mean for you.
The explanation must be easy enough for a person with no medical background.
Use everyday language and avoid medical terms unless necessary.
Keep sentences short and direct.
Never repeat all numbers unless they are essential to explain a trend.

SECTION 2: DOCTOR SUMMARY (fast-reading, point-based clinical notes)

Write in short numbered points, like a clinician's quick-review summary.
Include:
Report type (e.g., biochemistry, CBC, radiology, ECG).
Key findings explicitly mentioned in the report.
Trends or comparisons if multiple samples exist.
Relevant systems involved (hepatic, renal, hematologic, etc.).
Possible meaning of trends ONLY based on data (e.g., “trend suggests resolving leukocytosis”).
Limitations such as missing reference ranges, missing history, missing timestamps.
Any recommendations given in the report (if present).
Clinical follow-up required (e.g., “clinical correlation needed”).
Keep the doctor section concise, high-signal, and strictly based on the provided data.
Group raw values by system (Hepatic, Renal, Electrolytes, Hematology).
Show changes using “X→Y” format instead of listing both numbers separately.
Do not mix Findings and Trends. Findings = raw values. Trends = changes only.
Keep the doctor summary highly structured and compressed for speed-reading.
        
        REPORT TEXT:
        {raw_text}
        """
        
        response = llm.invoke(prompt)
        analysis_text = response.content

        pdf_filename = f"Report_Analysis_{int(time.time())}.pdf"
        generate_pdf(analysis_text, pdf_filename)

        return jsonify({
            "success": True,
            "analysis": analysis_text,
            "pdf_url": f"/download/{pdf_filename}"
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)