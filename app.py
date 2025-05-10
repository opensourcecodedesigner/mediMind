from flask import Flask, request, jsonify, render_template
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import os
from PyPDF2 import PdfReader
from io import BytesIO
import json

# Set your Gemini API key here
GEMINI_API_KEY = "AIzaSyCIXAA9OTSHpPON2xin-1ziqGaUpbEsuCE"  # Replace this with your actual API key

# Download required NLTK data
nltk.download('punkt')

app = Flask(__name__)

# Sample medical knowledge base
medical_kb = {
    "What is diabetes?": "Diabetes is a chronic disease that affects the way your body turns food into energy.",
    "What are the symptoms of diabetes?": "Increased thirst and urination, fatigue, blurred vision, and slow healing of cuts and wounds.",
}

# Tokenize and vectorize medical knowledge base
vectorizer = TfidfVectorizer()
kb_vectors = vectorizer.fit_transform(list(medical_kb.keys()))

# Store uploaded report text and summary in memory (for demo)
latest_report_text = None
latest_summary = None

def call_gemini_api(prompt, api_key=GEMINI_API_KEY):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    
    headers = {
        'Content-Type': 'application/json'
    }
    
    try:
        print(f"Sending request to Gemini API with prompt: {prompt[:100]}...")
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        # Print response for debugging
        print(f"Response status: {response.status_code}")
        print(f"Response content: {response.text[:200]}...")
        
        data = response.json()
        if 'candidates' in data and len(data['candidates']) > 0:
            return data['candidates'][0]['content']['parts'][0]['text']
        else:
            print(f"Unexpected response format: {json.dumps(data, indent=2)}")
            return "Sorry, I couldn't process that request properly."
            
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {str(e)}")
        if hasattr(e.response, 'text'):
            print(f"Error response: {e.response.text}")
        return f"Error: {str(e)}"

@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/', methods=['GET', 'POST'])
def analyze():
    if request.method == 'GET':
        return jsonify({'message': 'Welcome to the Medical Knowledge Base API!'})
    elif request.method == 'POST':
        if not request.json or 'report' not in request.json:
            return jsonify({'error': 'No report provided'}), 400
        report = request.json['report']
        tokens = word_tokenize(report)
        query_vector = vectorizer.transform([' '.join(tokens)])
        similarities = cosine_similarity(query_vector, kb_vectors)
        most_similar_index = similarities.argmax()
        most_similar_question = list(medical_kb.keys())[most_similar_index]
        response = medical_kb[most_similar_question]
        return jsonify({'response': response})

@app.route('/generate', methods=['POST'])
def generate():
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        return jsonify({'error': 'GEMINI_API_KEY environment variable not set'}), 500
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    
    if not request.json or 'text' not in request.json:
        return jsonify({'error': 'No text provided'}), 400
        
    user_input = request.json['text']
    
    payload = {
        "contents": [{
            "parts": [{"text": user_input}]
        }]
    }
    
    headers = {
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        return jsonify({'error': str(e)}), response.status_code if response else 500

@app.route('/upload', methods=['POST'])
def upload_pdf():
    global latest_report_text, latest_summary
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'File is not a PDF'}), 400

    try:
        # Extract text from PDF
        pdf_reader = PdfReader(BytesIO(file.read()))
        text = "\n".join(page.extract_text() or '' for page in pdf_reader.pages)
        latest_report_text = text
        
        # Generate summary
        summary = call_gemini_api(f"Please provide a concise summary of this medical report:\n\n{text}")
        latest_summary = summary
        
        return jsonify({'summary': summary})
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return jsonify({'error': f'Failed to process PDF: {str(e)}'}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    global latest_report_text
    
    if not latest_report_text:
        return jsonify({'error': 'No report uploaded yet'}), 400
        
    if not request.json or 'question' not in request.json:
        return jsonify({'error': 'No question provided'}), 400
        
    question = request.json['question']
    
    prompt = f"""Given this medical report:

{latest_report_text}

Please answer this question: {question}

Provide a clear and concise answer based only on the information in the report."""

    answer = call_gemini_api(prompt)
    return jsonify({'answer': answer})

if __name__ == "__main__":
    # Check if API key is set
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("WARNING: GEMINI_API_KEY environment variable is not set!")
        print("Please set it using:")
        print("Windows PowerShell: $env:GEMINI_API_KEY='your-api-key'")
        print("Linux/Mac: export GEMINI_API_KEY='your-api-key'")
    
    app.run(host='0.0.0.0', port=5000, debug=True) 