import numpy as np
import fitz  # PyMuPDF for PDF extraction (install via pip install PyMuPDF)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template

app = Flask(__name__)

def preprocess(text):
    """Preprocess text by converting to lowercase and removing punctuation."""
    return text.lower().replace('\n', ' ').replace('\r', '').translate(str.maketrans('', '', '!?.,;:'))

def extract_text(file):
    """Extract text from a .txt or .pdf file."""
    if file.filename.lower().endswith('.txt'):
        return file.read().decode('utf-8')
    elif file.filename.lower().endswith('.pdf'):
        try:
            file_bytes = file.read()
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text("text")
            return text
        except Exception as e:
            raise Exception("Error processing PDF file: " + str(e))
    else:
        raise Exception("Unsupported file type")

def check_plagiarism():
    """Check plagiarism between two uploaded files (.txt or .pdf)."""
    if 'file1' not in request.files or 'file2' not in request.files:
        return "Error: Please upload both files"
    
    file1 = request.files['file1']
    file2 = request.files['file2']
    
    if file1.filename == '' or file2.filename == '':
        return "Error: One or both files are empty"
    
    allowed_extensions = ('.txt', '.pdf')
    if not file1.filename.lower().endswith(allowed_extensions) or not file2.filename.lower().endswith(allowed_extensions):
        return "Error: Please upload .txt or .pdf files only"
    
    try:
        text1 = extract_text(file1)
        text2 = extract_text(file2)
        
        if not text1.strip() or not text2.strip():
            return "Error: One or both files contain no text"
        
        docs = [preprocess(text1), preprocess(text2)]
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(docs)
        
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        similarity_percent = similarity * 100
        return f"Similarity Score: {similarity_percent:.2f}%", similarity_percent
    
    except UnicodeDecodeError:
        return "Error: Files must be UTF-8 encoded text"
    except Exception as e:
        return f"Error: An unexpected error occurred - {str(e)}"

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    similarity = 0
    if request.method == 'POST':
        result_data = check_plagiarism()
        if isinstance(result_data, tuple):
            result, similarity = result_data
        else:
            result = result_data
    return render_template('index.html', result=result, similarity=similarity)

if __name__ == '__main__':
    app.run(debug=True)
