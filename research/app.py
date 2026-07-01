import os
import re
from collections import Counter
import pandas as pd
import spacy
from flask import Flask, render_template, request, send_file, redirect, url_for, flash
import pdfplumber
import unicodedata
from deep_translator import GoogleTranslator
import ollama
from PyPDF2 import PdfReader
from werkzeug.utils import secure_filename
import requests

# --- Flask App Setup ---
app = Flask(__name__)
app.secret_key = 'your_secret_key_here_change_this_in_production'

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
CSV_FILE = "data.csv"
OUTPUT_FILE = "output.csv"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Lazy Loading for SpaCy Models ---
nlp_models = {}

def load_spacy_model(model_name):
    """Loads a spaCy model, downloading it if necessary."""
    if model_name not in nlp_models:
        try:
            nlp_models[model_name] = spacy.load(model_name)
        except OSError:
            print(f"Downloading spaCy model '{model_name}'...")
            from spacy.cli import download
            download(model_name)
            nlp_models[model_name] = spacy.load(model_name)
    return nlp_models[model_name]

# --- Core Logic Class ---
class ResearchProcessor:
    def extract_text_from_pdf(self, pdf_file):
        text = ""
        pdf_file.seek(0)
        try:
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text(x_tolerance=1)
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"pdfplumber failed: {e}. Falling back to PyPDF2.")
            try:
                pdf_file.seek(0)
                pdf_reader = PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            except Exception as e_pypdf:
                print(f"PyPDF2 also failed: {e_pypdf}")
                return None
        return text.strip() if text.strip() else None

    def extract_abstract(self, text, language_key):
        if not text: return None
        abstract_patterns = {
            'english': r'abstract[:\s]*(.+?)(?:keywords?|introduction|1\.|references|\n\n)',
            'german': r'zusammenfassung[:\s]*(.+?)(?:schlüsselwörter|einleitung|1\.|literatur|\n\n)',
            'spanish': r'resumen[:\s]*(.+?)(?:palabras clave|introducción|1\.|referencias|\n\n)',
            'japanese': r'要約[:\s]*(.+?)(?:キーワード|はじめに|1\.|参考文献|\n\n)',
            'chinese': r'摘要[:\s]*(.+?)(?:关键词|引言|1\.|参考文献|\n\n)'
        }
        pattern = abstract_patterns.get(language_key, abstract_patterns['english'])
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return ' '.join(match.group(1).strip().split())
        
        print("Regex for abstract failed, using NLP fallback.")
        nlp = load_spacy_model('en_core_web_sm')
        paragraphs = text.split('\n\n')
        best_paragraph = max((p for p in paragraphs if 100 < len(p) < 3000), 
                             key=lambda p: len([token for token in nlp(p) if token.pos_ in ['NOUN', 'VERB', 'ADJ']]), 
                             default="")
        return best_paragraph if best_paragraph else text[:2000]

    def translate_text(self, text, target_lang, source_lang='auto'):
        if not text: return ""
        try:
            max_length, translated_chunks = 4500, []
            if len(text) <= max_length:
                return GoogleTranslator(source=source_lang, target=target_lang).translate(text)
            for i in range(0, len(text), max_length):
                chunk = text[i:i + max_length]
                translated_chunks.append(GoogleTranslator(source=source_lang, target=target_lang).translate(chunk))
            return ''.join(translated_chunks)
        except Exception as e:
            print(f"Translation error: {e}")
            return text

    def summarize_with_mistral(self, text):
        if not text: return "Cannot summarize empty text."
        try:
            prompt = f"Provide a comprehensive summary of the following research paper abstract...\n\nAbstract: {text}\n\nSummary:"
            response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': prompt}])
            return response['message']['content']
        except Exception as e:
            print(f"Mistral summarization error: {e}")
            return f"Error connecting to Ollama. Please ensure it's running and 'mistral' is installed. Error: {str(e)}"

    def match_keywords_spacy(self, abstract):
        try:
            if not os.path.exists(CSV_FILE):
                return []
            keywords_df = pd.read_csv(CSV_FILE, header=None, names=["keyword"])
            keywords = [str(kw).lower().strip() for kw in keywords_df["keyword"] if pd.notna(kw)]
            nlp = load_spacy_model('en_core_web_sm')
            doc = nlp(abstract.lower())
            token_freq = Counter(token.lemma_ for token in doc if not token.is_stop and token.is_alpha)
            matched = [(kw, token_freq.get(kw, 0)) for kw in keywords if token_freq.get(kw, 0) > 0]
            matched.sort(key=lambda x: x[1], reverse=True)
            return matched
        except Exception as e:
            print(f"Error matching keywords: {e}")
            return []

    def find_papers_by_keyword(self, keyword, limit=3):
        search_query = keyword.replace(" ", "+")
        api_url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={search_query}&limit={limit}&fields=title,authors,year,openAccessPdf,venue,url,abstract"
        try:
            response = requests.get(api_url, timeout=15)
            response.raise_for_status()
            return response.json().get('data', [])
        except requests.exceptions.RequestException as e:
            print(f"API request failed for '{keyword}': {e}")
            return []

    def generate_highlighted_html(self, text, keywords_to_highlight):
        if not text: return ""
        sorted_keywords = sorted(keywords_to_highlight, key=lambda x: len(x[1]), reverse=True)
        for en_kw, translated_kw in sorted_keywords:
            try:
                pattern = re.compile(r'\b(' + re.escape(translated_kw) + r')\b', re.IGNORECASE)
                replacement = f'<mark data-bs-toggle="tooltip" title="Source: {en_kw}">\\1</mark>'
                text = pattern.sub(replacement, text)
            except re.error:
                continue
        return text

# --- Main Processing Functions ---
def process_summarizer_paper(pdf_file, selected_language):
    processor = ResearchProcessor()
    text = processor.extract_text_from_pdf(pdf_file)
    if not text: return None, "Failed to extract text from the PDF.", None
    abstract = processor.extract_abstract(text, selected_language)
    if not abstract: return None, "Could not automatically find an abstract.", None
    english_abstract = abstract if selected_language == 'english' else processor.translate_text(abstract, target_lang='en')
    summary = processor.summarize_with_mistral(english_abstract)
    return abstract, english_abstract, summary

def process_retrieval_pdf(pdf_file):
    processor = ResearchProcessor()
    text = processor.extract_text_from_pdf(pdf_file)
    if not text: return "Failed to extract text from PDF.", [], {}
    
    abstract = processor.extract_abstract(text, 'english')
    if not abstract: return "Could not find abstract.", [], {}

    matched_keywords = processor.match_keywords_spacy(abstract)
    keyword_data, aligned_keywords = [], {}
    target_langs = {"japanese": "ja", "german": "de", "chinese": "zh-CN", "spanish": "es"}

    for kw, count in matched_keywords:
        row = {"keyword": kw, "frequency": count}
        for lang_name, lang_code in target_langs.items():
            translated = processor.translate_text(kw, lang_code)
            row[lang_name] = translated
            aligned_keywords.setdefault(lang_name, []).append((kw, translated))
        keyword_data.append(row)

    pd.DataFrame(keyword_data).to_csv(OUTPUT_FILE, index=False)
    
    similar_papers = {}
    for lang_name, kw_pairs in aligned_keywords.items():
        top_kw_pairs = kw_pairs[:5]
        papers_found = [p for _, trans_kw in top_kw_pairs for p in processor.find_papers_by_keyword(trans_kw)]
        unique_papers = {p['paperId']: p for p in papers_found if p.get('paperId')}
        
        for paper_id, paper in unique_papers.items():
            # Create a list of keywords that could have found this paper
            possible_kws = aligned_keywords.get(lang_name, [])
            paper['highlighted_title'] = processor.generate_highlighted_html(paper.get('title'), possible_kws)
            paper['highlighted_abstract'] = processor.generate_highlighted_html(paper.get('abstract'), possible_kws)

        similar_papers[lang_name] = list(unique_papers.values())
        
    return abstract, keyword_data, similar_papers

# --- Flask Routes ---
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/retrieval', methods=['GET', 'POST'])
def retrieval():
    if request.method == 'POST':
        if 'pdf' not in request.files or not request.files['pdf'].filename:
            flash("Please select a PDF file.", "error")
            return redirect(url_for('retrieval'))
        
        pdf_file = request.files['pdf']
        filename = secure_filename(pdf_file.filename)
        abstract, keyword_data, similar_papers = process_retrieval_pdf(pdf_file)
        
        return render_template("index.html",
                               abstract=abstract,
                               data=keyword_data,
                               similar_papers=similar_papers,
                               uploaded=True,
                               filename=filename)
    return render_template('index.html', uploaded=False)

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get("query")
    if not query or not query.strip():
        flash("Please enter a search term.", "error")
        return redirect(url_for('retrieval'))

    processor = ResearchProcessor()
    search_results = processor.find_papers_by_keyword(query, limit=15)
    
    return render_template("index.html",
                           query=query,
                           search_results=search_results,
                           uploaded=False)

@app.route('/summarizer', methods=['GET', 'POST'])
def summarizer():
    if request.method == 'POST':
        if 'pdf' not in request.files or not request.files['pdf'].filename:
            flash('No file selected. Please choose a PDF.', 'error')
            return redirect(request.url)

        file = request.files['pdf']
        language = request.form.get('language')

        if not language:
            flash('Please select the language of the paper.', 'error')
            return redirect(request.url)

        if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
            try:
                original_abstract, english_abstract, summary = process_summarizer_paper(file, language)
                if original_abstract is None:
                    flash(english_abstract, 'error')
                    return redirect(request.url)
                
                return render_template('summarizer.html', 
                                       uploaded=True,
                                       language=language,
                                       filename=secure_filename(file.filename),
                                       original_abstract=original_abstract,
                                       english_abstract=english_abstract,
                                       summary=summary)
            except Exception as e:
                flash(f'An unexpected error occurred: {str(e)}', 'error')
                return redirect(request.url)
        else:
            flash('Invalid file type. Only PDF files are allowed.', 'error')
            return redirect(request.url)
            
    return render_template('summarizer.html', uploaded=False)

@app.route('/reset')
def reset():
    return redirect(url_for('summarizer'))

@app.route('/download')
def download_file():
    if os.path.exists(OUTPUT_FILE):
        return send_file(OUTPUT_FILE, as_attachment=True)
    flash("No output file available to download.", "error")
    return redirect(url_for('retrieval'))

@app.errorhandler(413)
def request_entity_too_large(e):
    flash('File is too large. Max size is 16MB.', 'error')
    return redirect(url_for('summarizer'))

# --- Run App ---
if __name__ == '__main__':
    try:
        models = ollama.list()
        model_names = [model['name'] for model in models.get('models', [])]
        if not any('mistral' in name for name in model_names):
             print("\nWARNING: Mistral model not found. Please run: ollama pull mistral\n")
    except Exception as e:
        print(f"\nWARNING: Could not check Ollama models: {e}")
        print("Please ensure Ollama is installed and running.\n")
        
    app.run(debug=True)

