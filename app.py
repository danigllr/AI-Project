from flask import Flask, render_template, request, jsonify
import PyPDF2
import pandas as pd
import spacy
import os
import re
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import numpy as np
import time

app = Flask(__name__)

# Setze den Zeichensatz für die App
app.config['JSON_AS_ASCII'] = False  # Ermöglicht Umlaute in JSON-Antworten

# Initialisiere SpaCy
try:
    nlp = spacy.load('de_core_news_sm')
except OSError:
    print("Lade Sprachmodell...")
    os.system("python -m spacy download de_core_news_sm")
    nlp = spacy.load('de_core_news_sm')

# Modell für BERT-Embeddings
model_name = "bert-base-german-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

# Lade das Textvereinfachungsmodell (deutsches Modell)
simplifier = pipeline("text2text-generation", model="t5-base")

# Cache für BERT-Embeddings
embedding_cache = {}

def get_bert_embedding(text):
    """Extrahiert BERT-Embeddings und nutzt Cache."""
    if text in embedding_cache:
        return embedding_cache[text]
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.hidden_states[-1].mean(dim=1).squeeze().numpy()
    embedding_cache[text] = embedding
    return embedding

def normalize_vector(v):
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm

def ensure_csv_utf8(file_path):
    """Stellt sicher, dass die CSV-Datei in UTF-8 kodiert ist."""
    try:
        # Versuche, Datei mit verschiedenen Encodings zu lesen
        encodings = ['utf-8', 'latin1', 'cp1252']
        content = None
        used_encoding = None
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    content = file.read()
                    used_encoding = encoding
                    break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            print(f"Konnte die Datei {file_path} mit keinem der verfügbaren Encodings lesen.")
            return False
            
        # Wenn die Datei nicht bereits UTF-8 ist, konvertiere sie
        if used_encoding != 'utf-8':
            print(f"Konvertiere CSV von {used_encoding} nach UTF-8...")
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(content)
            print("Konvertierung abgeschlossen.")
            
        return True
    except Exception as e:
        print(f"Fehler bei der Konvertierung der CSV-Datei: {e}")
        return False

def load_medical_terms():
    """Lädt medizinische Begriffe und ihre Erklärungen aus einer CSV-Datei."""
    try:
        # Stelle sicher, dass die CSV in UTF-8 kodiert ist
        csv_file = 'medical_terms.csv'
        ensure_csv_utf8(csv_file)
        
        # Lade die CSV-Datei mit UTF-8-Encoding
        df = pd.read_csv(csv_file, encoding='utf-8', delimiter=';')
        print(df.head())
        
        # Erstelle zwei Wörterbücher: eines für Fachbegriffe und eines für vereinfachte Erklärungen
        medical_dict = {}
        simple_dict = {}
        
        # Stelle sicher, dass alle Werte Strings sind und Umlaute korrekt verarbeitet werden
        for index, row in df.iterrows():
            if pd.isna(row.iloc[0]):
                continue
                
            term = str(row.iloc[0])
            explanation = str(row.iloc[1]) if not pd.isna(row.iloc[1]) else ""
            
            if term:  # Nur Einträge mit gültigem Begriff hinzufügen
                medical_dict[term] = explanation
                
                # Wenn die Einfache_Erklärung-Spalte existiert, laden wir auch diese
                if len(row) >= 3 and not pd.isna(row.iloc[2]):
                    simple_dict[term] = str(row.iloc[2])
            
        return medical_dict, simple_dict
    except Exception as e:
        print(f"Fehler beim Laden der CSV-Datei: {e}")
        return {}, {}

# Lade medizinische Begriffe und precomputiere deren normalisierte Embeddings
medical_terms, simple_terms = load_medical_terms()
    
medical_terms_embeddings = {}
for term in medical_terms.keys():
    emb = get_bert_embedding(term)
    medical_terms_embeddings[term] = normalize_vector(emb)

def is_medical_term_bert(term, context):
    """Überprüft mittels Cosinus-Ähnlichkeit, ob ein Begriff ein medizinischer Fachbegriff ist."""
    context_text = f"{context} {term}"
    emb = get_bert_embedding(context_text)
    emb_norm = normalize_vector(emb)
    
    similarities = []
    for known_term, known_emb in medical_terms_embeddings.items():
        sim = np.dot(emb_norm, known_emb)
        similarities.append(sim)
    return np.mean(similarities) > 0.7 if similarities else False

def translate_medical_terms(text):
    """Sucht und ersetzt medizinische Begriffe im Text mit Arzt-Level-Übersetzungen."""
    doc = nlp(text)
    translated_text = text
    terms_to_check = set()
    
    # Einzelne Tokens prüfen (nur alphabetische Tokens)
    for i, token in enumerate(doc):
        if not token.text.isalpha():
            continue
        context_start = max(0, i - 5)
        context_end = min(len(doc), i + 6)
        context = " ".join([t.text for t in doc[context_start:context_end]])
        if is_medical_term_bert(token.text, context):
            terms_to_check.add(token.text)
    
    # Bigramme prüfen
    for i in range(len(doc) - 1):
        combined = f"{doc[i].text} {doc[i+1].text}"
        context_start = max(0, i - 5)
        context_end = min(len(doc), i + 7)
        context = " ".join([t.text for t in doc[context_start:context_end]])
        if is_medical_term_bert(combined, context):
            terms_to_check.add(combined)
    
    # Ersetze exakte Treffer (Wortgrenzen verwenden)
    sorted_terms = sorted(terms_to_check, key=len, reverse=True)
    for term in sorted_terms:
        if term in medical_terms:
            pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
            translated_text = pattern.sub(
                f'<span class="medical-term" title="{medical_terms[term]}">{term}</span>', 
                translated_text
            )
    
    return translated_text

def clean_html_tags(text):
    """Entfernt HTML-Tags aus dem Text."""
    return re.sub(r'<[^>]+>', '', text)

def simplify_text_for_level(text, difficulty):
    """Vereinfacht den bereits übersetzten Text für den gewählten Schwierigkeitsgrad."""
    # Entferne HTML-Tags für die Verarbeitung
    clean_text = clean_html_tags(text)
    
    # Überprüfen, ob der Text ein String ist
    if not isinstance(clean_text, str):
        clean_text = str(clean_text)
    
    # Teile den Text in Absätze
    paragraphs = clean_text.split('\n')
    simplified_paragraphs = []
    
    for paragraph in paragraphs:
        if not paragraph.strip():
            simplified_paragraphs.append('')
            continue
            
        # Teile den Text in Sätze auf
        nlp = spacy.load('de_core_news_sm')
        doc = nlp(paragraph)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        # Ersetze zuerst alle medizinischen Begriffe durch ihre einfachen Erklärungen
        simplified_sentences = []
        for sentence in sentences:
            if not sentence:  # Überspringe leere Sätze
                continue
                
            # Ersetze bekannte medizinische Begriffe mit ihren einfachen Erklärungen
            for term, explanation in simple_terms.items():
                if explanation and term in sentence:
                    # Stelle sicher, dass term und explanation Strings sind
                    term_str = str(term)
                    expl_str = str(explanation)
                    
                    # Ersetze den Begriff durch seine Erklärung ohne Klammern
                    pattern = re.compile(r'\b' + re.escape(term_str) + r'\b', re.IGNORECASE)
                    # Entscheide, ob wir den Begriff beibehalten oder komplett ersetzen
                    if len(term_str) <= 4 and term_str.isupper():  # Für Abkürzungen
                        replacement = f"{expl_str} ({term_str})"
                    else:
                        replacement = expl_str
                    sentence = pattern.sub(replacement, sentence)
            
            # Korrigiere häufige medizinische Werte und Einheiten
            sentence = re.sub(r'(\d+)\s*mg/D', r'\1 mg/dl (Milligramm pro Deziliter)', sentence)
            sentence = re.sub(r'(\d+)\s*-\s*H\s*-\s*EKG', r'\1-Stunden-EKG', sentence)
            sentence = re.sub(r'(\d+)\s*mm/H', r'\1 mm/h (Millimeter pro Stunde)', sentence)
            sentence = re.sub(r'(\d+)\s*U/l', r'\1 Einheiten pro Liter', sentence)
            
            # Korrigiere BMI-Werte
            sentence = re.sub(r'BMI.*?(\d+[.,]\d+)\s*kg/m²', 
                             r'Body-Mass-Index liegt bei \1 kg/m² (Übergewicht)', sentence)
            
            simplified_sentences.append(sentence)
        
        # Vereinfache spezifische medizinische Ausdrücke im gesamten Text
        simplified_text = " ".join(simplified_sentences)
        
        # Ersetze spezifische medizinische Ausdrücke
        replacements = {
            "kardiologischer Abklärung": "weiterer Untersuchung durch einen Herzspezialisten",
            "AV-Block I. Grades": "leichte Störung der Herzreizleitung",
            "Plaques im Rahmen der AVK": "Verengungen in den Blutgefäßen",
            "Thromboseprophylaxe": "Vorbeugung von Blutgerinnseln",
            "Anschlussheilbehandlung": "Anschließende Heilbehandlung",
            "engmaschige Betreuung": "regelmäßige ärztliche Kontrolle",
            "postprandial": "nach dem Essen",
            "nüchtern": "auf leeren Magen",
            "Body-Mass-Index": "Verhältnis von Körpergewicht zu Körpergröße",
            "Adipositas Grad I": "leichtes Übergewicht",
            "Broteinheiten": "Maßeinheit für Kohlenhydrate bei Diabetes",
            "erhöhte Alaninaminotransferase": "erhöhte Leberwerte",
            "erhöhte Aspartataminotransferase": "erhöhte Leberwerte",
            "beschleunigte Blutkörperchensenkungsgeschwindigkeit": "erhöhte Entzündungswerte im Blut"
        }
        
        for medical_term, simple_explanation in replacements.items():
            simplified_text = simplified_text.replace(medical_term, simple_explanation)
        
        # Korrigiere doppelte Klammern und Erklärungen
        simplified_text = re.sub(r'\(([^()]*)\)\s*\(\1\)', r'(\1)', simplified_text)
        simplified_text = re.sub(r'\(([^()]*)\)\s*\(([^()]*)\)', r'(\1, \2)', simplified_text)
        
        # Entferne "nan" Artefakte
        simplified_text = re.sub(r'\(nan\)', '', simplified_text)
        simplified_text = re.sub(r' nan ', ' ', simplified_text)
        
        # Allgemeine Nachbearbeitung
        simplified_text = re.sub(r'(?<!\.)\.\.(?!\.)', '.', simplified_text)  # Doppelpunkte korrigieren
        simplified_text = re.sub(r'\s+', ' ', simplified_text)  # Mehrfach-Leerzeichen entfernen
        simplified_text = re.sub(r'\s+([.,;:])', r'\1', simplified_text)  # Leerzeichen vor Satzzeichen entfernen
        
        # Füge den vereinfachten Absatz hinzu
        simplified_paragraphs.append(simplified_text)
    
    # Füge alle Absätze wieder zusammen
    result = '\n'.join(simplified_paragraphs)
    
    # Entferne Reste von KI-generierten Prompts
    result = re.sub(r'(Diesen|den|folgenden) medizinischen (Fach)?text (erkläre ich in|in) (sehr )?einfachen Worten( ohne Fachbegriffe)?:?\s*', '', result)
    result = re.sub(r'Sehr einfache Worte ohne Fachbegriffe:?\s*', '', result)
    result = re.sub(r'Dieser Fachtext ist in folgenden Worten ohne Fachbegriffe wiedergegeben:?\s*', '', result)
    
    return result

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        return f"Fehler beim Lesen der PDF-Datei: {e}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    if 'file' not in request.files:
        return jsonify({'error': 'Keine Datei hochgeladen'}), 400
        
    file = request.files['file']
    difficulty = request.form.get('difficulty', '0')  # '0' für Arzt, '1' für Neuling
    
    if file.filename == '':
        return jsonify({'error': 'Keine Datei ausgewählt'}), 400
        
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Nur PDF-Dateien sind erlaubt'}), 400
    
    try:
        # 1. Extrahiere Text aus PDF
        text = extract_text_from_pdf(file)
        if not text:
            return jsonify({'error': 'Konnte keinen Text aus der PDF extrahieren'}), 400
            
        # 2. Übersetze medizinische Begriffe
        translated_text = translate_medical_terms(text)
        
        # 3. Vereinfache den bereits übersetzten Text für Neuling-Level
        simple_text = ""
        if difficulty == '1':  # Neuling-Level
            simple_text = simplify_text_for_level(translated_text, difficulty)
            
            # Füge HTML-Tags für einfache Erklärungen hinzu
            for term in medical_terms:
                # Stelle sicher, dass term ein String ist
                term_str = str(term)
                if term_str in simple_text:
                    pattern = re.compile(r'\b' + re.escape(term_str) + r'\b', re.IGNORECASE)
                    if term_str in simple_terms and simple_terms[term_str]:
                        explanation_str = str(simple_terms[term_str])
                        simple_text = pattern.sub(
                            f'<span class="medical-term" title="{explanation_str}">{term_str}</span>', 
                            simple_text
                        )
        
        return jsonify({
            'original': text,
            'translated_text': translated_text if difficulty == '0' else simple_text
        })
        
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print(f"Fehler bei der Verarbeitung: {str(e)}\n{traceback_str}")
        return jsonify({'error': f'Fehler bei der Verarbeitung: {str(e)}'}), 500

if __name__ == '__main__':
    # Lade medizinische Begriffe und precomputiere deren normalisierte Embeddings
    medical_terms, simple_terms = load_medical_terms()
    
    medical_terms_embeddings = {}
    for term in medical_terms.keys():
        emb = get_bert_embedding(term)
        medical_terms_embeddings[term] = normalize_vector(emb)
        
    app.run(debug=True)