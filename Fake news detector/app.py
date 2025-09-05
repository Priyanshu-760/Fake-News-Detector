from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app = Flask(__name__)
CORS(app, resources={
    r"/analyze": {
        "origins": ["http://127.0.0.1:5500"],
        "methods": ["POST"],
        "allow_headers": ["Content-Type", "Accept"]
    }
})

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        # Your analysis logic here
        return jsonify({
            'label': 'Real',  # or 'Fake' based on your analysis
            'confidence': 0.95,
            'sentiment': 'Positive'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into text
    return ' '.join(tokens)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # For demonstration, using a simple rule-based approach
    # In production, you would load your trained model here
    suspicious_words = ['hoax', 'conspiracy', 'shocking', 'you won\'t believe',
                       'secret', 'miracle', 'cure', 'clickbait']
    
    # Count suspicious words
    count = sum(1 for word in suspicious_words if word in processed_text.lower())
    
    # Calculate confidence score
    confidence = min(count / len(suspicious_words), 1.0)
    
    # Determine if fake based on confidence threshold
    is_fake = confidence > 0.3
    
    # Simple sentiment analysis
    positive_words = ['good', 'great', 'excellent', 'true', 'fact']
    negative_words = ['bad', 'false', 'fake', 'hoax', 'conspiracy']
    
    pos_count = sum(1 for word in positive_words if word in processed_text.lower())
    neg_count = sum(1 for word in negative_words if word in processed_text.lower())
    
    sentiment = 'Neutral'
    if pos_count > neg_count:
        sentiment = 'Positive'
    elif neg_count > pos_count:
        sentiment = 'Negative'
    
    return jsonify({
        'label': 'FAKE' if is_fake else 'REAL',
        'confidence': confidence,
        'sentiment': sentiment
    })

if __name__ == '__main__':
    app.run(debug=True)
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into text
    return ' '.join(tokens)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # For demonstration, using a simple rule-based approach
    # In production, you would load your trained model here
    suspicious_words = ['hoax', 'conspiracy', 'shocking', 'you won\'t believe',
                       'secret', 'miracle', 'cure', 'clickbait']
    
    # Count suspicious words
    count = sum(1 for word in suspicious_words if word in processed_text.lower())
    
    # Calculate confidence score
    confidence = min(count / len(suspicious_words), 1.0)
    
    # Determine if fake based on confidence threshold
    is_fake = confidence > 0.3
    
    # Simple sentiment analysis
    positive_words = ['good', 'great', 'excellent', 'true', 'fact']
    negative_words = ['bad', 'false', 'fake', 'hoax', 'conspiracy']
    
    pos_count = sum(1 for word in positive_words if word in processed_text.lower())
    neg_count = sum(1 for word in negative_words if word in processed_text.lower())
    
    sentiment = 'Neutral'
    if pos_count > neg_count:
        sentiment = 'Positive'
    elif neg_count > pos_count:
        sentiment = 'Negative'
    
    return jsonify({
        'label': 'FAKE' if is_fake else 'REAL',
        'confidence': confidence,
        'sentiment': sentiment
    })

if __name__ == '__main__':
    app.run(debug=True)
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into text
    return ' '.join(tokens)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # For demonstration, using a simple rule-based approach
    # In production, you would load your trained model here
    suspicious_words = ['hoax', 'conspiracy', 'shocking', 'you won\'t believe',
                       'secret', 'miracle', 'cure', 'clickbait']
    
    # Count suspicious words
    count = sum(1 for word in suspicious_words if word in processed_text.lower())
    
    # Calculate confidence score
    confidence = min(count / len(suspicious_words), 1.0)
    
    # Determine if fake based on confidence threshold
    is_fake = confidence > 0.3
    
    # Simple sentiment analysis
    positive_words = ['good', 'great', 'excellent', 'true', 'fact']
    negative_words = ['bad', 'false', 'fake', 'hoax', 'conspiracy']
    
    pos_count = sum(1 for word in positive_words if word in processed_text.lower())
    neg_count = sum(1 for word in negative_words if word in processed_text.lower())
    
    sentiment = 'Neutral'
    if pos_count > neg_count:
        sentiment = 'Positive'
    elif neg_count > pos_count:
        sentiment = 'Negative'
    
    return jsonify({
        'label': 'FAKE' if is_fake else 'REAL',
        'confidence': confidence,
        'sentiment': sentiment
    })

if __name__ == '__main__':
    app.run(debug=True)
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into text
    return ' '.join(tokens)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # For demonstration, using a simple rule-based approach
    # In production, you would load your trained model here
    suspicious_words = ['hoax', 'conspiracy', 'shocking', 'you won\'t believe',
                       'secret', 'miracle', 'cure', 'clickbait']
    
    # Count suspicious words
    count = sum(1 for word in suspicious_words if word in processed_text.lower())
    
    # Calculate confidence score
    confidence = min(count / len(suspicious_words), 1.0)
    
    # Determine if fake based on confidence threshold
    is_fake = confidence > 0.3
    
    # Simple sentiment analysis
    positive_words = ['good', 'great', 'excellent', 'true', 'fact']
    negative_words = ['bad', 'false', 'fake', 'hoax', 'conspiracy']
    
    pos_count = sum(1 for word in positive_words if word in processed_text.lower())
    neg_count = sum(1 for word in negative_words if word in processed_text.lower())
    
    sentiment = 'Neutral'
    if pos_count > neg_count:
        sentiment = 'Positive'
    elif neg_count > pos_count:
        sentiment = 'Negative'
    
    return jsonify({
        'label': 'FAKE' if is_fake else 'REAL',
        'confidence': confidence,
        'sentiment': sentiment
    })

if __name__ == '__main__':
    app.run(debug=True)
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into text
    return ' '.join(tokens)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # For demonstration, using a simple rule-based approach
    # In production, you would load your trained model here
    suspicious_words = ['hoax', 'conspiracy', 'shocking', 'you won\'t believe',
                       'secret', 'miracle', 'cure', 'clickbait']
    
    # Count suspicious words
    count = sum(1 for word in suspicious_words if word in processed_text.lower())
    
    # Calculate confidence score
    confidence = min(count / len(suspicious_words), 1.0)
    
    # Determine if fake based on confidence threshold
    is_fake = confidence > 0.3
    
    # Simple sentiment analysis
    positive_words = ['good', 'great', 'excellent', 'true', 'fact']
    negative_words = ['bad', 'false', 'fake', 'hoax', 'conspiracy']
    
    pos_count = sum(1 for word in positive_words if word in processed_text.lower())
    neg_count = sum(1 for word in negative_words if word in processed_text.lower())
    
    sentiment = 'Neutral'
    if pos_count > neg_count:
        sentiment = 'Positive'
    elif neg_count > pos_count:
        sentiment = 'Negative'
    
    return jsonify({
        'label': 'FAKE' if is_fake else 'REAL',
        'confidence': confidence,
        'sentiment': sentiment
    })

if __name__ == '__main__':
    app.run(debug=True)
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into text
    return ' '.join(tokens)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # For demonstration, using a simple rule-based approach
    # In production, you would load your trained model here
    suspicious_words = ['hoax', 'conspiracy', 'shocking', 'you won\'t believe',
                       'secret', 'miracle', 'cure', 'clickbait']
    
    # Count suspicious words
    count = sum(1 for word in suspicious_words if word in processed_text.lower())
    
    # Calculate confidence score
    confidence = min(count / len(suspicious_words), 1.0)
    
    # Determine if fake based on confidence threshold
    is_fake = confidence > 0.3
    
    # Simple sentiment analysis
    positive_words = ['good', 'great', 'excellent', 'true', 'fact']
    negative_words = ['bad', 'false', 'fake', 'hoax', 'conspiracy']
    
    pos_count = sum(1 for word in positive_words if word in processed_text.lower())
    neg_count = sum(1 for word in negative_words if word in processed_text.lower())
    
    sentiment = 'Neutral'
    if pos_count > neg_count:
        sentiment = 'Positive'
    elif neg_count > pos_count:
        sentiment = 'Negative'
    
    return jsonify({
        'label': 'FAKE' if is_fake else 'REAL',
        'confidence': confidence,
        'sentiment': sentiment
    })

if __name__ == '__main__':
    app.run(debug=True)
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into text
    return ' '.join(tokens)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # For demonstration, using a simple rule-based approach
    # In production, you would load your trained model here
    suspicious_words = ['hoax', 'conspiracy', 'shocking', 'you won\'t believe',
                       'secret', 'miracle', 'cure', 'clickbait']
    
    # Count suspicious words
    count = sum(1 for word in suspicious_words if word in processed_text.lower())
    
    # Calculate confidence score
    confidence = min(count / len(suspicious_words), 1.0)
    
    # Determine if fake based on confidence threshold
    is_fake = confidence > 0.3
    
    # Simple sentiment analysis
    positive_words = ['good', 'great', 'excellent', 'true', 'fact']
    negative_words = ['bad', 'false', 'fake', 'hoax', 'conspiracy']
    
    pos_count = sum(1 for word in positive_words if word in processed_text.lower())
    neg_count = sum(1 for word in negative_words if word in processed_text.lower())
    
    sentiment = 'Neutral'
    if pos_count > neg_count:
        sentiment = 'Positive'
    elif neg_count > pos_count:
        sentiment = 'Negative'
    
    return jsonify({
        'label': 'FAKE' if is_fake else 'REAL',
        'confidence': confidence,
        'sentiment': sentiment
    })

if __name__ == '__main__':
    app.run(debug=True)
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into text
    return ' '.join(tokens)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # For demonstration, using a simple rule-based approach
    # In production, you would load your trained model here
    suspicious_words = ['hoax', 'conspiracy', 'shocking', 'you won\'t believe',
                       'secret', 'miracle', 'cure', 'clickbait']
    
    # Count suspicious words
    count = sum(1 for word in suspicious_words if word in processed_text.lower())
    
    # Calculate confidence score
    confidence = min(count / len(suspicious_words), 1.0)
    
    # Determine if fake based on confidence threshold
    is_fake = confidence > 0.3
    
    # Simple sentiment analysis
    positive_words = ['good', 'great', 'excellent', 'true', 'fact']
    negative_words = ['bad', 'false', 'fake', 'hoax', 'conspiracy']
    
    pos_count = sum(1 for word in positive_words if word in processed_text.lower())
    neg_count = sum(1 for word in negative_words if word in processed_text.lower())
    
    sentiment = 'Neutral'
    if pos_count > neg_count:
        sentiment = 'Positive'
    elif neg_count > pos_count:
        sentiment = 'Negative'
    
    return jsonify({
        'label': 'FAKE' if is_fake else 'REAL',
        'confidence': confidence,
        'sentiment': sentiment
    })

if __name__ == '__main__':
    app.run(debug=True)
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into text
    return ' '.join(tokens)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # For demonstration, using a simple rule-based approach
    # In production, you would load your trained model here
    suspicious_words = ['hoax', 'conspiracy', 'shocking', 'you won\'t believe',
                       'secret', 'miracle', 'cure', 'clickbait']
    
    # Count suspicious words
    count = sum(1 for word in suspicious_words if word in processed_text.lower())
    
    # Calculate confidence score
    confidence = min(count / len(suspicious_words), 1.0)
    
    # Determine if fake based on confidence threshold
    is_fake = confidence > 0.3
    
    # Simple sentiment analysis
    positive_words = ['good', 'great', 'excellent', 'true', 'fact']
    negative_words = ['bad', 'false', 'fake', 'hoax', 'conspiracy']
    
    pos_count = sum(1 for word in positive_words if word in processed_text.lower())
    neg_count = sum(1 for word in negative_words if word in processed_text.lower())
    
    sentiment = 'Neutral'
    if pos_count > neg_count:
        sentiment = 'Positive'
    elif neg_count > pos_count:
        sentiment = 'Negative'
    
    return jsonify({
        'label': 'FAKE' if is_fake else 'REAL',
        'confidence': confidence,
        'sentiment': sentiment
    })

if __name__ == '__main__':
    app.run(debug=True)
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into text
    return ' '.join(tokens)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # For demonstration, using a simple rule-based approach
    # In production, you would load your trained model here
    suspicious_words = ['hoax', 'conspiracy', 'shocking', 'you won\'t believe',
                       'secret', 'miracle', 'cure', 'clickbait']
    
    # Count suspicious words
    count = sum(1 for word in suspicious_words if word in processed_text.lower())
    
    # Calculate confidence score
    confidence = min(count / len(suspicious_words), 1.0)
    
    # Determine if fake based on confidence threshold
    is_fake = confidence > 0.3
    
    # Simple sentiment analysis
    positive_words = ['good', 'great', 'excellent', 'true', 'fact']
    negative_words = ['bad', 'false', 'fake', 'hoax', 'conspiracy']
    
    pos_count = sum(1 for word in positive_words if word in processed_text.lower())
    neg_count = sum(1 for word in negative_words if word in processed_text.lower())
    
    sentiment = 'Neutral'
    if pos_count > neg_count:
        sentiment = 'Positive'
    elif neg_count > pos_count:
        sentiment = 'Negative'
    
    return jsonify({
        'label': 'FAKE' if is_fake else 'REAL',
        'confidence': confidence,
        'sentiment': sentiment
    })

if __name__ == '__main__':
    app.run(debug=True)
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into text
    return ' '.join(tokens)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # For demonstration, using a simple rule-based approach
    # In production, you would load your trained model here
    suspicious_words = ['hoax', 'conspiracy', 'shocking', 'you won\'t believe',
                       'secret', 'miracle', 'cure', 'clickbait']
    
    # Count suspicious words
    count = sum(1 for word in suspicious_words if word in processed_text.lower())
    
    # Calculate confidence score
    confidence = min(count / len(suspicious_words), 1.0)
    
    # Determine if fake based on confidence threshold
    is_fake = confidence > 0.3
    
    # Simple sentiment analysis
    positive_words = ['good', 'great', 'excellent', 'true', 'fact']
    negative_words = ['bad', 'false', 'fake', 'hoax', 'conspiracy']
    
    pos_count = sum(1 for word in positive_words if word in processed_text.lower())
    neg_count = sum(1 for word in negative_words if word in processed_text.lower())
    
    sentiment = 'Neutral'
    if pos_count > neg_count:
        sentiment = 'Positive'
    elif neg_count > pos_count:
        sentiment = 'Negative'
    
    return jsonify({
        'label': 'FAKE' if is_fake else 'REAL',
        'confidence': confidence,
        'sentiment': sentiment
    })

if __name__ == '__main__':
    app.run(debug=True)
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into text
    return ' '.join(tokens)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # For demonstration, using a simple rule-based approach
    # In production, you would load your trained model here
    suspicious_words = ['hoax', 'conspiracy', 'shocking', 'you won\'t believe',
                       'secret', 'miracle', 'cure', 'clickbait']
    
    # Count suspicious words
    count = sum(1 for word in suspicious_words if word in processed_text.lower())
    
    # Calculate confidence score
    confidence = min(count / len(suspicious_words), 1.0)
    
    # Determine if fake based on confidence threshold
    is_fake = confidence > 0.3
    
    # Simple sentiment analysis
    positive_words = ['good', 'great', 'excellent', 'true', 'fact']
    negative_words = ['bad', 'false', 'fake', 'hoax', 'conspiracy']
    
    pos_count = sum(1 for word in positive_words if word in processed_text.lower())
    neg_count = sum(1 for word in negative_words if word in processed_text.lower())
    
    sentiment = 'Neutral'
    if pos_count > neg_count:
        sentiment = 'Positive'
    elif neg_count > pos_count:
        sentiment = 'Negative'
    
    return jsonify({
        'label': 'FAKE' if is_fake else 'REAL',
        'confidence': confidence,
        'sentiment': sentiment
    })

if __name__ == '__main__':
    app.run(debug=True)
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into text
    return ' '.join(tokens)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # For demonstration, using a simple rule-based approach
    # In production, you would load your trained model here
    suspicious_words = ['hoax', 'conspiracy', 'shocking', 'you won\'t believe',
                       'secret', 'miracle', 'cure', 'clickbait']
    
    # Count suspicious words
    count = sum(1 for word in suspicious_words if word in processed_text.lower())
    
    # Calculate confidence score
    confidence = min(count / len(suspicious_words), 1.0)
    
    # Determine if fake based on confidence threshold
    is_fake = confidence > 0.3
    
    # Simple sentiment analysis
    positive_words = ['good', 'great', 'excellent', 'true', 'fact']
    negative_words = ['bad', 'false', 'fake', 'hoax', 'conspiracy']
    
    pos_count = sum(1 for word in positive_words if word in processed_text.lower())
    neg_count = sum(1 for word in negative_words if word in processed_text.lower())
    
    sentiment = 'Neutral'
    if pos_count > neg_count:
        sentiment = 'Positive'
    elif neg_count > pos_count:
        sentiment = 'Negative'
    
    return jsonify({
        'label': 'FAKE' if is_fake else 'REAL',
        'confidence': confidence,
        'sentiment': sentiment
    })

if __name__ == '__main__':
    app.run(debug=True)
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into text
    return ' '.join(tokens)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # For demonstration, using a simple rule-based approach
    # In production, you would load your trained model here
    suspicious_words = ['hoax', 'conspiracy', 'shocking', 'you won\'t believe',
                       'secret', 'miracle', 'cure', 'clickbait']
    
    # Count suspicious words
    count = sum(1 for word in suspicious_words if word in processed_text.lower())
    
    # Calculate confidence score
    confidence = min(count / len(suspicious_words), 1.0)
    
    # Determine if fake based on confidence threshold
    is_fake = confidence > 0.3
    
    # Simple sentiment analysis
    positive_words = ['good', 'great', 'excellent', 'true', 'fact']
    negative_words = ['bad', 'false', 'fake', 'hoax', 'conspiracy']
    
    pos_count = sum(1 for word in positive_words if word in processed_text.lower())
    neg_count = sum(1 for word in negative_words if word in processed_text.lower())
    
    sentiment = 'Neutral'
    if pos_count > neg_count:
        sentiment = 'Positive'
    elif neg_count > pos_count:
        sentiment = 'Negative'
    
    return jsonify({
        'label': 'FAKE' if is_fake else 'REAL',
        'confidence': confidence,
        'sentiment': sentiment
    })

if __name__ == '__main__':
    app.run(debug=True)
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into text
    return ' '.join(tokens)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # For demonstration, using a simple rule-based approach
    # In production, you would load your trained model here
    suspicious_words = ['hoax', 'conspiracy', 'shocking', 'you won\'t believe',
                       'secret', 'miracle', 'cure', 'clickbait']
    
    # Count suspicious words
    count = sum(1 for word in suspicious_words if word in processed_text.lower())
    
    # Calculate confidence score
    confidence = min(count / len(suspicious_words), 1.0)
    
    # Determine if fake based on confidence threshold
    is_fake = confidence > 0.3
    
    # Simple sentiment analysis
    positive_words = ['good', 'great', 'excellent', 'true', 'fact']
    negative_words = ['bad', 'false', 'fake', 'hoax', 'conspiracy']
    
    pos_count = sum(1 for word in positive_words if word in processed_text.lower())
    neg_count = sum(1 for word in negative_words if word in processed_text.lower())
    
    sentiment = 'Neutral'
    if pos_count > neg_count:
        sentiment = 'Positive'
    elif neg_count > pos_count:
        sentiment = 'Negative'
    
    return jsonify({
        'label': 'FAKE' if is_fake else 'REAL',
        'confidence': confidence,
        'sentiment': sentiment
    })

if __name__ == '__main__':
    app.run(debug=True)
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into text
    return ' '.join(tokens)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # For demonstration, using a simple rule-based approach
    # In production, you would load your trained model here
    suspicious_words = ['hoax', 'conspiracy', 'shocking', 'you won\'t believe',
                       'secret', 'miracle', 'cure', 'clickbait']
    
    # Count suspicious words
    count = sum(1 for word in suspicious_words if word in processed_text.lower())
    
    # Calculate confidence score
    confidence = min(count / len(suspicious_words), 1.0)
    
    # Determine if fake based on confidence threshold
    is_fake = confidence > 0.3
    
    # Simple sentiment analysis
    positive_words = ['good', 'great', 'excellent', 'true', 'fact']
    negative_words = ['bad', 'false', 'fake', 'hoax', 'conspiracy']
    
    pos_count = sum(1 for word in positive_words if word in processed_text.lower())
    neg_count = sum(1 for word in negative_words if word in processed_text.lower())
    
    sentiment = 'Neutral'
    if pos_count > neg_count:
        sentiment = 'Positive'
    elif neg_count > pos_count:
        sentiment = 'Negative'
    
    return jsonify({
        'label': 'FAKE' if is_fake else 'REAL',
        'confidence': confidence,
        'sentiment': sentiment
    })

if __name__ == '__main__':
    app.run(debug=True)
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into text
    return ' '.join(tokens)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # For demonstration, using a simple rule-based approach
    # In production, you would load your trained model here
    suspicious_words = ['hoax', 'conspiracy', 'shocking', 'you won\'t believe',
                       'secret', 'miracle', 'cure', 'clickbait']
    
    # Count suspicious words
    count = sum(1 for word in suspicious_words if word in processed_text.lower())
    
    # Calculate confidence score
    confidence = min(count / len(suspicious_words), 1.0)
    
    # Determine if fake based on confidence threshold
    is_fake = confidence > 0.3
    
    # Simple sentiment analysis
    positive_words = ['good', 'great', 'excellent', 'true', 'fact']
    negative_words = ['bad', 'false', 'fake', 'hoax', 'conspiracy']
    
    pos_count = sum(1 for word in positive_words if word in processed_text.lower())
    neg_count = sum(1 for word in negative_words if word in processed_text.lower())
    
    sentiment = 'Neutral'
    if pos_count > neg_count:
        sentiment = 'Positive'
    elif neg_count > pos_count:
        sentiment = 'Negative'
    
    return jsonify({
        'label': 'FAKE' if is_fake else 'REAL',
        'confidence': confidence,
        'sentiment': sentiment
    })

if __name__ == '__main__':
    app.run(debug=True)
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into text
    return ' '.join(tokens)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # For demonstration, using a simple rule-based approach
    # In production, you would load your trained model here
    suspicious_words = ['hoax', 'conspiracy', 'shocking', 'you won\'t believe',
                       'secret', 'miracle', 'cure', 'clickbait']
    
    # Count suspicious words
    count = sum(1 for word in suspicious_words if word in processed_text.lower())
    
    # Calculate confidence score
    confidence = min(count / len(suspicious_words), 1.0)
    
    # Determine if fake based on confidence threshold
    is_fake = confidence > 0.3
    
    # Simple sentiment analysis
    positive_words = ['good', 'great', 'excellent', 'true', 'fact']
    negative_words = ['bad', 'false', 'fake', 'hoax', 'conspiracy']
    
    pos_count = sum(1 for word in positive_words if word in processed_text.lower())
    neg_count = sum(1 for word in negative_words if word in processed_text.lower())
    
    sentiment = 'Neutral'
    if pos_count > neg_count:
        sentiment = 'Positive'
    elif neg_count > pos_count:
        sentiment = 'Negative'
    
    return jsonify({
        'label': 'FAKE' if is_fake else 'REAL',
        'confidence': confidence,
        'sentiment': sentiment
    })

if __name__ == '__main__':
    app.run(debug=True)
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into text
    return ' '.join(tokens)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # For demonstration, using a simple rule-based approach
    # In production, you would load your trained model here
    suspicious_words = ['hoax', 'conspiracy', 'shocking', 'you won\'t believe',
                       'secret', 'miracle', 'cure', 'clickbait']
    
    # Count suspicious words
    count = sum(1 for word in suspicious_words if word in processed_text.lower())
    
    # Calculate confidence score
    confidence = min(count / len(suspicious_words), 1.0)
    
    # Determine if fake based on confidence threshold
    is_fake = confidence > 0.3
    
    # Simple sentiment analysis
    positive_words = ['good', 'great', 'excellent', 'true', 'fact']
    negative_words = ['bad', 'false', 'fake', 'hoax', 'conspiracy']
    
    pos_count = sum(1 for word in positive_words if word in processed_text.lower())
    neg_count = sum(1 for word in negative_words if word in processed_text.lower())
    
    sentiment = 'Neutral'
    if pos_count > neg_count:
        sentiment = 'Positive'
    elif neg_count > pos_count:
        sentiment = 'Negative'
    
    return jsonify({
        'label': 'FAKE' if is_fake else 'REAL',
        'confidence': confidence,
        'sentiment': sentiment
    })

if __name__ == '__main__':
    app.run(debug=True)
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into text
    return ' '.join(tokens)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # For demonstration, using a simple rule-based approach
    # In production, you would load your trained model here
    suspicious_words = ['hoax', 'conspiracy', 'shocking', 'you won\'t believe',
                       'secret', 'miracle', 'cure', 'clickbait']
    
    # Count suspicious words
    count = sum(1 for word in suspicious_words if word in processed_text.lower())
    
    # Calculate confidence score
    confidence = min(count / len(suspicious_words), 1.0)
    
    # Determine if fake based on confidence threshold
    is_fake = confidence > 0.3
    
    # Simple sentiment analysis
    positive_words = ['good', 'great', 'excellent', 'true', 'fact']
    negative_words = ['bad', 'false', 'fake', 'hoax', 'conspiracy']
    
    pos_count = sum(1 for word in positive_words if word in processed_text.lower())
    neg_count = sum(1 for word in negative_words if word in processed_text.lower())
    
    sentiment = 'Neutral'
    if pos_count > neg_count:
        sentiment = 'Positive'
    elif neg_count > pos_count:
        sentiment = 'Negative'
    
    return jsonify({
        'label': 'FAKE' if is_fake else 'REAL',
        'confidence': confidence,
        'sentiment': sentiment
    })

if __name__ == '__main__':
    app.run(debug=True)
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into text
    return ' '.join(tokens)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # For demonstration, using a simple rule-based approach
    # In production, you would load your trained model here
    suspicious_words = ['hoax', 'conspiracy', 'shocking', 'you won\'t believe',
                       'secret', 'miracle', 'cure', 'clickbait']
    
    # Count suspicious words
    count = sum(1 for word in suspicious_words if word in processed_text.lower())
    
    # Calculate confidence score
    confidence = min(count / len(suspicious_words), 1.0)
    
    # Determine if fake based on confidence threshold
    is_fake = confidence > 0.3
    
    # Simple sentiment analysis
    positive_words = ['good', 'great', 'excellent', 'true', 'fact']
    negative_words = ['bad', 'false', 'fake', 'hoax', 'conspiracy']
    
    pos_count = sum(1 for word in positive_words if word in processed_text.lower())
    neg_count = sum(1 for word in negative_words if word in processed_text.lower())
    
    sentiment = 'Neutral'
    if pos_count > neg_count:
        sentiment = 'Positive'
    elif neg_count > pos_count:
        sentiment = 'Negative'
    
    return jsonify({
        'label': 'FAKE' if is_fake else 'REAL',
        'confidence': confidence,
        'sentiment': sentiment
    })

if __name__ == '__main__':
    app.run(debug=True)
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into text
    return ' '.join(tokens)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # For demonstration, using a simple rule-based approach
    # In production, you would load your trained model here
    suspicious_words = ['hoax', 'conspiracy', 'shocking', 'you won\'t believe',
                       'secret', 'miracle', 'cure', 'clickbait']
    
    # Count suspicious words
    count = sum(1 for word in suspicious_words if word in processed_text.lower())
    
    # Calculate confidence score
    confidence = min(count / len(suspicious_words), 1.0)
    
    # Determine if fake based on confidence threshold
    is_fake = confidence > 0.3
    
    # Simple sentiment analysis
    positive_words = ['good', 'great', 'excellent', 'true', 'fact']
    negative_words = ['bad', 'false', 'fake', 'hoax', 'conspiracy']
    
    pos_count = sum(1 for word in positive_words if word in processed_text.lower())
    neg_count = sum(1 for word in negative_words if word in processed_text.lower())
    
    sentiment = 'Neutral'
    if pos_count > neg_count:
        sentiment = 'Positive'
    elif neg_count > pos_count:
        sentiment = 'Negative'
    
    return jsonify({
        'label': 'FAKE' if is_fake else 'REAL',
        'confidence': confidence,
        'sentiment': sentiment
    })

if __name__ == '__main__':
    app.run(debug=True)
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into text
    return ' '.join(tokens)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # For demonstration, using a simple rule-based approach
    # In production, you would load your trained model here
    suspicious_words = ['hoax', 'conspiracy', 'shocking', 'you won\'t believe',
                       'secret', 'miracle', 'cure', 'clickbait']
    
    # Count suspicious words
    count = sum(1 for word in suspicious_words if word in processed_text.lower())
    
    # Calculate confidence score
    confidence = min(count / len(suspicious_words), 1.0)
    
    # Determine if fake based on confidence threshold
    is_fake = confidence > 0.3
    
    # Simple sentiment analysis
    positive_words = ['good', 'great', 'excellent', 'true', 'fact']
    negative_words = ['bad', 'false', 'fake', 'hoax', 'conspiracy']
    
    pos_count = sum(1 for word in positive_words if word in processed_text.lower())
    neg_count = sum(1 for word in negative_words if word in processed_text.lower())
    
    sentiment = 'Neutral'
    if pos_count > neg_count:
        sentiment = 'Positive'
    elif neg_count > pos_count:
        sentiment = 'Negative'
    
    return jsonify({
        'label': 'FAKE' if is_fake else 'REAL',
        'confidence': confidence,
        'sentiment': sentiment
    })

if __name__ == '__main__':
    app.run(debug=True)
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into text
    return ' '.join(tokens)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # For demonstration, using a simple rule-based approach
    # In production, you would load your trained model here
    suspicious_words = ['hoax', 'conspiracy', 'shocking', 'you won\'t believe',
                       'secret', 'miracle', 'cure', 'clickbait']
    
    # Count suspicious words
    count = sum(1 for word in suspicious_words if word in processed_text.lower())
    
    # Calculate confidence score
    confidence = min(count / len(suspicious_words), 1.0)
    
    # Determine if fake based on confidence threshold
    is_fake = confidence > 0.3
    
    # Simple sentiment analysis
    positive_words = ['good', 'great', 'excellent', 'true', 'fact']
    negative_words = ['bad', 'false', 'fake', 'hoax', 'conspiracy']
    
    pos_count = sum(1 for word in positive_words if word in processed_text.lower())
    neg_count = sum(1 for word in negative_words if word in processed_text.lower())
    
    sentiment = 'Neutral'
    if pos_count > neg_count:
        sentiment = 'Positive'
    elif neg_count > pos_count:
        sentiment = 'Negative'
    
    return jsonify({
        'label': 'FAKE' if is_fake else 'REAL',
        'confidence': confidence,
        'sentiment': sentiment
    })

if __name__ == '__main__':
    app.run(debug=True)
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into text
    return ' '.join(tokens)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # For demonstration, using a simple rule-based approach
    # In production, you would load your trained model here
    suspicious_words = ['hoax', 'conspiracy', 'shocking', 'you won\'t believe',
                       'secret', 'miracle', 'cure', 'clickbait']
    
    # Count suspicious words
    count = sum(1 for word in suspicious_words if word in processed_text.lower())
    
    # Calculate confidence score
    confidence = min(count / len(suspicious_words), 1.0)
    
    # Determine if fake based on confidence threshold
    is_fake = confidence > 0.3
    
    # Simple sentiment analysis
    positive_words = ['good', 'great', 'excellent', 'true', 'fact']
    negative_words = ['bad', 'false', 'fake', 'hoax', 'conspiracy']
    
    pos_count = sum(1 for word in positive_words if word in processed_text.lower())
    neg_count = sum(1 for word in negative_words if word in processed_text.lower())
    
    sentiment = 'Neutral'
    if pos_count > neg_count:
        sentiment = 'Positive'
    elif neg_count > pos_count:
        sentiment = 'Negative'
    
    return jsonify({
        'label': 'FAKE' if is_fake else 'REAL',
        'confidence': confidence,
        'sentiment': sentiment
    })

if __name__ == '__main__':
    app.run(debug=True)
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into text
    return ' '.join(tokens)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # For demonstration, using a simple rule-based approach
    # In production, you would load your trained model here
    suspicious_words = ['hoax', 'conspiracy', 'shocking', 'you won\'t believe',
                       'secret', 'miracle', 'cure', 'clickbait']
    
    # Count suspicious words
    count = sum(1 for word in suspicious_words if word in processed_text.lower())
    
    # Calculate confidence score
    confidence = min(count / len(suspicious_words), 1.0)
    
    # Determine if fake based on confidence threshold
    is_fake = confidence > 0.3
    
    # Simple sentiment analysis
    positive_words = ['good', 'great', 'excellent', 'true', 'fact']
    negative_words = ['bad', 'false', 'fake', 'hoax', 'conspiracy']
    
    pos_count = sum(1 for word in positive_words if word in processed_text.lower())
    neg_count = sum(1 for word in negative_words if word in processed_text.lower())
    
    sentiment = 'Neutral'
    if pos_count > neg_count:
        sentiment = 'Positive'
    elif neg_count > pos_count:
        sentiment = 'Negative'
    
    return jsonify({
        'label': 'FAKE' if is_fake else 'REAL',
        'confidence': confidence,
        'sentiment': sentiment
    })

if __name__ == '__main__':
    app.run(debug=True)
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into text
    return ' '.join(tokens)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # For demonstration, using a simple rule-based approach
    # In production, you would load your trained model here
    suspicious_words = ['hoax', 'conspiracy', 'shocking', 'you won\'t believe',
                       'secret', 'miracle', 'cure', 'clickbait']
    
    # Count suspicious words
    count = sum(1 for word in suspicious_words if word in processed_text.lower())
    
    # Calculate confidence score
    confidence = min(count / len(suspicious_words), 1.0)
    
    # Determine if fake based on confidence threshold
    is_fake = confidence > 0.3
    
    # Simple sentiment analysis
    positive_words = ['good', 'great', 'excellent', 'true', 'fact']
    negative_words = ['bad', 'false', 'fake', 'hoax', 'conspiracy']
    
    pos_count = sum(1 for word in positive_words if word in processed_text.lower())
    neg_count = sum(1 for word in negative_words if word in processed_text.lower())
    
    sentiment = 'Neutral'
    if pos_count > neg_count:
        sentiment = 'Positive'
    elif neg_count > pos_count:
        sentiment = 'Negative'
    
    return jsonify({
        'label': 'FAKE' if is_fake else 'REAL',
        'confidence': confidence,
        'sentiment': sentiment
    })

if __name__ == '__main__':
    app.run(debug=True)
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into text
    return ' '.join(tokens)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # For demonstration, using a simple rule-based approach
    # In production, you would load your trained model here
    suspicious_words = ['hoax', 'conspiracy', 'shocking', 'you won\'t believe',
                       'secret', 'miracle', 'cure', 'clickbait']
    
    # Count suspicious words
    count = sum(1 for word in suspicious_words if word in processed_text.lower())
    
    # Calculate confidence score
    confidence = min(count / len(suspicious_words), 1.0)
    
    # Determine if fake based on confidence threshold
    is_fake = confidence > 0.3
    
    # Simple sentiment analysis
    positive_words = ['good', 'great', 'excellent', 'true', 'fact']
    negative_words = ['bad', 'false', 'fake', 'hoax', 'conspiracy']
    
    pos_count = sum(1 for word in positive_words if word in processed_text.lower())
    neg_count = sum(1 for word in negative_words if word in processed_text.lower())
    
    sentiment = 'Neutral'
    if pos_count > neg_count:
        sentiment = 'Positive'
    elif neg_count > pos_count:
        sentiment = 'Negative'
    
    return jsonify({
        'label': 'FAKE' if is_fake else 'REAL',
        'confidence': confidence,
        'sentiment': sentiment
    })

if __name__ == '__main__':
    app.run(debug=True)
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into text
    return ' '.join(tokens)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # For demonstration, using a simple rule-based approach
    # In production, you would load your trained model here
    suspicious_words = ['hoax', 'conspiracy', 'shocking', 'you won\'t believe',
                       'secret', 'miracle', 'cure', 'clickbait']
    
    # Count suspicious words
    count = sum(1 for word in suspicious_words if word in processed_text.lower())
    
    # Calculate confidence score
    confidence = min(count / len(suspicious_words), 1.0)
    
    # Determine if fake based on confidence threshold
    is_fake = confidence > 0.3
    
    # Simple sentiment analysis
    positive_words = ['good', 'great', 'excellent', 'true', 'fact']
    negative_words = ['bad', 'false', 'fake', 'hoax', 'conspiracy']
    
    pos_count = sum(1 for word in positive_words if word in processed_text.lower())
    neg_count = sum(1 for word in negative_words if word in processed_text.lower())
    
    sentiment = 'Neutral'
    if pos_count > neg_count:
        sentiment = 'Positive'
    elif neg_count > pos_count:
        sentiment = 'Negative'
    
    return jsonify({
        'label': 'FAKE' if is_fake else 'REAL',
        'confidence': confidence,
        'sentiment': sentiment
    })

if __name__ == '__main__':
    app.run(debug=True)
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into text
    return ' '.join(tokens)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # For demonstration, using a simple rule-based approach
    # In production, you would load your trained model here
    suspicious_words = ['hoax', 'conspiracy', 'shocking', 'you won\'t believe',
                       'secret', 'miracle', 'cure', 'clickbait']
    
    # Count suspicious words
    count = sum(1 for word in suspicious_words if word in processed_text.lower())
    
    # Calculate confidence score
    confidence = min(count / len(suspicious_words), 1.0)
    
    # Determine if fake based on confidence threshold
    is_fake = confidence > 0.3
    
    # Simple sentiment analysis
    positive_words = ['good', 'great', 'excellent', 'true', 'fact']
    negative_words = ['bad', 'false', 'fake', 'hoax', 'conspiracy']
    
    pos_count = sum(1 for word in positive_words if word in processed_text.lower())
    neg_count = sum(1 for word in negative_words if word in processed_text.lower())
    
    sentiment = 'Neutral'
    if pos_count > neg_count:
        sentiment = 'Positive'
    elif neg_count > pos_count:
        sentiment = 'Negative'
    
    return jsonify({
        'label': 'FAKE' if is_fake else 'REAL',
        'confidence': confidence,
        'sentiment': sentiment
    })

if __name__ == '__main__':
    app.run(debug=True)
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into text
    return ' '.join(tokens)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # For demonstration, using a simple rule-based approach
    # In production, you would load your trained model here
    suspicious_words = ['hoax', 'conspiracy', 'shocking', 'you won\'t believe',
                       'secret', 'miracle', 'cure', 'clickbait']
    
    # Count suspicious words
    count = sum(1 for word in suspicious_words if word in processed_text.lower())
    
    # Calculate confidence score
    confidence = min(count / len(suspicious_words), 1.0)
    
    # Determine if fake based on confidence threshold
    is_fake = confidence > 0.3
    
    # Simple sentiment analysis
    positive_words = ['good', 'great', 'excellent', 'true', 'fact']
    negative_words = ['bad', 'false', 'fake', 'hoax', 'conspiracy']
    
    pos_count = sum(1 for word in positive_words if word in processed_text.lower())
    neg_count = sum(1 for word in negative_words if word in processed_text.lower())
    
    sentiment = 'Neutral'
    if pos_count > neg_count:
        sentiment = 'Positive'
    elif neg_count > pos_count:
        sentiment = 'Negative'
    
    return jsonify({
        'label': 'FAKE' if is_fake else 'REAL',
        'confidence': confidence,
        'sentiment': sentiment
    })

if __name__ == '__main__':
    app.run(debug=True)
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into text
    return ' '.join(tokens)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # For demonstration, using a simple rule-based approach
    # In production, you would load your trained model here
    suspicious_words = ['hoax', 'conspiracy', 'shocking', 'you won\'t believe',
                       'secret', 'miracle', 'cure', 'clickbait']
    
    # Count suspicious words
    count = sum(1 for word in suspicious_words if word in processed_text.lower())
    
    # Calculate confidence score
    confidence = min(count / len(suspicious_words), 1.0)
    
    # Determine if fake based on confidence threshold
    is_fake = confidence > 0.3
    
    # Simple sentiment analysis
    positive_words = ['good', 'great', 'excellent', 'true', 'fact']
    negative_words = ['bad', 'false', 'fake', 'hoax', 'conspiracy']
    
    pos_count = sum(1 for word in positive_words if word in processed_text.lower())
    neg_count = sum(1 for word in negative_words if word in processed_text.lower())
