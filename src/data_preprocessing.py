import re
import string
import nltk
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import email
from email.policy import default

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class EmailPreprocessor:
    """Comprehensive email preprocessing for spam classification."""
    
    def __init__(self):
        # Fallback stop words if NLTK data is not available
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            # Basic English stop words as fallback
            self.stop_words = set([
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
                'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                'further', 'then', 'once'
            ])
        self.stemmer = PorterStemmer()
        
    def clean_email_content(self, text):
        """Clean and preprocess email text."""
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Parse HTML if present
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()
        
        # Remove email headers and metadata
        text = re.sub(r'^(From|To|Subject|Date|Reply-To|Message-ID):.*$', '', text, flags=re.MULTILINE)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize_and_stem(self, text):
        """Tokenize text and apply stemming."""
        if not text:
            return []
        
        # Tokenize with fallback
        try:
            tokens = word_tokenize(text)
        except LookupError:
            # Simple fallback tokenization
            import string
            # Remove punctuation and split
            tokens = text.translate(str.maketrans('', '', string.punctuation)).split()
        
        # Remove stopwords and apply stemming
        tokens = [self.stemmer.stem(token) for token in tokens 
                 if token.lower() not in self.stop_words and len(token) > 2]
        
        return tokens
    
    def preprocess_text(self, text):
        """Complete text preprocessing pipeline."""
        cleaned_text = self.clean_email_content(text)
        tokens = self.tokenize_and_stem(cleaned_text)
        return ' '.join(tokens)
    
    def extract_features(self, text):
        """Extract additional features from email text."""
        if pd.isna(text) or text is None:
            text = ""
        
        text = str(text)
        
        features = {
            'length': len(text),
            'num_words': len(text.split()),
            'num_sentences': len(re.findall(r'[.!?]+', text)),
            'num_exclamation': text.count('!'),
            'num_question': text.count('?'),
            'num_uppercase': sum(1 for c in text if c.isupper()),
            'num_digits': sum(1 for c in text if c.isdigit()),
            'has_money_words': int(any(word in text.lower() for word in 
                                    ['money', 'cash', 'prize', 'winner', 'free', 'offer'])),
            'has_urgent_words': int(any(word in text.lower() for word in 
                                      ['urgent', 'immediate', 'act now', 'limited time'])),
            'capital_ratio': sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0,
        }
        
        return features

class DataLoader:
    """Load and prepare spam classification datasets."""
    
    def __init__(self):
        self.preprocessor = EmailPreprocessor()
    
    def load_spam_dataset(self, file_path=None):
        """Load spam dataset (SMS Spam Collection or similar format)."""
        if file_path is None:
            # Create a larger sample dataset for demonstration
            sample_data = [
                # Ham emails (legitimate)
                ("ham", "Hey, how are you doing today? Hope everything is going well."),
                ("ham", "Can we meet for coffee tomorrow at 3 PM? I'd love to catch up."),
                ("ham", "Thanks for your help with the project yesterday. Really appreciated it."),
                ("ham", "I'll be late for the meeting, please start without me. Traffic is terrible."),
                ("ham", "Don't forget about mom's birthday next week. Should we plan something?"),
                ("ham", "The weather is really nice today, perfect for a walk in the park."),
                ("ham", "Looking forward to seeing you at the conference next month."),
                ("ham", "Could you please send me the report when you get a chance?"),
                ("ham", "Great job on the presentation! The client was really impressed."),
                ("ham", "Let me know if you need any help with the assignment."),
                ("ham", "Happy birthday! Hope you have a wonderful day."),
                ("ham", "The meeting has been moved to next Tuesday at 2 PM."),
                ("ham", "Thank you for joining our team. We're excited to work with you."),
                ("ham", "Please review the attached document and let me know your thoughts."),
                ("ham", "Reminder: Your appointment is scheduled for tomorrow at 10 AM."),
                ("ham", "I'm running a few minutes late, but I'll be there soon."),
                ("ham", "Thanks for the recommendation. I'll definitely check it out."),
                ("ham", "Hope you're feeling better. Take care of yourself."),
                ("ham", "The new software update is available for download."),
                ("ham", "Your order has been processed and will ship tomorrow."),
                ("ham", "Welcome to our newsletter! Here's what's new this month."),
                ("ham", "Just wanted to check in and see how things are going."),
                ("ham", "The event was a great success. Thanks for organizing it."),
                ("ham", "Please confirm your attendance for the workshop next week."),
                ("ham", "Your subscription will expire next month. Renew to continue."),
                
                # Spam emails (suspicious/promotional)
                ("spam", "WINNER! You've won $1000! Click here to claim your prize NOW!"),
                ("spam", "Urgent! Your account will be closed unless you verify immediately."),
                ("spam", "FREE MONEY! No strings attached! Call now for instant cash!"),
                ("spam", "Congratulations! You're pre-approved for a $5000 loan! Apply now!"),
                ("spam", "ALERT: Suspicious activity detected. Verify your account now!"),
                ("spam", "You've been selected for a special offer! Limited time only!"),
                ("spam", "BREAKING: Make $500/day working from home! No experience needed!"),
                ("spam", "URGENT: Your computer has been infected! Download our antivirus NOW!"),
                ("spam", "Amazing weight loss secret! Lose 30 pounds in 30 days guaranteed!"),
                ("spam", "FINAL NOTICE: Your warranty is about to expire. Renew immediately!"),
                ("spam", "Click here for FREE iPhone! Limited quantity available!"),
                ("spam", "You've inherited $2 million! Contact us to claim your fortune!"),
                ("spam", "STOP! Don't miss this incredible investment opportunity!"),
                ("spam", "Miracle cure discovered! Doctors hate this one simple trick!"),
                ("spam", "Act NOW! 90% discount expires in 1 hour! Don't miss out!"),
                ("spam", "CONGRATULATIONS! You're our 1 millionth visitor! Claim your prize!"),
                ("spam", "Easy money! Work from home and earn $1000 daily!"),
                ("spam", "Your credit score can be improved overnight! Call now!"),
                ("spam", "FREE vacation to Hawaii! All expenses paid! Click to enter!"),
                ("spam", "URGENT: IRS owes you money! Claim your refund today!"),
                ("spam", "Amazing opportunity! Get rich quick with this secret method!"),
                ("spam", "WARNING: Your internet will be disconnected unless you pay now!"),
                ("spam", "Exclusive offer just for you! 99% off everything!"),
                ("spam", "HURRY! Only 24 hours left to claim your free gift!"),
                ("spam", "You've been approved! Get your cash advance today!"),
                ("spam", "Revolutionary product! Transform your life in just 7 days!"),
                ("spam", "FINAL WARNING: Your account security is compromised!"),
                ("spam", "Incredible deal! Buy one get ten free! Limited time!"),
                ("spam", "You're eligible for a $10,000 grant! No repayment required!"),
                ("spam", "Secret method banks don't want you to know! Click here!"),
            ]
            
            df = pd.DataFrame(sample_data, columns=['label', 'text'])
        else:
            try:
                df = pd.read_csv(file_path, delimiter='\t', names=['label', 'text'])
            except:
                df = pd.read_csv(file_path)
        
        return df
    
    def prepare_dataset(self, df):
        """Prepare dataset for training."""
        # Clean and preprocess text
        df['cleaned_text'] = df['text'].apply(self.preprocessor.preprocess_text)
        
        # Extract additional features
        feature_dicts = df['text'].apply(self.preprocessor.extract_features)
        feature_df = pd.DataFrame(list(feature_dicts))
        
        # Combine with main dataframe
        df = pd.concat([df, feature_df], axis=1)
        
        # Convert labels to binary (0 for ham, 1 for spam)
        df['target'] = (df['label'] == 'spam').astype(int)
        
        return df

def create_vectorizers():
    """Create different types of vectorizers for feature extraction."""
    vectorizers = {
        'tfidf_word': TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True
        ),
        'tfidf_char': TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 4),
            max_features=3000
        ),
        'count': CountVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True
        )
    }
    
    return vectorizers

if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    df = loader.load_spam_dataset()
    prepared_df = loader.prepare_dataset(df)
    
    print("Dataset shape:", prepared_df.shape)
    print("Label distribution:")
    print(prepared_df['label'].value_counts())
    print("\nSample preprocessed text:")
    print(prepared_df[['text', 'cleaned_text']].head())
