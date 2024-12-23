import tensorflow as tf
from transformers import DistilBertTokenizer
from sklearn.model_selection import StratifiedShuffleSplit
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from transformers import pipeline
from datasets import Dataset
import logging
import re
from components.utils import Utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

class DataProcessor:
    def __init__(self, bucket=None, raw_key=None, clean_key=None):
        self.bucket = bucket
        self.raw_key = raw_key
        self.clean_key = clean_key

        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

        self.amazon_df = None
        self.utils = Utils()

    def import_data_from_s3(self):
        self.amazon_df = self.utils.get_data_s3(self.bucket, self.raw_key)

    def detect_language(self, text):
        try:
            return detect(text)
        except LangDetectException:
            return 'unknown'

    def clean_text(self, review):
        # Remove URLs, HTML tags, special characters, and digits
        review = re.sub(r'http\S+|www\S+|https\S+', '', review, flags=re.MULTILINE)
        review = re.sub(r'<.*?>', '', review)
        review = re.sub(r'\d+', '', review)
        review = re.sub(r'[^A-Za-z\s]+', '', review)  # Keep only alphabets and spaces
        return review

    def remove_stopwords(self, review):
        return ' '.join([word for word in review.split() if word not in self.stop_words])

    def lemmatize_text(self, review):
        return ' '.join([self.lemmatizer.lemmatize(word) for word in review.split()])

    def preprocess_review(self, review):
        if not isinstance(review, str):
            return None
        if self.detect_language(review) != 'en':
            return None
        review = self.clean_text(review)
        review = review.lower()
        review = self.remove_stopwords(review)
        review = self.lemmatize_text(review)
        return review.strip() if len(review.strip()) > 0 else None

    def apply_preprocessing(self):
        self.amazon_df['Cleaned_Reviews'] = self.amazon_df['Review'].apply(self.preprocess_review)
        initial_count = len(self.amazon_df)
        self.amazon_df.dropna(subset=['Cleaned_Reviews'], inplace=True)
        final_count = len(self.amazon_df)
        print(f"Preprocessing complete. Dropped {initial_count - final_count} non-English or empty reviews.")

    def map_to_sentiment(self, rating):
        if rating <= 2:
            return 0
        elif 2 < rating < 4:
            return 1
        else:
            return 2

    def clean_rating(self):
        self.amazon_df["Cleaned_Rating"] = [float(rating.split()[0]) for rating in self.amazon_df["Rating"]]

        self.amazon_df["Sentiment"] = self.amazon_df["Cleaned_Rating"].apply(self.map_to_sentiment)

    def ProcessReviews(self):
        """
        Full processing pipeline: preprocess and classify reviews.
        """
        print("Starting preprocessing of reviews...")
        self.apply_preprocessing()
        print("Preprocessing done. Starting classification of reviews...")
        self.clean_rating()
        print("Classification done.")
        return self.amazon_df[["Cleaned_Reviews", "Sentiment"]]

    def save_cleaned_data_to_s3(self, classified_df):
        return self.utils.put_data_s3(classified_df, self.bucket, self.clean_key)
    

class Split_Tokenize_Data:
    def __init__(self, bucket=None, clean_key=None, train_key=None, val_key=None, test_key=None):

        self.bucket = bucket
        self.clean_key = clean_key
        self.train_key = train_key
        self.val_key = val_key
        self.test_key = test_key

        self.tokenizer = None

        self.utils = Utils()
        self.clean_df = None

    def import_data_from_s3(self):
        self.clean_df = self.utils.get_data_s3(self.bucket, self.clean_key)

    def preprocess_data(self):
        """
        Maps labels and splits the data into training, validation, and testing sets.
        """
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in sss.split(self.clean_df, self.clean_df['Sentiment']):
            train_df = self.clean_df.iloc[train_index]
            test_df = self.clean_df.iloc[test_index]
        print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
        
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, val_index in sss_val.split(train_df, train_df['Sentiment']):
            train_split_df = train_df.iloc[train_index]
            val_df = train_df.iloc[val_index]
        print(f"Training size: {len(train_split_df)}, Validation size: {len(val_df)}")
        
        return train_split_df, val_df, test_df

    def download_tokenizer(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def tokenize_data(self, texts, padding=True, truncation=True, return_tensors="tf"):
        return self.tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors
        )

    def create_datasets(self, train_df, val_df, test_df, batch_size=16):
        """
        Creates TensorFlow datasets from the DataFrames.
        """
        print("Creating TensorFlow datasets...")
        self.download_tokenizer()
        train_encodings = self.tokenize_data(train_df['Cleaned_Reviews'].tolist())
        val_encodings = self.tokenize_data(val_df['Cleaned_Reviews'].tolist())
        test_encodings = self.tokenize_data(test_df['Cleaned_Reviews'].tolist())
        
        train_dataset = tf.data.Dataset.from_tensor_slices((
            dict(train_encodings),
            train_df['Sentiment'].values
        )).shuffle(10000).batch(batch_size)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((
            dict(val_encodings),
            val_df['Sentiment'].values
        )).batch(batch_size)
        
        test_dataset = tf.data.Dataset.from_tensor_slices((
            dict(test_encodings),
            test_df['Sentiment'].values
        )).batch(batch_size)
        
        print("Datasets created successfully.")
        return train_dataset, val_dataset, test_dataset
    
    def save_all_datasets(self, train_dataset, val_dataset, test_dataset):
        train_dataset.save('artifacts/train_data')
        val_dataset.save('artifacts/val_data')
        test_dataset.save('artifacts/test_data')

        self.utils.upload_directory_to_s3('artifacts/train_data', self.bucket, self.train_key)
        self.utils.upload_directory_to_s3('artifacts/val_data', self.bucket, self.val_key)
        self.utils.upload_directory_to_s3('artifacts/test_data', self.bucket, self.test_key)

        print("-----------Saved Train, Val and Test datasets--------\n")