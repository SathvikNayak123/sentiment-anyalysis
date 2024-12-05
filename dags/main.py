from dotenv import load_dotenv
import os
import pandas as pd
from components.data_transform import DataProcessor, Split_Tokenize_Data
from components.model_train import ModelTrainer
from components.data_collect import ScrapeData
from components.utils import Utils
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import os
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize parameters for DataProcessor and Split_Tokenize_Data
bucket_name = os.getenv('bucket')
raw_data_key = os.getenv('raw')
clean_data_key = os.getenv('clean')
train_data_key = os.getenv('train')
val_data_key = os.getenv('val')
test_data_key = os.getenv('test')
model_key = os.getenv('model')

# Define the URL and path for scraping
scraping_url = 'https://www.amazon.com/s?k=redmi+earbuds&crid=3TZZRJIPFBP2E&sprefix=redmi+earbuds%2Caps%2C380&ref=nb_sb_noss_1'
chromedriver_path = "chromedriver.exe"

# Epochs for Model Training
epochs = 50

# Define the DAG
default_args = {
    'owner': 'airflow',
    'retries': 3
}

dag = DAG(
    'model_training_pipeline',
    default_args=default_args,
    description='An end-to-end model training pipeline',
    schedule_interval=None,
    start_date=datetime(2024, 11, 10), 
    catchup=False,
)

def scrape_reviews():
    obj = ScrapeData(scraping_url, chromedriver_path)
    obj.scrapeReviews(100000)
    logger.info("---Data Collection Complete---")

    os.makedirs("artifacts", exist_ok=True)
    data = [{"Review": review, "Rating": rating} for review, rating in zip(obj.amazon_reviews, obj.amazon_ratings)]
    amazon_df = pd.DataFrame(data)
    amazon_df.to_csv("artifacts/amazon_data.csv", index=False)

    util = Utils()
    util.put_data_s3(amazon_df, bucket_name, raw_data_key)
    logger.info("---Uploaded Raw Data to S3---")

def preprocess_and_classify_reviews():
    data_processor = DataProcessor(bucket=bucket_name, raw_key=raw_data_key, clean_key=clean_data_key)
    data_processor.import_data_from_s3()  # Load raw data
    clean_df = data_processor.ProcessReviews()  # Process and classify reviews
    data_processor.save_cleaned_data_to_s3(clean_df)  # Save processed data to S3

def split_and_tokenize_data():
    split_tokenizer = Split_Tokenize_Data(
        bucket=bucket_name,
        clean_key=clean_data_key,
        train_key=train_data_key,
        val_key=val_data_key,
        test_key=test_data_key
    )
    split_tokenizer.import_data_from_s3()  # Load cleaned data
    train_df, val_df, test_df = split_tokenizer.preprocess_data()  # Split the data
    train_dataset, val_dataset, test_dataset = split_tokenizer.create_datasets(train_df, val_df, test_df)  # Tokenize and create datasets
    split_tokenizer.save_all_datasets(train_dataset, val_dataset, test_dataset)

def train_and_evaluate_model():
    trainer = ModelTrainer(bucket=bucket_name, model_key=model_key)
    trainer.initialize_model()
    trainer.train(epochs=epochs)
    trainer.evaluate()
    trainer.save_model()

# Define tasks
scraping_task = PythonOperator(
    task_id='scrape_reviews',
    python_callable=scrape_reviews,
    dag=dag,
)

preprocess_task = PythonOperator(
    task_id='preprocess_and_classify_reviews',
    python_callable=preprocess_and_classify_reviews,
    dag=dag,
)

split_tokenize_task = PythonOperator(
    task_id='split_and_tokenize_data',
    python_callable=split_and_tokenize_data,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_and_evaluate_model',
    python_callable=train_and_evaluate_model,
    dag=dag,
)

# Set task dependencies
scraping_task >> preprocess_task >> split_tokenize_task >> train_task