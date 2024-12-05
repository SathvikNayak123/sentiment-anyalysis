# Sentiment Analysis with DistilBERT

## Project Overview  
This project focuses on building a sentiment analysis model to predict the sentiment of customer reviews using **DistilBERT**. The pipeline involves scraping reviews, preprocessing the data, training a fine-tuned DistilBERT model, and deploying it through a Flask application. The workflow is automated using **Apache Airflow**.

---

## Features  
- **Data Collection**:  
  - Scraped 20,000+ Amazon reviews using **Selenium WebDriver**.  
  - Stored the collected data securely in an **AWS S3 bucket**.  

- **Data Preprocessing**:  
  - Cleaned and processed reviews and ratings:  
    - Removed non-English Reviews
    - Removed stopwords, special characters, and extra spaces.  
    - Performed lemmatization and stemming to normalize text.
    - Encoded ratings ranging from 0-5 stars to labels negative(0), neutral(1) and positive(2)  
  - Stored the processed data in **S3** for further use.  

- **Data Tokenization and Preparation**:  
  - Tokenized reviews using **DistilBERT tokenizer**.  
  - Divided the dataset into **training**, **validation**, and **testing** sets.

- **Model Training**:  
  - Imported **DistilBERT** from **Hugging Face Transformers**.  
  - Fine-tuned the model on the dataset.  
  - Implemented **early stopping** to optimize training.  
  - Used **customized class weights** to handle class imbalance in training dataset.

- **Deployment**:  
  - Built a user-friendly **Flask application** for making predictions.
    ![Flask](docs/Screenshot%202024-09-29%20171650.png)

- **Workflow Automation**:  
  - Integrated **Apache Airflow** to automate the entire pipeline. 
    ![Airflow](docs/Screenshot%202024-11-10%20215335.png)

---

## Result

- Accuracy: Achieved an accuracy of 95% on the test dataset.
    ![Result](docs/download.png)

## Getting Started  

To get started with this project, follow these steps:

1. Clone the repository:  
   ```bash
   git clone https://github.com/SathvikNayak123/sentiment-anyalysis
   ```
2. Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Add AWS S3 credentials in .env file

4. Run Airflow to execute scraping and training pipeline:
    ```bash
    astro dev init
    astro dev start

5. Run app for prediction
    ```bash
    python app.py
    ```
