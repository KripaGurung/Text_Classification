# Sentiment Analysis of Financial Text Using LSTM, RNN, and Word2Vec
This project focuses on analyzing the sentiment of financial text data using machine learning and deep learning techniques. The aim is to classify text (e.g., news headlines or articles) as positive, negative, or neutral to assist in financial decision-making processes.

Developed in Python, the project integrates traditional models like Logistic Regression with deep learning architectures such as LSTM and RNN, and uses Word2Vec embeddings for capturing semantic meaning in text.

## Features
- Identifies the sentiment behind financial texts (positive, negative, or neutral).
- Removes noise, stop words, and performs tokenization and stemming for clean input.
- Implemented both TF-IDF and Word2Vec to convert textual data into numeric form.
- Trained and compared models including:
  - Recurrent Neural Network (RNN)
  - Long Short-Term Memory (LSTM)
- Used accuracy, confusion matrix, and classification report to evaluate model performance.
- Employed Matplotlib and Seaborn to plot sentiment distribution and model results.

## Technologies Used
- Python – Core programming language
- Pandas / NumPy – Data preprocessing and analysis
- Matplotlib / Seaborn – Data visualization
- Scikit-learn – Machine learning models and evaluation
- NLTK / spaCy – Natural Language Processing
- TensorFlow / Keras – Deep learning (LSTM, RNN)
- Gensim – Word2Vec embeddings

## Project Workflow
- Data Collection – Loaded financial text dataset (e.g. headlines/news).
- Data Cleaning – Removed unwanted characters, stopwords, and normalized text.
- Feature Extraction – Applied TF-IDF and Word2Vec for vectorization.
- Model Training – Built and trained ML & DL models for sentiment classification.
- Evaluation – Compared model performance with relevant metrics.
