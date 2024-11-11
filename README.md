# IMDB Movie Reviews Sentiment Analysis

## Project Overview
This project demonstrates a complete pipeline for sentiment analysis on a dataset of 50,000 IMDB movie reviews. Using a Long Short-Term Memory (LSTM) neural network, this code classifies movie reviews as either positive or negative. The project includes data preprocessing, model training, evaluation, and a function for sentiment prediction.

## Key Steps

### 1. Dataset Access and Preparation
- **Kaggle Integration**: The code starts by loading the `kaggle.json` file to set up Kaggle API credentials and download the dataset (`IMDB Dataset of 50K Movie Reviews`).
- **Data Extraction**: The dataset is extracted from a ZIP file, and a `pandas` DataFrame is created from the CSV file.
- **Label Encoding**: The 'sentiment' column is transformed into numerical labels using `LabelEncoder`, where 'positive' is encoded as `1` and 'negative` as `0`.
- **Data Splitting**: The dataset is split into training (80%) and testing (20%) sets for model evaluation.

### 2. Text Preprocessing
- **Tokenization**: The `Tokenizer` from Keras is used to tokenize the text reviews, restricting to the top 5,000 most common words.
- **Padding**: Reviews are converted to sequences of integers and padded to a uniform length of 200 words to ensure consistent input size.

### 3. Model Architecture
- **Embedding Layer**: Converts integer-encoded words into dense vectors of fixed size (128).
- **LSTM Layer**: A recurrent layer with 128 units that helps in capturing long-term dependencies in the text. Dropout and recurrent dropout are set at 0.2 to prevent overfitting.
- **Dense Layer**: A single unit with a sigmoid activation function outputs a probability score for binary classification.

### 4. Model Compilation and Training
- **Compilation**: The model uses `binary_crossentropy` as the loss function, `adam` optimizer, and tracks accuracy as the metric.
- **Training**: The model is trained with a batch size of 128 over 5 epochs and validated using 20% of the training data.

### 5. Model Evaluation
- **Performance Metrics**: The model is evaluated on the test set, reporting a loss of approximately 0.334 and an accuracy of around 86.57%.

### 6. Sentiment Prediction Function
- **Functionality**: The `predict_sentiment` function takes a raw text review, processes it through the trained tokenizer, and returns a prediction of either "Positive" or "Negative" based on the output probability.

## Results and Conclusion
The LSTM model achieved a test accuracy of approximately **86.57%**, demonstrating its effectiveness for text classification tasks.
