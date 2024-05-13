# News Text Classification using Transformers

This project demonstrates the effectiveness of the Transformers library in text classification tasks, specifically applied to news data. The objective is to classify news articles into six predefined categories using state-of-the-art Transformer models.

## Overview

The project is divided into two main parts:

1. **Generating Labels with Natural Language Inference (NLI)**
2. **Fine-tuning BERT for Text Classification**

Sure, here are the steps for each part of the project listed in a numbered format:

### Part 1: Generating Labels by NLI

1. Obtain news data from a Git repository.
2. Process the data into a DataFrame.
3. Clean the data.
4. Provide the cleaned data to the `bert-base-nli-mean-tokens` Sentence Transformer model.
5. Distribute the news articles into six different categories: "Realistic", "Investigative", "Artistic", "Social", "Enterprising", and "Conventional" based on embeddings and cosine similarity.
6. Combine every three sentences of labeled data into a single row along with their respective labels.
7. Split the combined categories by commas and expand them into separate columns.
8. Generate binary columns for each unique value, where 1 indicates the presence of the category and 0 indicates absence.
9. Export the processed data to a single CSV file.

### Part 2: Fine-tuning BERT

1. Load the labeled data obtained from the previous model from the CSV file into a DataFrame.
2. Split the data into training and testing sets.
3. Process the testing data using `bert-base-uncased` by tokenizing it with its respective tokenizer and embedding it.
4. Build the model by creating layers for each category.
5. Apply a Dropout layer to the [CLS] token embedding for each category to introduce regularization and prevent overfitting.
6. Use a Dense layer to map the dropout output to a single output value for each category.
7. Utilize the sigmoid activation function, indicating a binary classification task.
8. Employ binary cross-entropy loss, treating each layer/class separately.
9. Train the model.
10. Test the model and evaluate accuracy scores for each category.

## Requirements

- Google Collab Python Compute
- Transformers library
- Pandas
- Scikit-learn
- TensorFlow

## Usage

1. Clone the repository.
2. Run each part of the project sequentially as described in the code or documentation.

## Acknowledgments

- deepakat002 for making such a great tutorial
- Contributors to this Git repository for implementing code for this project.
- The Transformers library by Hugging Face for providing pre-trained models and utilities for natural language processing tasks.

## Data Source

[Kaggle Link](https://www.kaggle.com/code/deepakat002/text-classification-without-training-data-bert)
