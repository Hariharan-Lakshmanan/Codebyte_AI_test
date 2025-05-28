# Python & Machine Learning Project

This repository contains solutions to a set of Python, data manipulation, machine learning, and AI/ML conceptual questions. Each section corresponds to specific tasks involving data cleanup, statistical calculations, model training, evaluation, and simple natural language processing.

---

## Project Structure

- **Section A: Python & Data Manipulation**
- **Section B: Machine Learning**
- **Section C: AI/ML Application & Thinking**

---

## Section A: Python & Data Manipulation

### Q1. Data Cleanup & Summary
- Input: `student_scores.csv` with columns: Name, Math, Science, English, Gender.
- Tasks:
  - Fill missing numeric values with the mean of the column.
  - Convert Gender column into binary values (e.g., Male = 0, Female = 1).
  - Return a summary DataFrame showing average scores per gender.

### Q2. Dictionary-Based Stats
- Input: A dictionary with user IDs as keys and a list of scores as values.
- Output: A new dictionary with each user’s average, minimum, and maximum scores.

---

## Section B: Machine Learning

### Q3. Classifier on Iris Dataset
- Load Iris dataset from `sklearn.datasets`.
- Train a Decision Tree classifier.
- Perform an 80-20 train-test split.
- Predict and print the accuracy on the test set.
- Plot a confusion matrix using Matplotlib/Seaborn.

### Q4. Simple Regression on Housing Data
- Input: `simple_housing.csv` with columns: area, bedrooms, price.
- Build a linear regression model to predict price.
- Evaluate the model using Mean Absolute Error (MAE).
- Plot a scatter plot comparing actual vs predicted prices.

---

## Section C: AI/ML Application & Thinking

### Q5. Conceptual Questions
Brief answers to:
1. What is overfitting in machine learning?
2. When would you use a decision tree over logistic regression?
3. Explain the train-test split and why it’s important.
4. What’s the purpose of normalization?
5. What’s the difference between classification and regression?

### Q6. Simple NLP Task – Sentiment Classification
- Load two categories (`rec.autos` and `comp.sys.mac.hardware`) from `sklearn.datasets.fetch_20newsgroups`.
- Convert text data to vectors using `TfidfVectorizer`.
- Train a Logistic Regression classifier.
- Print classification accuracy.
- Display 5 most important words per class based on the model coefficients.

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Hariharan-Lakshmanan/Codebyte_AI_test.git
   cd Codebyte_AI_test

2. Install Dependencies:

- Python 3.x
- Jupyter Notebook
- pandas, numpy, scikit-learn, matplotlib, seaborn

  Install requirements using:

  ```bash
  pip install pandas numpy scikit-learn matplotlib seaborn

3. Run the scripts:
   
Section A:

Q1: Run python Data Cleanup & Summary
(This script reads the inputting dataframe (student_scores.csv), processes it, and prints the summary dataframe.)

Q2: Run python Dictionary-Based Stats
(This script demonstrates dictionary-based statistics and prints results.)

Section B:

Q3: Run python Classifier on iris
(Trains Decision Tree on Iris dataset, prints accuracy, and shows confusion matrix plot.)

Q4: Run python Simple Regression
(Builds regression model on synthetically created housing data, prints MAE, and displays actual vs predicted price plot.)

Section C:

Q5: Conceptual questions are answered in the Conceptual markdown line (no script to run).

Q6: Run python Simple NLP Task - Sentiment Classification
(Trains Logistic Regression on 20 Newsgroups data and prints accuracy.)

### Contact
Hariharan Lakshmanan
Email: harish872000@gmail.com
GitHub: Hariharan-Lakshmanan


