# Sentiment Analysis Using Machine Learning

A machine learning-based sentiment analysis system that classifies user reviews into three categories:

* **Positive**
* **Negative**
* **Neutral**

The project uses Natural Language Processing (NLP), TF-IDF feature extraction, and multiple machine learning algorithms. A hybrid rule-based layer is also included to handle special cases such as short comments, negation, emojis, and common sentiment expressions.

---

## Features

* Text preprocessing using NLP
* Emoji processing
* Stopword removal
* Lemmatization
* Negation handling
* Sarcasm pattern handling
* Contrast handling for words such as `but` and `however`
* TF-IDF feature extraction
* Unigram and bigram features
* Multiple machine learning models
* 5-fold cross-validation
* Classification reports
* Confusion matrices
* Model accuracy comparison
* Interactive sentiment prediction
* Hybrid rule-based + machine learning prediction
* Handles short comments such as `ok`, `ok ok`, `fine`, and `so so`
* Handles positive and negative expressions
* Handles phrases such as `not good` and `not bad`
* Handles common positive and negative emojis

---

## Machine Learning Models

The project compares four classification algorithms:

1. Logistic Regression
2. Multinomial Naive Bayes
3. Linear Support Vector Machine (SVM)
4. Random Forest

The model with the best cross-validation performance is selected as the final machine learning model.

---

## Dataset

The current dataset contains **1,152 reviews**.

### Sentiment Distribution

| Sentiment | Number of Reviews |
| --------- | ----------------: |
| Positive  |               440 |
| Negative  |               394 |
| Neutral   |               318 |
| **Total** |         **1,152** |

The dataset is divided into:

* **70% training data**
* **30% testing data**

The split is stratified to preserve the sentiment distribution.

---

## Text Preprocessing

The project performs several preprocessing operations before training.

### 1. Lowercasing

Example:

```
"This Product Is GREAT"
```

becomes:

```
this product is great
```

### 2. Emoji Processing

Common emojis are converted into meaningful words.

Examples:

```
👍 → good
🔥 → great
❤️ → love
😢 → sad
👎 → bad
```

### 3. Negation Handling

Negation is important in sentiment analysis.

Examples:

```
not good
not happy
never works
```

The system creates special features such as:

```
not_good
not_happy
```

This helps the model distinguish between:

```
good
```

and:

```
not good
```

### 4. Stopword Removal

Common words that provide little sentiment information are removed while preserving important negation words.

### 5. Lemmatization

Words are reduced to their base forms.

Example:

```
working
worked
works
```

are normalized toward their base form.

### 6. Contrast Handling

The system gives additional importance to the part of a sentence following contrast words such as:

```
but
however
although
though
```

For example:

```
The product looks good but it does not work.
```

The negative part receives additional importance.

---

## TF-IDF Feature Extraction

The project uses `TfidfVectorizer` with:

```python
ngram_range=(1, 2)
max_features=5000
min_df=2
max_df=0.8
sublinear_tf=True
```

Both individual words and two-word combinations are considered.

For example:

```
very good
not good
works well
poor quality
```

can become useful features.

---

## Model Performance

The models were evaluated using both test accuracy and 5-fold cross-validation.

| Model               | Test Accuracy | Cross-Validation Accuracy |
| ------------------- | ------------: | ------------------------: |
| Logistic Regression |        92.49% |                    92.31% |
| Naive Bayes         |        91.62% |                    90.82% |
| **SVM**             |    **92.77%** |                **93.79%** |
| Random Forest       |        89.02% |                    89.58% |

### Best Model

The **Linear SVM** achieved the best cross-validation performance:

```
Cross-Validation Accuracy: 93.79%
Test Accuracy: 92.77%
```

Therefore, SVM is selected as the final machine learning model.

---

## Hybrid Sentiment Analysis

Instead of relying entirely on machine learning, the project uses a hybrid approach.

The system first checks for clear rules and special cases. If no rule matches, the review is passed to the trained SVM.

### Neutral Examples

```
ok
ok ok
okay
okay okay
fine
so so
average
nothing special
not sure
maybe
I don't know
works as expected
```

These are classified as:

```
neutral
```

### Positive Examples

```
I love this product.
This is excellent.
It works perfectly.
Amazing quality.
Not bad.
👍
🔥
❤️
```

These are classified as:

```
positive
```

### Negative Examples

```
I hate this product.
This is terrible.
The product does not work.
The quality is very poor.
Not good.
👎
😡
```

These are classified as:

```
negative
```

---

## Handling Negation

The system specifically handles expressions where negation changes sentiment.

Examples:

```
This is good.
```

→ Positive

```
This is not good.
```

→ Negative

```
This is bad.
```

→ Negative

```
This is not bad.
```

→ Positive

The hybrid rules help make these predictions more reliable.

---

## Handling Short or Informal Comments

Short comments can be difficult for machine learning models because they contain very little information.

The system includes special handling for expressions such as:

```
ok
ok ok
okay
fine
so so
meh
maybe
nothing special
```

These are treated as neutral when appropriate.

---

---

## Project Structure

sentiment-analysis/
│
├── data/
│   └── reviews.csv
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md

---

## Installation

### 1. Clone the repository

```
git clone https://github.com/YOUR_USERNAME/sentiment-analysis.git
```

Move into the project directory:

```
cd sentiment-analysis
```

### 2. Create a virtual environment

Windows:

```
python -m venv venv
```

Activate it:

```
venv\Scripts\activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

If you do not have a `requirements.txt` file yet, install the main dependencies:

```
pip install pandas numpy nltk emoji matplotlib seaborn scikit-learn
```

### 4. Run the program

```
python main.py
```

---

## Example

After starting the program:

```text
======================================
       SENTIMENT ANALYSIS SYSTEM
======================================

Enter a review:
```

Enter:

```text
I absolutely love this product. It works perfectly!
```

Output:

```text
Predicted Sentiment: positive
```

Another example:

```text
Enter a review: This product stopped working after two days.
```

Output:

```text
Predicted Sentiment: negative
```

Another example:

```text
Enter a review: ok ok
```

Output:

```text
Predicted Sentiment: neutral
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* NLTK
* Scikit-learn
* Matplotlib
* Seaborn
* Emoji

---

## Python Libraries

The main libraries used in the project are:

```text
pandas
numpy
nltk
emoji
matplotlib
seaborn
scikit-learn
```

---

## Future Improvements

Possible improvements include:

* Increasing the size of the training dataset
* Adding more diverse neutral reviews
* Adding more examples of sarcasm
* Adding spelling-error handling
* Adding slang detection
* Improving mixed-sentiment detection
* Hyperparameter tuning
* Trying transformer-based models such as BERT
* Building a web interface
* Creating an API for sentiment prediction
* Saving the trained model with `joblib`
* Adding confidence scores
* Deploying the application online

---

## Limitations

Although the system achieves strong accuracy, sentiment analysis can be difficult for:

* Sarcasm
* Very short comments
* Spelling mistakes
* Ambiguous statements
* Context-dependent expressions
* Unusual slang
* Mixed emotions
* Words with different meanings in different contexts

The hybrid rules help with several of these cases, but they cannot guarantee perfect predictions.

---

## Conclusion

This project demonstrates how Natural Language Processing and machine learning can be used to classify reviews into positive, negative, and neutral sentiments.

Among the evaluated models, **Linear SVM achieved the best performance**, with:

```text
93.79% 5-Fold Cross-Validation Accuracy
92.77% Test Accuracy
```

The combination of **TF-IDF, NLP preprocessing, SVM, and rule-based sentiment handling** provides an effective approach for building a practical sentiment analysis system.

---

