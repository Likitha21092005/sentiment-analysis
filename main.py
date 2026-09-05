import pandas as pd
import numpy as np
import re
import nltk
import string
import emoji
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# DOWNLOAD NLTK RESOURCES
# ============================================================

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("reviews.csv")

# Remove duplicate rows
df = df.drop_duplicates()

# Remove rows with missing review/sentiment
df = df.dropna(subset=["review", "sentiment"])

# Shuffle dataset
df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


print("\n======================================")
print("           DATASET INFORMATION")
print("======================================")

print("Total reviews:", len(df))

print("\nSentiment Distribution:")
print(df["sentiment"].value_counts())


# ============================================================
# NLP SETUP
# ============================================================

stop_words = set(
    stopwords.words("english")
)

# IMPORTANT:
# Do not remove negation words such as:
# not, no, never, don't, can't
#
# They are important for sentiment analysis.

important_negation_words = {
    "not",
    "no",
    "nor",
    "never",
    "don't",
    "dont",
    "didn't",
    "didnt",
    "isn't",
    "isnt",
    "wasn't",
    "wasnt",
    "can't",
    "cant",
    "cannot",
    "couldn't",
    "couldnt",
    "wouldn't",
    "wouldnt",
    "won't",
    "wont"
}

# Remove normal stopwords but preserve negation words
custom_stopwords = stop_words.difference(
    important_negation_words
)

# Remove these generic words
custom_stopwords.update({
    "product",
    "item"
})

lemmatizer = WordNetLemmatizer()


# ============================================================
# EMOJI PROCESSING
# ============================================================

def process_emojis(text):

    text = emoji.demojize(
        text,
        delimiters=(" ", " ")
    )

    replacements = {

        # Positive
        "smiling_face_with_heart_eyes": "love",
        "red_heart": "love",
        "heart": "love",
        "smiling_face": "happy",
        "grinning_face": "happy",
        "grinning_face_with_smiling_eyes": "happy",
        "thumbs_up": "good",
        "fire": "great",
        "star": "good",
        "sparkles": "great",
        "party_popper": "great",
        "clapping_hands": "good",

        # Negative
        "angry_face": "angry",
        "crying_face": "sad",
        "loudly_crying_face": "sad",
        "sad_but_relieved_face": "sad",
        "thumbs_down": "bad",
        "broken_heart": "bad",
        "disappointed_face": "disappointed",
        "face_with_symbols_on_mouth": "angry",
        "pensive_face": "sad",

        # Neutral
        "thinking_face": "thinking",
        "neutral_face": "neutral"
    }

    for key, value in replacements.items():
        text = text.replace(
            key,
            value
        )

    return text


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def preprocess(text):

    text = str(text).lower()

    # Convert emojis
    text = process_emojis(text)

    # Normalize apostrophes
    text = text.replace("’", "'")

    # Expand common contractions
    contractions = {

        "don't": "do not",
        "didn't": "did not",
        "doesn't": "does not",
        "isn't": "is not",
        "wasn't": "was not",
        "weren't": "were not",
        "can't": "cannot",
        "couldn't": "could not",
        "wouldn't": "would not",
        "won't": "will not",
        "shouldn't": "should not",
        "haven't": "have not",
        "hasn't": "has not",
        "hadn't": "had not"
    }

    for old, new in contractions.items():
        text = text.replace(
            old,
            new
        )


    # ========================================================
    # SARCASM
    # ========================================================

    sarcasm_patterns = [

        "yeah right",
        "wow great",
        "just perfect",
        "nice job",
        "great job",
        "thanks for nothing",
        "exactly what i needed",
        "what a surprise"
    ]

    for pattern in sarcasm_patterns:

        if pattern in text:

            text += " negative"


    # ========================================================
    # CONTRAST HANDLING
    # ========================================================

    # Give more importance to the part after:
    #
    # but
    # however
    # although
    # though

    contrast_words = [
        " but ",
        " however ",
        " although ",
        " though "
    ]

    for contrast in contrast_words:

        if contrast in text:

            parts = text.split(
                contrast
            )

            if len(parts) > 1:

                second_part = parts[-1]

                text = (
                    text
                    + " "
                    + second_part
                    + " "
                    + second_part
                )

                break


    # ========================================================
    # REMOVE PUNCTUATION
    # ========================================================

    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )


    # ========================================================
    # REMOVE NUMBERS
    # ========================================================

    text = re.sub(
        r"\d+",
        "",
        text
    )


    # ========================================================
    # TOKENIZATION
    # ========================================================

    tokens = nltk.word_tokenize(
        text
    )


    # ========================================================
    # NEGATION HANDLING
    # ========================================================

    negation_words = {
        "not",
        "no",
        "never",
        "cannot"
    }

    new_tokens = []

    negate = 0

    for word in tokens:

        # Start negation
        if word in negation_words:

            new_tokens.append(
                word
            )

            negate = 3

            continue


        # Add not_ prefix
        if negate > 0:

            neg_word = "not_" + word

            new_tokens.append(
                neg_word
            )

            # Boost importance
            new_tokens.append(
                neg_word
            )

            negate -= 1

        else:

            new_tokens.append(
                word
            )


    # ========================================================
    # REMOVE STOPWORDS
    # ========================================================

    tokens = [

        word

        for word in new_tokens

        if word not in custom_stopwords
    ]


    # ========================================================
    # LEMMATIZATION
    # ========================================================

    tokens = [

        lemmatizer.lemmatize(word)

        for word in tokens
    ]


    return " ".join(tokens)


# ============================================================
# PREPROCESS DATASET
# ============================================================

print("\n======================================")
print("        PREPROCESSING REVIEWS")
print("======================================")

df["cleaned"] = df["review"].apply(
    preprocess
)

print("Preprocessing completed.")


# ============================================================
# INPUT AND TARGET
# ============================================================

X_text = df["cleaned"]

y = df["sentiment"]


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train_text, X_test_text, y_train, y_test = train_test_split(

    X_text,
    y,

    test_size=0.30,

    random_state=42,

    stratify=y
)


print("\n======================================")
print("          TRAIN TEST SPLIT")
print("======================================")

print(
    "Training reviews:",
    len(X_train_text)
)

print(
    "Testing reviews:",
    len(X_test_text)
)


# ============================================================
# TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(

    ngram_range=(1, 2),

    max_features=5000,

    min_df=2,

    max_df=0.8,

    sublinear_tf=True
)


# Fit only on training data
X_train = vectorizer.fit_transform(
    X_train_text
)

# Transform test data
X_test = vectorizer.transform(
    X_test_text
)


print("\n======================================")
print("         TF-IDF FEATURES")
print("======================================")

print(
    "Number of features:",
    X_train.shape[1]
)


# ============================================================
# MACHINE LEARNING MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000
        ),

    "Naive Bayes":
        MultinomialNB(),

    "SVM":
        LinearSVC(
            C=0.5
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
}


# ============================================================
# CROSS VALIDATION
# ============================================================

print("\n======================================")
print("       CROSS VALIDATION RESULTS")
print("======================================")


cv_results = {}

cv_std_results = {}


for name, model in models.items():

    print(
        f"\nRunning 5-Fold CV for {name}..."
    )

    scores = cross_val_score(

        model,

        X_train,

        y_train,

        cv=5,

        scoring="accuracy"
    )

    mean_score = scores.mean()

    std_score = scores.std()

    cv_results[name] = mean_score

    cv_std_results[name] = std_score

    print(
        "Fold Scores:",
        np.round(
            scores,
            4
        )
    )

    print(
        f"Average Accuracy: "
        f"{mean_score:.4f}"
    )

    print(
        f"Standard Deviation: "
        f"{std_score:.4f}"
    )


# ============================================================
# TRAIN AND EVALUATE MODELS
# ============================================================

results = {}

trained_models = {}


print("\n======================================")
print("       MODEL TRAINING RESULTS")
print("======================================")


for name, model in models.items():

    print(
        f"\nTraining {name}..."
    )


    # Train
    model.fit(
        X_train,
        y_train
    )


    # Save trained model
    trained_models[name] = model


    # Predict
    y_pred = model.predict(
        X_test
    )


    # Accuracy
    accuracy = accuracy_score(
        y_test,
        y_pred
    )


    results[name] = accuracy


    print(
        f"\n{name} Accuracy: "
        f"{accuracy:.4f}"
    )


    # Classification report
    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            y_pred
        )
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(

        y_test,

        y_pred,

        labels=sorted(
            y.unique()
        )
    )


    plt.figure(
        figsize=(6, 5)
    )


    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        xticklabels=sorted(
            y.unique()
        ),

        yticklabels=sorted(
            y.unique()
        )
    )


    plt.title(
        f"{name} Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Sentiment"
    )

    plt.ylabel(
        "Actual Sentiment"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# MODEL COMPARISON
# ============================================================

print("\n======================================")
print("          MODEL COMPARISON")
print("======================================")


for name in models.keys():

    print(
        f"\n{name}:"
    )

    print(
        f"  Test Accuracy: "
        f"{results[name]:.4f}"
    )

    print(
        f"  CV Accuracy:   "
        f"{cv_results[name]:.4f}"
    )


# ============================================================
# TEST ACCURACY GRAPH
# ============================================================

plt.figure(
    figsize=(10, 5)
)

sns.barplot(

    x=list(results.keys()),

    y=list(results.values())
)

plt.title(
    "Model Test Accuracy Comparison"
)

plt.xlabel(
    "Machine Learning Model"
)

plt.ylabel(
    "Accuracy"
)

plt.ylim(
    0,
    1
)

plt.xticks(
    rotation=20
)

plt.tight_layout()

plt.show()


# ============================================================
# CROSS VALIDATION GRAPH
# ============================================================

plt.figure(
    figsize=(10, 5)
)

sns.barplot(

    x=list(cv_results.keys()),

    y=list(cv_results.values())
)

plt.title(
    "5-Fold Cross Validation Accuracy"
)

plt.xlabel(
    "Machine Learning Model"
)

plt.ylabel(
    "CV Accuracy"
)

plt.ylim(
    0,
    1
)

plt.xticks(
    rotation=20
)

plt.tight_layout()

plt.show()


# ============================================================
# SELECT BEST MODEL
# ============================================================

# Select based on cross-validation accuracy

best_model_name = max(
    cv_results,
    key=cv_results.get
)

best_model = trained_models[
    best_model_name
]

best_cv_accuracy = cv_results[
    best_model_name
]

best_test_accuracy = results[
    best_model_name
]


print("\n======================================")
print("             BEST MODEL")
print("======================================")

print(
    "Best Model:",
    best_model_name
)

print(
    f"Cross Validation Accuracy: "
    f"{best_cv_accuracy:.4f}"
)

print(
    f"Test Accuracy: "
    f"{best_test_accuracy:.4f}"
)


# ============================================================
# HYBRID SENTIMENT ANALYZER
# ============================================================

def predict_sentiment(text):

    # ========================================================
    # BASIC NORMALIZATION
    # ========================================================

    original_text = str(text).strip()

    text_lower = original_text.lower()

    # Normalize apostrophe
    text_lower = text_lower.replace(
        "’",
        "'"
    )

    # Remove repeated spaces
    text_lower = re.sub(
        r"\s+",
        " ",
        text_lower
    ).strip()


    # ========================================================
    # EMPTY INPUT
    # ========================================================

    if not text_lower:

        return "neutral"


    # ========================================================
    # PURE NEUTRAL / SHORT EXPRESSIONS
    # ========================================================

    neutral_exact = {

        "ok",
        "okay",
        "ok ok",
        "okay okay",
        "ok ok ok",

        "fine",
        "fine fine",

        "alright",
        "all right",

        "so so",
        "so-so",

        "average",

        "normal",

        "acceptable",

        "nothing special",

        "nothing much",

        "not sure",

        "maybe",

        "perhaps",

        "no idea",

        "no opinion",

        "i dont know",

        "i don't know",

        "dont know",

        "don't know",

        "idk",

        "meh",

        "whatever",

        "it is okay",

        "it's okay",

        "it is ok",

        "it's ok",

        "its okay",

        "its ok",

        "works as expected",

        "works as it should",

        "does what it should",

        "nothing to say",

        "no comment"
    }


    if text_lower in neutral_exact:

        return "neutral"


    # ========================================================
    # REPEATED OK / FINE / NEUTRAL WORDS
    # ========================================================

    neutral_repeat_pattern = re.fullmatch(

        r"(ok|okay|fine|alright|average)"
        r"(\s+\1){1,4}",

        text_lower
    )

    if neutral_repeat_pattern:

        return "neutral"


    # ========================================================
    # VERY SHORT NEUTRAL TEXT
    # ========================================================

    words = text_lower.split()

    if len(words) <= 3:

        neutral_phrases = [

            "not sure",

            "maybe yes",

            "maybe no",

            "could be",

            "i guess",

            "probably",

            "not really",

            "kind of",

            "sort of"
        ]

        if text_lower in neutral_phrases:

            return "neutral"


    # ========================================================
    # STRONG POSITIVE RULES
    # ========================================================

    strong_positive = [

        "i love it",
        "i absolutely love it",
        "i really love it",

        "love this",
        "love it",

        "excellent",
        "excellent product",
        "excellent quality",

        "perfect",
        "absolutely perfect",

        "amazing",
        "absolutely amazing",

        "fantastic",
        "absolutely fantastic",

        "wonderful",

        "outstanding",

        "brilliant",

        "highly recommend",

        "strongly recommend",

        "best product",

        "best purchase",

        "very satisfied",

        "extremely satisfied",

        "really happy",

        "very happy",

        "works perfectly",

        "works great",

        "works very well",

        "great quality",

        "very good",

        "really good",

        "so good",

        "super good",

        "awesome",

        "great"
    ]


    for phrase in strong_positive:

        if phrase in text_lower:

            return "positive"


    # ========================================================
    # STRONG NEGATIVE RULES
    # ========================================================

    strong_negative = [

        "i hate it",
        "i absolutely hate it",
        "i really hate it",

        "hate this",
        "hate it",

        "terrible",

        "absolutely terrible",

        "horrible",

        "absolutely horrible",

        "awful",

        "worst",

        "worst product",

        "worst purchase",

        "useless",

        "completely useless",

        "very disappointed",

        "extremely disappointed",

        "really disappointed",

        "very bad",

        "really bad",

        "so bad",

        "super bad",

        "poor quality",

        "very poor quality",

        "does not work",

        "doesn't work",

        "did not work",

        "didn't work",

        "stopped working",

        "not working",

        "cannot use",

        "can't use",

        "cannot use it",

        "can't use it",

        "very unhappy",

        "really unhappy",

        "do not recommend",

        "don't recommend"
    ]


    for phrase in strong_negative:

        if phrase in text_lower:

            return "negative"


    # ========================================================
    # NEGATION RULES
    # ========================================================

    positive_words = [

        "good",
        "great",
        "excellent",
        "amazing",
        "awesome",
        "perfect",
        "love",
        "happy",
        "fantastic",
        "wonderful"
    ]

    negative_words = [

        "bad",
        "terrible",
        "horrible",
        "awful",
        "hate",
        "worst",
        "poor",
        "disappointed",
        "useless"
    ]


    # "not bad" = positive
    if re.search(
        r"\b(not|never)\s+(very\s+|really\s+|too\s+)?bad\b",
        text_lower
    ):

        return "positive"


    # "not terrible" = positive
    if re.search(
        r"\bnot\s+(very\s+|really\s+)?(bad|terrible|horrible|awful)\b",
        text_lower
    ):

        return "positive"


    # "not good" = negative
    if re.search(
        r"\b(not|never)\s+(very\s+|really\s+|too\s+)?good\b",
        text_lower
    ):

        return "negative"


    # "not great" = negative
    if re.search(
        r"\bnot\s+(very\s+|really\s+)?great\b",
        text_lower
    ):

        return "negative"


    # "not happy" = negative
    if re.search(
        r"\bnot\s+(very\s+|really\s+)?happy\b",
        text_lower
    ):

        return "negative"


    # "not satisfied" = negative
    if re.search(
        r"\bnot\s+(very\s+|really\s+)?satisfied\b",
        text_lower
    ):

        return "negative"


    # ========================================================
    # "DON'T LOVE / DON'T HATE" = NEUTRAL
    # ========================================================

    neutral_emotion_patterns = [

        r"\bdo not love\b",
        r"\bdon't love\b",
        r"\bdont love\b",

        r"\bdo not hate\b",
        r"\bdon't hate\b",
        r"\bdont hate\b",

        r"\bcannot love\b",
        r"\bcan't love\b",
        r"\bcant love\b",

        r"\bcannot hate\b",
        r"\bcan't hate\b",
        r"\bcant hate\b"
    ]


    for pattern in neutral_emotion_patterns:

        if re.search(
            pattern,
            text_lower
        ):

            return "neutral"


    # ========================================================
    # MIXED SENTIMENT
    # ========================================================

    has_positive = any(
        word in text_lower
        for word in positive_words
    )

    has_negative = any(
        word in text_lower
        for word in negative_words
    )


    # If the sentence clearly contains both
    # positive and negative sentiment,
    # use the part after "but".

    if has_positive and has_negative:

        if " but " in text_lower:

            after_but = text_lower.split(
                " but "
            )[-1]

            after_but_positive = any(
                word in after_but
                for word in positive_words
            )

            after_but_negative = any(
                word in after_but
                for word in negative_words
            )

            if after_but_negative:

                return "negative"

            if after_but_positive:

                return "positive"


    # ========================================================
    # EMOJI-ONLY INPUT
    # ========================================================

    emoji_positive = [
        "❤️",
        "❤",
        "😍",
        "😊",
        "😀",
        "😁",
        "👍",
        "🔥",
        "🥰",
        "😃",
        "😄"
    ]

    emoji_negative = [
        "😡",
        "😠",
        "😢",
        "😭",
        "👎",
        "💔",
        "😞",
        "😔",
        "😤",
        "🤬"
    ]


    if (
        any(
            e in original_text
            for e in emoji_positive
        )
        and
        not any(
            e in original_text
            for e in emoji_negative
        )
    ):

        return "positive"


    if (
        any(
            e in original_text
            for e in emoji_negative
        )
        and
        not any(
            e in original_text
            for e in emoji_positive
        )
    ):

        return "negative"


    # ========================================================
    # MACHINE LEARNING FALLBACK
    # ========================================================

    processed_text = preprocess(
        original_text
    )


    # If preprocessing produces nothing useful,
    # classify as neutral.

    if not processed_text.strip():

        return "neutral"


    text_vector = vectorizer.transform(
        [processed_text]
    )


    prediction = best_model.predict(
        text_vector
    )


    return prediction[0]


# ============================================================
# INTERACTIVE SENTIMENT ANALYSIS
# ============================================================

print("\n======================================")
print("       SENTIMENT ANALYSIS SYSTEM")
print("======================================")

print(
    "\nEnter a review to predict its sentiment."
)

print(
    "Type 'exit' to close the program."
)


while True:

    user_input = input(
        "\nEnter a review: "
    )


    # Exit
    if user_input.lower().strip() == "exit":

        print(
            "\nProgram terminated."
        )

        break


    # Empty input
    if not user_input.strip():

        print(
            "Please enter a review."
        )

        continue


    # Predict
    prediction = predict_sentiment(
        user_input
    )


    print(
        "Predicted Sentiment:",
        prediction
    )
