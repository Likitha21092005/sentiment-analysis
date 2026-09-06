import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import string
import emoji

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)


# ============================================================
# NLTK DOWNLOAD
# ============================================================

@st.cache_resource
def download_nltk():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)


download_nltk()


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv("reviews.csv")

    df = df.drop_duplicates()

    df = df.dropna(subset=["review", "sentiment"])

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


df = load_data()


# ============================================================
# NLP SETUP
# ============================================================

stop_words = set(stopwords.words("english"))

custom_stopwords = stop_words.union({
    "product",
    "item"
})

lemmatizer = WordNetLemmatizer()


# ============================================================
# EMOJI PROCESSING
# ============================================================

def process_emojis(text):

    text = emoji.demojize(text)

    replacements = {

        "smiling_face_with_heart_eyes": "love",

        "smiling_face": "happy",

        "thumbs_up": "good",

        "fire": "great",

        "angry_face": "angry",

        "crying_face": "sad",

        "thumbs_down": "bad",

        "broken_heart": "bad",

        "red_heart": "love",

        "heart": "love",

        "grinning_face": "happy",

        "slightly_smiling_face": "happy",

        "beaming_face_with_smiling_eyes": "happy",

        "clapping_hands": "good",

        "party_popper": "great",

        "face_with_tears_of_joy": "happy",

        "loudly_crying_face": "sad",

        "pensive_face": "sad",

        "disappointed_face": "sad"
    }

    for key, value in replacements.items():

        text = text.replace(key, value)

    return text


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def preprocess(text):

    text = str(text).lower()

    # Emoji handling
    text = process_emojis(text)

    # Common contractions
    contractions = {

        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "can't": "cannot",
        "couldn't": "could not",
        "wouldn't": "would not",
        "shouldn't": "should not",
        "isn't": "is not",
        "wasn't": "was not",
        "aren't": "are not",
        "weren't": "were not",
        "won't": "will not",
        "haven't": "have not",
        "hasn't": "has not",
        "never": "never"
    }

    for key, value in contractions.items():

        text = text.replace(key, value)

    # Sarcasm patterns
    sarcasm_patterns = [

        "yeah right",
        "wow great",
        "just perfect",
        "nice job",
        "great job"
    ]

    for pattern in sarcasm_patterns:

        if pattern in text:

            text += " not"


    # Contrast handling
    contrast_words = [

        "but",
        "however",
        "although",
        "though"
    ]

    for word in contrast_words:

        if word in text:

            parts = text.split(word)

            if len(parts) > 1:

                text = parts[-1] + " " + parts[-1]

                break


    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Tokenization
    tokens = nltk.word_tokenize(text)

    # Negation handling
    negation_words = [

        "not",
        "no",
        "never",
        "dont",
        "didnt",
        "isnt",
        "wasnt",
        "cant",
        "cannot",
        "couldnt",
        "wouldnt",
        "shouldnt"
    ]

    new_tokens = []

    negate = 0

    for word in tokens:

        if word in negation_words:

            negate = 3

            continue

        if negate > 0:

            neg_word = "not_" + word

            new_tokens.append(neg_word)

            new_tokens.append(neg_word)

            negate -= 1

        else:

            new_tokens.append(word)


    # Remove stopwords
    tokens = [

        word
        for word in new_tokens
        if word not in custom_stopwords

    ]


    # Lemmatization
    tokens = [

        lemmatizer.lemmatize(word)
        for word in tokens

    ]

    return " ".join(tokens)


# ============================================================
# PREPROCESS DATASET
# ============================================================

@st.cache_data
def preprocess_dataset(data):

    data = data.copy()

    data["cleaned"] = data["review"].apply(preprocess)

    return data


df = preprocess_dataset(df)


# ============================================================
# TF-IDF + SVM TRAINING
# ============================================================

@st.cache_resource
def train_model(data):

    vectorizer = TfidfVectorizer(

        ngram_range=(1, 2),

        max_features=5000,

        min_df=2,

        max_df=0.8,

        sublinear_tf=True
    )


    X = vectorizer.fit_transform(data["cleaned"])

    y = data["sentiment"]


    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.30,

        random_state=42,

        stratify=y
    )


    model = LinearSVC(C=0.5)

    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)


    return model, vectorizer, accuracy


model, vectorizer, test_accuracy = train_model(df)


# ============================================================
# RULE-BASED SENTIMENT
# ============================================================

def rule_based_sentiment(text):

    text_lower = text.lower().strip()


    # --------------------------------------------------------
    # EMPTY INPUT
    # --------------------------------------------------------

    if not text_lower:

        return None


    # --------------------------------------------------------
    # NEUTRAL SHORT COMMENTS
    # --------------------------------------------------------

    neutral_exact = {

        "ok",
        "ok ok",
        "okay",
        "okay okay",
        "fine",
        "fine fine",
        "so so",
        "so-so",
        "average",
        "normal",
        "meh",
        "maybe",
        "perhaps",
        "not sure",
        "nothing special",
        "nothing much",
        "i dont know",
        "i don't know",
        "no idea",
        "works as expected",
        "it works as expected",
        "acceptable",
        "fair",
        "decent",
        "alright",
        "all right"
    }


    if text_lower in neutral_exact:

        return "neutral"


    # --------------------------------------------------------
    # POSITIVE SPECIAL EXPRESSIONS
    # --------------------------------------------------------

    positive_exact = {

        "great",
        "excellent",
        "amazing",
        "awesome",
        "fantastic",
        "perfect",
        "wonderful",
        "brilliant",
        "love it",
        "i love it",
        "i love this",
        "very good",
        "really good",
        "super good",
        "highly recommended",

        # Common playful positive expression
        "supercalifragilisticexpialidocious"
    }


    if text_lower in positive_exact:

        return "positive"


    # --------------------------------------------------------
    # NEGATIVE SPECIAL EXPRESSIONS
    # --------------------------------------------------------

    negative_exact = {

        "bad",
        "terrible",
        "awful",
        "horrible",
        "worst",
        "hate it",
        "i hate it",
        "very bad",
        "really bad",
        "poor",
        "poor quality",
        "not working",
        "does not work",
        "doesnt work",
        "not useful"
    }


    if text_lower in negative_exact:

        return "negative"


    # --------------------------------------------------------
    # NEUTRAL NEGATION EXPRESSIONS
    # --------------------------------------------------------

    neutral_patterns = [

        "dont hate",
        "do not hate",
        "don't hate",

        "dont love",
        "do not love",
        "don't love",

        "cant hate",
        "cannot hate",
        "can't hate",

        "cant love",
        "cannot love",
        "can't love"
    ]


    for pattern in neutral_patterns:

        if pattern in text_lower:

            return "neutral"


    # --------------------------------------------------------
    # POSITIVE NEGATION
    # --------------------------------------------------------

    positive_patterns = [

        "not bad",
        "not terrible",
        "not awful",
        "not horrible",
        "not the worst",
        "not too bad"
    ]


    for pattern in positive_patterns:

        if pattern in text_lower:

            return "positive"


    # --------------------------------------------------------
    # NEGATIVE NEGATION
    # --------------------------------------------------------

    negative_patterns = [

        "not good",
        "not great",
        "not excellent",
        "not happy",
        "not useful",
        "not working",
        "cannot use",
        "can't use",
        "cant use",
        "does not work",
        "doesn't work",
        "did not work",
        "didn't work",
        "never works"
    ]


    for pattern in negative_patterns:

        if pattern in text_lower:

            return "negative"


    # --------------------------------------------------------
    # REPEATED NEUTRAL WORDS
    # --------------------------------------------------------

    if re.fullmatch(
        r"(ok|okay|fine|alright|average)(\s+\1)+",
        text_lower
    ):

        return "neutral"


    # --------------------------------------------------------
    # ONLY PUNCTUATION / SYMBOLS
    # --------------------------------------------------------

    if re.fullmatch(
        r"[\W_]+",
        text_lower
    ):

        return "neutral"


    return None


# ============================================================
# FINAL PREDICTION
# ============================================================

def predict_sentiment(text):

    # First use rules
    rule_prediction = rule_based_sentiment(text)

    if rule_prediction is not None:

        return rule_prediction


    # Machine learning prediction
    processed = preprocess(text)

    if not processed.strip():

        return "neutral"


    vector = vectorizer.transform([processed])

    prediction = model.predict(vector)[0]

    return prediction


# ============================================================
# USER INTERFACE
# ============================================================

st.title("💬 Sentiment Analysis")

st.write(
    "Enter a review and the system will classify it as "
    "**Positive**, **Negative**, or **Neutral**."
)


# ------------------------------------------------------------
# MODEL INFORMATION
# ------------------------------------------------------------

with st.expander("📊 Model Information"):

    st.write(
        f"**Dataset size:** {len(df)} reviews"
    )

    st.write(
        f"**TF-IDF features:** {len(vectorizer.get_feature_names_out())}"
    )

    st.write(
        "**Model:** Linear Support Vector Machine (SVM)"
    )

    st.write(
        f"**Test Accuracy:** {test_accuracy:.2%}"
    )


# ============================================================
# INPUT
# ============================================================

review = st.text_area(

    "Enter your review:",

    placeholder="Example: I absolutely love this product!",

    height=150
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button(
    "🔍 Analyze Sentiment",
    use_container_width=True
):

    if not review.strip():

        st.warning(
            "Please enter a review first."
        )

    else:

        prediction = predict_sentiment(review)


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        if prediction == "positive":

            st.success(
                "😊 Positive Sentiment"
            )

        elif prediction == "negative":

            st.error(
                "😞 Negative Sentiment"
            )

        else:

            st.info(
                "😐 Neutral Sentiment"
            )


        st.write(
            f"**Your review:** {review}"
        )


# ============================================================
# EXAMPLES
# ============================================================

st.divider()

st.subheader("Try these examples")

examples = [

    "I absolutely love this product!",
    "The product stopped working after two days.",
    "ok ok",
    "The quality is amazing!",
    "This is not good.",
    "This is not bad.",
    "supercalifragilisticexpialidocious"
]


for example in examples:

    st.code(example)
