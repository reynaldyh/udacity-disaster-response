import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import re

from typing import Any, Union, Dict, Tuple, List
import pickle
import sys

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """Extract starting VB (Verb Base Form) and VBP
    (Verb non-3rd person singular present form) for additional feature

    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ["VB", "VBP"] or first_word == "RT":
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def tokenize(text: str) -> List[str]:
    """Tokenize setences to word tokens

    Args:
        text (str): setences

    Returns:
        List[str]: word tokens
    """
    url_regex = (
        "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """clean irrelevant data

    Args:
        df (pd.DataFrame): disaster response dataframe

    Returns:
        pd.DataFrame: cleaned disaster response dataframe
    """
    # Replace class 2 to class 1 for "related" category
    df["related"] = df["related"].replace(2, 1)

    # Remove unused columns
    df = df.drop(columns=["child_alone", "original", "id", "genre"])

    # Remove rows if all values in that row is np.nan
    return df.dropna(axis=0, how="all")


def load_data(database_filepath: str) -> Tuple[pd.Series, pd.Series, List[str]]:
    """load dataset from given db

    Args:
        database_filepath (str): disaster response SQL's database filepath

    Returns:
        Tuple[pd.Series, pd.Series, List[str]]: splitted features, target, and target names
    """
    engine = create_engine(f"sqlite:///{database_filepath}")

    database_filename = database_filepath.split("/")[-1]
    table_name = database_filename.split(".db")[0]

    df = pd.read_sql_table(table_name, engine)
    df = clean(df)

    X = df["message"]
    Y = df[df.columns[4:]]
    return X, Y, Y.columns.values


def build_model() -> GridSearchCV:
    """model initialization with auto tone hyperparameters using GridSearch

    Returns:
        Pipeline: model pipeline
    """
    pipeline = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "text_pipeline",
                            Pipeline(
                                [
                                    ("vect", CountVectorizer(tokenizer=tokenize)),
                                    ("tfidf", TfidfTransformer()),
                                ]
                            ),
                        ),
                        ("starting_verb", StartingVerbExtractor()),
                    ]
                ),
            ),
            (
                "clf",
                MultiOutputClassifier(AdaBoostClassifier()),
            ),
        ]
    )

    parameters = {
        "features__text_pipeline__vect__ngram_range": ((1, 1), (1, 2)),
        "features__text_pipeline__vect__max_df": (0.5, 0.75, 1.0),
        "features__text_pipeline__vect__max_features": (None, 5000, 10000),
        "features__text_pipeline__tfidf__use_idf": (True, False),
        "clf__estimator__n_estimators": [5, 10, 20, 40],
        "clf__estimator__learning_rate": [0.01, 0.05, 0.1, 0.2, 0.5],
        "features__transformer_weights": (
            {"text_pipeline": 1, "starting_verb": 0.5},
            {"text_pipeline": 0.5, "starting_verb": 1},
            {"text_pipeline": 0.8, "starting_verb": 1},
        ),
    }

    return GridSearchCV(pipeline, param_grid=parameters, verbose=10)


def evaluate_model(
    model: Pipeline, X_test: pd.Series, Y_test: pd.Series, category_names: List[str]
):
    """Evaluate model with F1-score, recall, precision metrics

    Args:
        model (Pipeline): disaster response model
        X_test (pd.Series): test features
        Y_test (pd.Series): test target
        category_names (List[str]): target names
    """
    y_pred_test = model.predict(X_test)
    print(classification_report(Y_test, y_pred_test, target_names=category_names))


def save_model(model: Pipeline, model_filepath: str):
    """[dumping model to pickle format

    Args:
        model (Pipeline): disaster response model
        model_filepath (str): filepath for pickle
    """
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    main()