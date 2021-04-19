from ua_gec import Corpus
import stanza
import pandas as pd
import string
import re
from gensim import corpora
from sklearn.feature_extraction.text import CountVectorizer

# stanza.download('uk')

NLP = stanza.Pipeline(lang='uk', processors='tokenize, lemma', tokenize_no_ssplit=True)


def load_data():
    corpus = Corpus(partition="test")
    target_corpus = []
    for doc in corpus:
        target_corpus.append(doc.target.lower())
    return " ".join(target_corpus)


# def tokenizer(corpus):
#     doc = NLP(corpus)
#     result = []
#     for sentence in doc.sentences:
#         for token in sentence.tokens:
#             result.append(token.text)
#     return result


def clean_text(text):
    stopwords_ua = pd.read_csv("./resources/stopwords_ua.txt", header=None, names=['stopwords'])
    stop_words_ua = list(stopwords_ua.stopwords)
    text = "".join([word for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [word for word in tokens if word not in stop_words_ua and word != ""]
    return text


def lemmatizer(tokenized):
    doc_tokenized = NLP(tokenized)
    return [word.lemma for sentence in doc_tokenized.sentences for word in sentence.words]


def preprocess():
    loaded = load_data()
    cleaned_data = clean_text(loaded)
    return lemmatizer(cleaned_data)


def prepare_corpus(doc_clean):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(cleaned)
    Y = vectorizer.get_feature_names()
    return pd.DataFrame(X.toarray(), columns=Y)


if __name__ == "__main__":
    cleaned = preprocess()
    print(len(cleaned))