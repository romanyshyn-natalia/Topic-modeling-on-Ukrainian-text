from ua_gec import Corpus
import stanza
import pandas as pd
import string
import re
import nltk

# stanza.download('uk')

NLP = stanza.Pipeline(lang='uk', processors='tokenize, lemma', tokenize_no_ssplit=True)


def load_data():
    """
    Load test dataset from UA-GEC with lowercase text.
    """
    corpus = Corpus(partition="train")

    target_corpus = []
    for doc in corpus:
        target_corpus.append(doc.target.lower())
    return " ".join(target_corpus)


def remove_url(sample_list):
    """
    Remove URLs from a sample list.
    """
    regex = re.compile(r"//\S+")
    return [token for token in sample_list if not regex.match(token)]


def remove_digits(sample_list):
    """
    Remove numbers from a sample list.
    """
    return [x for x in sample_list if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())]


def remove_stopwords(sample_list):
    """
    Cleans dataset from stopwords of Ukrainian language.
    """
    stopwords_ua = pd.read_csv("./resources/stopwords_ua.txt", header=None, names=['stopwords'])
    stop_words_ua = list(stopwords_ua.stopwords)
    return [word for word in sample_list if word not in stop_words_ua and word != ""]


def remove_punctuation(sample_list):
    """
    Removes punctuations from text.
    """
    punctuation_marks = string.punctuation + '«' + '…' + '»' + '–' + '...' \
                        + '“' + '”' + '``' + "''" + '—' + '’' + '.'
    return [word for word in sample_list if word not in punctuation_marks]


def normalize(corpus):
    """
    Perform basic cleaning steps:
        * removing stopwords, punctuation, links, digits
        * split data by word tokens
    """
    tokens = nltk.word_tokenize(corpus)
    clean_stopwords = remove_stopwords(tokens)
    clean_punctuation = remove_punctuation(clean_stopwords)
    clean_numbers = remove_digits(clean_punctuation)
    return remove_url(clean_numbers)


def lemmatizer(tokenized):
    """
    Extract lemmas from words using stanza
    """
    doc_tokenized = NLP(tokenized)
    return [word.lemma for sentence in doc_tokenized.sentences for word in sentence.words]


def preprocess():
    """
    Combine all steps to obtained preprocessed data
    """
    loaded_corpus = load_data()
    normalized = normalize(loaded_corpus)
    return lemmatizer(normalized)


if __name__ == "__main__":
    preprocess()
