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
    corpus_train = Corpus(partition="train")
    corpus_test = Corpus(partition="test")
    target_corpus = []
    for doc in corpus_train:
        target_corpus.append(doc.target.lower())
    for doc in corpus_test:
        target_corpus.append(doc.target.lower())
    return target_corpus


def remove_url(sample_list):
    """
    Remove URLs from a sample list.
    """
    regex = re.compile(r"//\S+")
    return [token for token in sample_list if not regex.match(token)]


def remove_www(sample_list):
    """
    Remove www. URLs from a sample list.
    """
    regex = re.compile(r"www.\S+")
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
                        + '“' + '”' + '``' + "''" + '—' + '’' + '.' + '....' \
                        + '//' + '-' + '…' + '⠀' + '…' + '-' + ',' + '/'
    return [word for word in sample_list if word not in punctuation_marks]


def normalize(corpus):
    """
    Perform basic cleaning steps:
        * removing stopwords, punctuation, links, digits
        * split data by word tokens
    """
    tokens = [nltk.word_tokenize(sentence) for sentence in corpus]
    clean_stopwords = [remove_stopwords(sentence_token) for sentence_token in tokens]
    clean_punctuation = [remove_punctuation(sentence_sample) for sentence_sample in clean_stopwords]
    clean_numbers = [remove_digits(sentence_sample) for sentence_sample in clean_punctuation]
    clean_url = [remove_url(sentence_sample) for sentence_sample in clean_numbers]
    return [remove_www(sentence_sample) for sentence_sample in clean_url]


def lemmatizer(tokenized):
    """
    Extract lemmas from words using stanza
    """
    result = []
    for sentence in tokenized:
        doc_tokenized = NLP(sentence)
        result.append([word.lemma for sentence in doc_tokenized.sentences for word in sentence.words])
    return result


def preprocess():
    """
    Combine all steps to obtained preprocessed data
    """
    loaded_corpus = load_data()
    normalized = normalize(loaded_corpus)
    lemmas = lemmatizer(normalized)
    return [" ".join(elem) for elem in lemmas]


if __name__ == "__main__":
    preprocess()
