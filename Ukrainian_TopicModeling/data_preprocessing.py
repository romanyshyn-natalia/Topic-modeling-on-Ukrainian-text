from ua_gec import Corpus
import stanza


# stanza.download('uk')

def load_data():
    corpus = Corpus(partition="test")
    target_corpus = []
    for doc in corpus:
        target_corpus.append(doc.target)
    return target_corpus


def lemmatizer(value):
    nlp = stanza.Pipeline('uk')
    doc = nlp(value)
    return [word.lemma for sent in doc.sentences for word in sent.words]


if __name__ == "__main__":
    sent = load_data()[2]
    print(lemmatizer(sent))