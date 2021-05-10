# Gensim in the name of the module refers to the Coherence Model from the gensim library
# pLSA itself is imported from a plsa module, because gensim does not provide pLSA

from plsa import Corpus, Pipeline
from plsa.algorithms import PLSA
from plsa.preprocessors import tokenize
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
import matplotlib.pyplot as plt

from preprocessor import preprocess_normalized, preprocess_with_lemmatization


def build_coherence_graph():
    # Testing to find the best number of topics
    plsa_coherence_results = []

    for number_of_topics in range(2, 13, 1):
        plsa = PLSA(corpus, number_of_topics, using_tf_idf)
        result = plsa.fit()

        result_for_coherence = []
        for topic in result.word_given_topic:
            result_for_coherence.append([])
            for word_tuple in topic:
                result_for_coherence[-1].append(word_tuple[0])

        coherence_model = CoherenceModel(topics=result_for_coherence, texts=clean_text,
                                         dictionary=Dictionary(clean_text), coherence='c_v')
        plsa_coherence_results.append(coherence_model.get_coherence())

    # Show graph
    x = range(2, 13, 1)
    plt.plot(x, plsa_coherence_results)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.legend("coherence_values", loc='best')
    plt.savefig('plsa_coherence_measure_graph.png')


if __name__ == "__main__":
    # For the first time using: uncomment code for downloading ua corpus
    # in the preprocessor module

    # Uncomment to use preprocessed text with lemmatization
    # clean_text = preprocess_with_lemmatization()

    # Use preprocessed text without lemmatization
    clean_text = preprocess_normalized()  # 142822 words

    optimal_number_of_topics = 12
    using_tf_idf = True

    # For that pLSA implementation, we will need to use each document not as a list, but as a string
    plsa_text = []
    for text in clean_text:
        plsa_text.append(" ".join(text))

    # pLSa
    pipeline = Pipeline(tokenize)
    corpus = Corpus(plsa_text, pipeline)

    plsa = PLSA(corpus, optimal_number_of_topics, using_tf_idf)
    result = plsa.fit()

    result_for_coherence = []
    for topic in result.word_given_topic:
        result_for_coherence.append([])
        for word_tuple in topic:
            result_for_coherence[-1].append(word_tuple[0])

    coherence_model = CoherenceModel(topics=result_for_coherence, texts=clean_text,
                                     dictionary=Dictionary(clean_text), coherence='c_v')
