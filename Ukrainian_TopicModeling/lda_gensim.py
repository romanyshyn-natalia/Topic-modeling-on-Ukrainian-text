from gensim.models import LdaMulticore, CoherenceModel
from pprint import pprint
import matplotlib.pyplot as plt
import pyLDAvis.gensim_models
import pyLDAvis

from preprocessor import preprocess_normalized, preprocess_with_lemmatization
from vectorizer import vectorize_corpus_tf_idf

SOME_FIXED_SEED = 42


def create_gensim_lda_model(document_term_matrix, corpus_dictionary, number_of_topics):
    """
    Create LDA model using gensim
    """
    lda_model = LdaMulticore(corpus=document_term_matrix,
                             id2word=corpus_dictionary,
                             num_topics=number_of_topics,
                             random_state=SOME_FIXED_SEED,
                             chunksize=100,
                             passes=10,
                             alpha=0.01,
                             eta=0.9)
    pprint(lda_model.print_topics())
    return lda_model


def compute_coherence_values(dct, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = LdaMulticore(corpus=corpus, id2word=dct, num_topics=num_topics, random_state=SOME_FIXED_SEED,
                             chunksize=100, passes=10, alpha=0.01, eta=0.9)
        model_list.append(model)
        coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dct, coherence='c_v')
        coherence_values.append(coherence_model.get_coherence())
        print(f"For {num_topics} topics: {coherence_model.get_coherence()} coherence score.")
    return model_list, coherence_values


def plot_graph(doc_clean, start, stop, step):
    dct, doc_term_matrix = vectorize_corpus_tf_idf(doc_clean)
    model_list, coherence_values = compute_coherence_values(dct, doc_term_matrix, doc_clean,
                                                            stop, start, step)
    # Show graph
    x = range(start, stop, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.legend("coherence_values", loc='best')
    plt.savefig('coherence_measure_graph_lda1.png')


if __name__ == "__main__":
    clean_text = preprocess_with_lemmatization()

    # create baseline model for LDA topic modeling
    dictionary, term_doc_matrix = vectorize_corpus_tf_idf(clean_text)
    base_model = create_gensim_lda_model(term_doc_matrix, dictionary, 10)

    # hyperparameter tuning
    # compute coherence score for different number of topics and plot graph with results
    plot_graph(clean_text, 2, 25, 1)
