from gensim.models import LdaMulticore, CoherenceModel
from pprint import pprint
import matplotlib.pyplot as plt
from gensim.matutils import corpus2csc
import pandas as pd
from scipy.sparse import csc_matrix
from collections import Counter
import matplotlib.colors as m_colors
from preprocessor import preprocess_with_lemmatization
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
                             chunksize=10,
                             passes=3,
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
                             chunksize=10, passes=3, alpha=0.01, eta=0.9)
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
    plt.savefig('./resulting_plots/lda/coherence_measure_graph_lda_lemmatized.png')


def get_vectorized_sparse_matrix(gensim_vectorized, dct, num_of_topics):
    lda_model = LdaMulticore(corpus=gensim_vectorized,
                             id2word=dct,
                             num_topics=num_of_topics,
                             random_state=SOME_FIXED_SEED,
                             chunksize=10,
                             passes=3,
                             alpha=0.01,
                             eta=0.9)
    df_topics = pd.DataFrame(
        lda_model.get_document_topics(gensim_vectorized, minimum_probability=0.0, per_word_topics=False))
    doc_topic_df = pd.DataFrame([[k[1] for k in df_topics.iloc[x]] for x in df_topics.index])
    doc_topic_matrix = csc_matrix(doc_topic_df.values)
    return doc_topic_matrix, corpus2csc(gensim_vectorized).transpose()


if __name__ == "__main__":
    clean_text = preprocess_with_lemmatization()

    # create baseline model for LDA topic modeling
    dictionary, term_doc_matrix = vectorize_corpus_tf_idf(clean_text)
    base_model = create_gensim_lda_model(term_doc_matrix, dictionary, 4)

    # hyperparameter tuning
    # compute coherence score for different number of topics and plot graph with results
    # plot_graph(clean_text, 2, 17, 1)

    # optimal_number_of_topics = 15
    # optimized_model = create_gensim_lda_model(term_doc_matrix, dictionary, optimal_number_of_topics)

    topics = base_model.show_topics(formatted=False)
    data_flat = [w for w_list in clean_text for w in w_list]
    counter = Counter(data_flat)

    out = []
    for i, topic in topics:
        for word, weight in topic:
            out.append([word, i, weight, counter[word]])

    df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])

    # Plot Word Count and Weights of Topic Keywords
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=160)
    cols = [color for name, color in m_colors.TABLEAU_COLORS.items()]
    for i, ax in enumerate(axes.flatten()):
        ax.bar(x='word', height="word_count", data=df.loc[df.topic_id == i, :], color=cols[i], width=0.5, alpha=0.3,
               label='Word Count')
        ax_twin = ax.twinx()
        ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id == i, :], color=cols[i], width=0.2,
                    label='Weights')
        ax.set_ylabel('Word Count', color=cols[i])
        ax_twin.set_ylim(0, 0.008)
        ax.set_ylim(0, 800)
        ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
        ax.tick_params(axis='y', left=False)
        ax.set_xticklabels(df.loc[df.topic_id == i, 'word'], rotation=30, horizontalalignment='right')
        ax.legend(loc='upper left')
        ax_twin.legend(loc='upper right')

    fig.tight_layout(w_pad=2)
    fig.suptitle('Word Count and Importance of Topic Keywords LDA', fontsize=22, y=1.05)
    plt.savefig('./resulting_plots/lda/importance_of_topic_keywords_lda.png')
