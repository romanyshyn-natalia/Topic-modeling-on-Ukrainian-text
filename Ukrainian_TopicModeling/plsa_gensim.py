# Gensim in the name of the module refers to the Coherence Model from the gensim library
# pLSA itself is imported from a plsa module, because gensim does not provide pLSA

from plsa import Corpus, Pipeline, Visualize
from plsa.algorithms import PLSA
from plsa.preprocessors import tokenize
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import umap.umap_ as umap

from preprocessor import preprocess_normalized, preprocess_with_lemmatization


def build_plsa_coherence_graph(corpus, clean_text):
    # Testing to find the best number of topics
    plsa_coherence_results = []

    for number_of_topics in range(2, 13, 1):
        plsa = PLSA(corpus, number_of_topics, using_tf_idf)
        plsa_result = plsa.fit()

        result_for_coherence = []
        for topic in plsa_result.word_given_topic:
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


def perform_plsa(text, pipeline):
    corpus = Corpus(text, pipeline)

    plsa = PLSA(corpus, optimal_number_of_topics, using_tf_idf)
    result = plsa.fit()
    return result


def prepare_preprocessed_text(preprocessed_text):
    prepared_text = []
    for raw_text in preprocessed_text:
        prepared_text.append(" ".join(raw_text))
    return prepared_text


def create_k_means_model(num_of_clusters, sparse_matrix):
    """
    Create k-means clustering model using sklearn
    """
    km = KMeans(n_clusters=num_of_clusters, random_state=SOME_FIXED_SEED)
    km.fit(sparse_matrix)
    return km.labels_.tolist()


def plot_clusters_with_topics(topics_matrix, clusters):
    """
    Visualise topics using clusters
    """
    embedding = umap.UMAP(n_neighbors=100, min_dist=0.5, random_state=100).fit_transform(topics_matrix)
    plt.figure(figsize=(7, 5))
    plt.scatter(embedding[:, 0], embedding[:, 1],
                c=clusters,
                s=10,  # size
                edgecolor='none')
    plt.savefig('./resulting_plots/topics_clustering_plsa_normalized.png')


if __name__ == "__main__":
    # For the first time using: uncomment code for downloading ua corpus in the preprocessor module

    # Uncomment to use preprocessed text with lemmatization
    # text = preprocess_with_lemmatization()

    # Use preprocessed text without lemmatization
    text = preprocess_normalized()

    # # For that pLSA implementation, we will need to use each document not as a list, but as a string,
    # # so we apply additional preparation
    prepared_text = prepare_preprocessed_text(text)  # 142822 words
    #
    # # Performing pLSa
    # # TODO: uncomment
    optimal_number_of_topics = 7
    #
    using_tf_idf = True
    pipeline = Pipeline(tokenize)

    result = perform_plsa(prepared_text, pipeline)
    #
    # # Visualization with the help of clusters and UMAP
    SOME_FIXED_SEED = 42
    #
    # # compute (m ⨉ t) document-topic matrix
    # # V = corpus2dense(lsa_model[gensim_vectorized], len(lsa_model.projection.s)).T / lsa_model.projection.s
    # # # create topics matrix for clustering plot
    # # X_topics = V * lsa_model.projection.s
    # # # convert vertorized gensim matrix to sparse matrix
    # # X_topics, corpus2csc(gensim_vectorized).transpose()
    #
    # # x_topics, tf_idf_sparse = get_vectorized_sparse_matrix(term_doc_matrix, dictionary, optimal_number_of_topics)
    # # Creating sparse matrix
    # # sparse = []
    # # for topic in result.word_given_topic:
    # #     sparse.append([])
    # #     sparse[-1].append(len(sparse) - 1)
    # #     topic = sorted(topic, key=lambda x: abs(x[1]))
    # #     string = ''
    # #     for word in topic[-7:]:
    # #         string += f'{word[1]}*"{word[0]}" + '
    # #     string = string[:-3]
    # #     sparse[-1].append(string)
    #
    result_for_plot = []
    for topic in result.word_given_topic:
        result_for_plot.append([])
        for word_tuple in topic:
            result_for_plot[-1].append(word_tuple[1])

    plt.figure(figsize=(7, 5))
    # plt.scatter(result_for_plot, result_for_plot,
    #             s=optimal_number_of_topics,  # size
    #             edgecolor='none')
    plt.scatter(result_for_plot[:, 0], result_for_plot[:, 1],
                s=optimal_number_of_topics,  # size
                edgecolor='none')
    plt.show()
