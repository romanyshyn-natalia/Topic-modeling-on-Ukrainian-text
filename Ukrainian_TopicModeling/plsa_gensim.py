# Gensim in the name of the module refers to the Coherence Model from the gensim library
# pLSA itself is imported from a plsa module, because gensim does not provide pLSA

from plsa import Corpus, Pipeline
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
    x = range(2, 13, 1)
    plt.plot(x, [0.24862516871891016, 0.3460672466169208, 0.31760168499772407, 0.3285116590501592, 0.33832683377588246,
                 0.35241208177724, 0.38477830107102784, 0.3729613650152243, 0.3778253405113229, 0.3810417049391049,
                 0.38329968618732657])
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.legend("coherence_values", loc='best')
    plt.savefig('coherence_measure_graph_lemmatized.png')
    # # For the first time using: uncomment code for downloading ua corpus in the preprocessor module
    #
    # # Uncomment to use preprocessed text with lemmatization
    # # text = preprocess_with_lemmatization()
    #
    # # Use preprocessed text without lemmatization
    # text = preprocess_normalized()
    # dct = Dictionary(text[:2])
    # print(dct.keys())
    #
    # # For that pLSA implementation, we will need to use each document not as a list, but as a string,
    # # so we apply additional preparation
    # prepared_text = prepare_preprocessed_text(text)  # 142822 words
    #
    # # Performing pLSa
    # # TODO: uncomment
    # # optimal_number_of_topics = 7
    # optimal_number_of_topics = 2
    #
    # using_tf_idf = True
    # pipeline = Pipeline(tokenize)
    #
    # result = perform_plsa(prepared_text, pipeline)
    #
    # # Visualization with the help of clusters and UMAP
    # SOME_FIXED_SEED = 42
    # # x_topics, tf_idf_sparse = get_vectorized_sparse_matrix(term_doc_matrix, dictionary, optimal_number_of_topics)
    # # Creating sparse matrix
    # dct = Dictionary(prepared_text)
    # sparse = []
    # for topic in result.word_given_topic:
    #     sparse.append([])
    #     for word_tuple in topic:
    #         sparse[-1].append(word_tuple[0])
    #
    # # K-Means
    # cluster_labels = create_k_means_model(optimal_number_of_topics, sparse)
    # print(len(sparse), len(sparse[0]))
    # # plot_clusters_with_topics(x_topics, cluster_labels)
