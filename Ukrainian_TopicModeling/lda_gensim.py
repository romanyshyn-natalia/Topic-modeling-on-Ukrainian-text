from gensim.models import LdaMulticore, CoherenceModel
from pprint import pprint
import matplotlib.pyplot as plt
from gensim.matutils import corpus2csc
import pandas as pd
from scipy.sparse import csc_matrix
from sklearn.cluster import KMeans
import umap.umap_ as umap
from sklearn.manifold import TSNE
from bokeh.plotting import figure, show
import matplotlib.colors as mcolors
import numpy as np

from preprocessor import preprocess_normalized, preprocess_with_lemmatization
from vectorizer import vectorize_corpus_tf_idf, vectorize_corpus_word_to_vec

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
                             passes=1,
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
                             chunksize=10, passes=1, alpha=0.01, eta=0.9)
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
    plt.savefig('./resulting_plots/coherence_measure_graph_lda_lemmatized.png')


def get_vectorized_sparse_matrix(gensim_vectorized, dct, num_of_topics):
    lda_model = LdaMulticore(corpus=gensim_vectorized,
                             id2word=dct,
                             num_topics=num_of_topics,
                             random_state=SOME_FIXED_SEED,
                             chunksize=10,
                             passes=1,
                             alpha=0.01,
                             eta=0.9)
    df = pd.DataFrame(lda_model.get_document_topics(gensim_vectorized, minimum_probability=0.0, per_word_topics=False))
    doc_topic_df = pd.DataFrame([[k[1] for k in df.iloc[x]] for x in df.index])
    doc_topic_matrix = csc_matrix(doc_topic_df.values)
    return doc_topic_matrix, corpus2csc(gensim_vectorized).transpose()


def create_k_means_model(num_of_clusters, vectorized_sparse_matrix):
    """
    Create k-means clustering model using sklearn
    """
    km = KMeans(n_clusters=num_of_clusters, random_state=SOME_FIXED_SEED)
    km.fit(vectorized_sparse_matrix)
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
    plt.savefig('./resulting_plots/topics_clustering_graph_lda_lemmatized.png')


if __name__ == "__main__":
    clean_text = preprocess_with_lemmatization()

    # create baseline model for LDA topic modeling
    dictionary, term_doc_matrix = vectorize_corpus_word_to_vec(clean_text)
    base_model = create_gensim_lda_model(term_doc_matrix, dictionary, 5)

    # hyperparameter tuning
    # compute coherence score for different number of topics and plot graph with results
    # plot_graph(clean_text, 2, 21, 1)

    optimal_number_of_topics = 5
    #
    # # create final model with optimized number of topics and visualize with the help of clusters ans UMAP
    # x_topics, tf_idf_sparse = get_vectorized_sparse_matrix(term_doc_matrix, dictionary, optimal_number_of_topics)
    # cluster_labels = create_k_means_model(optimal_number_of_topics, tf_idf_sparse)
    # print(x_topics.shape, tf_idf_sparse.shape)
    # plot_clusters_with_topics(x_topics, cluster_labels)

    # Get topic weights and dominant topics ------------

    # n-1 rows each is a vector with i-1 positions, where n the number of documents
    # i the topic number and tmp[i] = probability of topic i
    topic_weights = []
    for row_list in base_model[term_doc_matrix]:
        tmp = np.zeros(5)
        for i, w in row_list:
            tmp[i] = w
        topic_weights.append(tmp)

    # Array of topic weights
    arr = pd.DataFrame(topic_weights).fillna(0).values
    cluster_labels = create_k_means_model(optimal_number_of_topics, corpus2csc(term_doc_matrix).transpose())
    plot_clusters_with_topics(arr, cluster_labels)
    print(arr.shape)

    # # # Keep the well separated points (optional)
    # # # arr = arr[np.amax(arr, axis=1) > 0.01]
    # #
    # # Dominant topic number in each doc
    # topic_num = np.argmax(arr, axis=1)
    #
    # # tSNE Dimension Reduction
    # tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
    # tsne_lda = tsne_model.fit_transform(arr)
    #
    #
    # n_topics = 4
    # mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
    # plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics),
    #               plot_width=900, plot_height=700)
    # plot.scatter(x=tsne_lda[:, 0], y=tsne_lda[:, 1], color=mycolors[topic_num])
    # show(plot)
