from gensim import corpora, models


def vectorize_corpus_word_to_vec(doc_clean):
    """
    Create term dictionary of our corpus and Converting list of documents into Document Term Matrix
    """
    dct = corpora.Dictionary(doc_clean)
    doc_term_matrix = [dct.doc2bow(doc) for doc in doc_clean]
    return dct, doc_term_matrix


def vectorize_corpus_tf_idf(clean_corpus):
    """
    Create term dictionary of our corpus and Converting list of documents into Document Term Matrix
    """
    dct = corpora.Dictionary(clean_corpus)
    doc_to_vec = [dct.doc2bow(text) for text in clean_corpus]
    tf_idf = models.TfidfModel(doc_to_vec)  # step 1 -- initialize a model
    doc_term_matrix_tfidf = tf_idf[doc_to_vec]
    return dct, doc_term_matrix_tfidf
