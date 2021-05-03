from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from preprocessor import preprocess
from sklearn.cluster import KMeans
from sklearn.utils.extmath import randomized_svd
import matplotlib.pyplot as plt
import umap.umap_ as umap

clean_data = preprocess()

vectorizer = TfidfVectorizer(max_features=10000, max_df=0.5, use_idf=True, ngram_range=(1, 3))

tfidf_train_sparse = vectorizer.fit_transform(clean_data)
terms = vectorizer.get_feature_names()

# tfidf_train_df = pd.DataFrame(tfidf_train_sparse.toarray(), columns=vectorizer.get_feature_names())
# tfidf_train_df.head()

# lsa_obj = TruncatedSVD(n_components=20, n_iter=100, random_state=42)
# tfidf_lsa_data = lsa_obj.fit_transform(tfidf_train_df)
# Sigma = lsa_obj.singular_values_
# V_T = lsa_obj.components_.T

km = KMeans(n_clusters=10)
km.fit(tfidf_train_sparse)
clusters = km.labels_.tolist()

U, Sigma, VT = randomized_svd(tfidf_train_sparse, n_components=20, n_iter=100,
                              random_state=122)
# printing the concepts
for i, comp in enumerate(VT):
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:7]
    print("Concept " + str(i) + ": ")
    for t in sorted_terms:
        print(t[0])
    print(" ")

X_topics = U * Sigma
embedding = umap.UMAP(n_neighbors=100, min_dist=0.5, random_state=12).fit_transform(X_topics)
plt.figure(figsize=(7, 5))
plt.scatter(embedding[:, 0], embedding[:, 1],
            c=clusters,
            s=10,  # size
            edgecolor='none'
            )
plt.show()

# sns.barplot(x=list(range(len(Sigma))), y=Sigma)
# plt.title('Singular values')
# plt.xlabel('latent components')
# plt.ylabel('relative importance of each component')
# plt.show()
