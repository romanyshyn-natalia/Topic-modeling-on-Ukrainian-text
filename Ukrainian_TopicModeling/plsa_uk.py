from plsa import Corpus, Pipeline
from plsa.pipeline import DEFAULT_PIPELINE
from plsa.algorithms import PLSA

from preprocessor import preprocess_normalized

if __name__ == "__main__":
    # For the first time using: uncomment code for downloading ua corpus
    # in the preprocessor module

    clean_text = preprocess_normalized()
    number_of_topics = 3
    using_tf_idf = True

    # pLSa
    pipeline = Pipeline(*DEFAULT_PIPELINE)
    corpus = Corpus(clean_text, pipeline)

    plsa = PLSA(corpus, number_of_topics, using_tf_idf)
    result1 = plsa.fit()

    result2 = plsa.best_of(5)

    print(result1.topic)
