from plsa import Corpus, Pipeline
from plsa.pipeline import DEFAULT_PIPELINE
from plsa.algorithms import PLSA

csv_file = "data/Full-Economic-News-DFE-839861.csv"

pipeline = Pipeline(*DEFAULT_PIPELINE)

corpus = Corpus.from_csv(csv_file, pipeline)

n_topics = 5
plsa = PLSA(corpus, n_topics, True)

result = plsa.fit()
result = plsa.best_of(5)

print(result.topic)

new_doc = 'Hello! This is the federal humpty dumpty agency for state funding.'

topic_components, number_of_new_words, new_words = result.predict(new_doc)

print('Relative topic importance in new document:', topic_components)
print('Number of previously unseen words in new document:', number_of_new_words)
print('Previously unseen words in new document:', new_words)

print(result.word_given_topic[0][:10])

