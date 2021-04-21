import urllib.request

import pandas as pd
import os
import re
from wordcloud import WordCloud

import gensim
from gensim.utils import simple_preprocess
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
import gensim.corpora as corpora
from pprint import pprint


def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in texts]

if __name__ == '__main__':

    # Loading Data
    os.chdir('..')

    # Read data into papers
    papers = pd.read_csv('C://course_2//sem_2//linear_algebra//project//gensim_lda//data//NIPS Papers//papers.csv')
    papers.head()


    # Data Cleaning
    # Remove the columns
    papers = papers.drop(columns=['id', 'event_type', 'pdf_name'], axis=1).sample(100)
    papers.head()


    # Remove punctuation/lower casing
    # Remove punctuation
    papers['paper_text_processed'] = \
    papers['paper_text'].map(lambda x: re.sub('[,\.!?]', '', x))

    # Convert the titles to lowercase
    papers['paper_text_processed'] = \
    papers['paper_text_processed'].map(lambda x: x.lower())

    # Print out the first rows of papers
    papers['paper_text_processed'].head()


    # Exploratory Analysis
    # Join the different processed titles together.
    long_string = ','.join(list(papers['paper_text_processed'].values))

    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=1000, contour_width=3, contour_color='steelblue')

    # Generate a word cloud
    wordcloud.generate(long_string)

    # Visualize the word cloud
    wordcloud.to_image()


    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])


    data = papers.paper_text_processed.values.tolist()
    data_words = list(sent_to_words(data))

    # remove stop words
    data_words = remove_stopwords(data_words)

    print(data_words[:1][0][:30])



    # Create Dictionary
    id2word = corpora.Dictionary(data_words)

    # Create Corpus
    texts = data_words

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    # View
    print(corpus[:1][0][:30])



    # number of topics
    num_topics = 10

    # Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics)

    # Print the Keyword in the 10 topics
    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]