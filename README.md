# Topic-modeling-on-Ukrainian-text
Linear algebra course project

## Description:
In this project we utilized available topic modeling methods — LSA, pLSA, LDA — on Ukrainian corpora, evaluated it with the help of coherence measures and achieved the best performance — 63% on TF-IDF features with the LSA algorithm.

## Pipeline:
1. Preprocessing: tokenization, stopwords, punctuation, hyperlinks and numbers removal. Also, we extracted lemmas from each words using stanza pretrained model for Ukrainian language and created two models for further comparison - with and without lemmatization.
2. To feed text data into algorithms we vectorized it with Bag of Words model and TF-IDF.
3. Utilized gensim implementation of LSA, pLSA and LDA.
4. Evaluated their performance using topic coherence score. 


## Quick Start
```
git clone https://github.com/romanyshyn-natalia/Topic-modeling-on-Ukrainian-text.git
cd Topic-modeling-on-Ukrainian-text
pip3 install -r requirements.txt --no-cache-dir
```

### Dataset:
The dataset which we have used is [UA-GEC](https://github.com/grammarly/ua-gec) corpus. This is a collection of texts written by ordinary people: essays, blog and social network posts, reviews, letters, etc., which are splitted in two parts - train and test, we used both to have bigger corpus. All data has  3 types of representations: in annotated format (annotated), original (source) and the corrected (target which we have used) versions of documents.

## Results:
![image](https://user-images.githubusercontent.com/57792587/118008713-01947d00-b356-11eb-962b-1b87f3d6ba50.png)
![image](https://user-images.githubusercontent.com/57792587/118008886-31dc1b80-b356-11eb-8b15-614b7fa00fdd.png)

From Plot 1 we see that the best model is LSA with 63% coherence score on TF-IDF not lemmatizated features. From Plot B we see that lemmatized features work better for LDA — 62% coherence score. 

### Credits:
* Natalia Romanyshyn
* Daria Omelkina
* Anna Korabliova

Ukrainian Catholic University, 2021
