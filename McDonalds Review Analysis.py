
# coding: utf-8

# In[23]:


import pandas as pd
#Read the data
df_review=pd.read_csv(r'C:\Users\AMIT\Desktop\Machine Learning\mcdonalds_yelp_sentiment_dfe.csv')
#append the positive and negative reviews
df_review.head()


# In[24]:


#Return the wordnet Object value corresponding to the POS tag
from nltk.corpus import wordnet

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
def clean_text(text):
    # Lower case text
    text = text.lower()
    # tokenizing text and removing puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove=ing words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # removing stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # removing empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # removing words with only one letter
    text = [t for t in text if len(t) > 1]
    # joining all
    text = " ".join(text)
    return(text)

# clean text data
df_review["review_clean"] = df_review["review"].apply(lambda x: clean_text(x))


# In[25]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
df_review["sentiments"] = df_review["review"].apply(lambda x: sid.polarity_scores(x))
df_review = pd.concat([df_review.drop(['sentiments'], axis=1), df_review['sentiments'].apply(pd.Series)], axis=1)


# In[26]:


# adding number of characters column
df_review["nb_chars"] = df_review["review"].apply(lambda x: len(x))

# adding number of words column
df_review["nb_words"] = df_review["review"].apply(lambda x: len(x.split(" ")))


# In[27]:


# createing doc2vec vector columns
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(df_review["review_clean"].apply(lambda x: x.split(" ")))]

# training a Doc2Vec model with our text data
model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

# transforming each document into a vector data
df_doc2vec = df_review["review_clean"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
df_doc2vec.columns = ["doc2vec_vector_" + str(x) for x in df_doc2vec.columns]
df_review = pd.concat([df_review, df_doc2vec], axis=1)


# In[28]:


# add tf-idfs columns
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df = 10)
tfidf_result = tfidf.fit_transform(df_review["review_clean"]).toarray()
df_tfidf = pd.DataFrame(tfidf_result, columns = tfidf.get_feature_names())
df_tfidf.columns = ["word_" + str(x) for x in df_tfidf.columns]
df_tfidf.index = df_review.index
df_review = pd.concat([df_review, df_tfidf], axis=1)


# In[29]:


# wordcloud function
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color = 'white',
        max_words = 200,
        max_font_size = 40, 
        scale = 3,
        random_state = 42
    ).generate(str(data))

    fig = plt.figure(1, figsize = (20, 20))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize = 20)
        fig.subplots_adjust(top = 2.3)

    plt.imshow(wordcloud)
    plt.show()
    
# print wordcloud
show_wordcloud(df_review["review"])

