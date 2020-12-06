import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize,sent_tokenize
from nltk.stem import PorterStemmer,LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

from nltk import pos_tag
from nltk import ne_chunk
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from bs4 import BeautifulSoup 

import re
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

#evaluation
from sklearn.metrics import accuracy_score,roc_auc_score 
from sklearn.metrics import classification_report
from mlxtend.plotting import plot_confusion_matrix
from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
stop_words=set(nltk.corpus.stopwords.words('english'))

from tensorflow import keras
from tensorflow.keras.preprocessing.text import one_hot,Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Flatten ,Embedding,Input,CuDNNLSTM,LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import text_to_word_sequence
#gensim w2v
#word2vec
from gensim.models import Word2Vec
nltk.download('punkt')

rev_frame=pd.read_csv('Reviews.csv')
df=rev_frame.copy()
df.head()

df=df[['Text','Score']]
df['review']=df['Text']
df['rating']=df['Score']
df.drop(['Text','Score'],axis=1,inplace=True)
print(df.shape)
df.head()
print(df['rating'].isnull().sum())
df['review'].isnull().sum()
df.drop_duplicates(subset=['rating','review'],keep='first',inplace=True) 
print(df.shape)
df.head()
for review in df['review'][:5]:
    print(review+'\n'+'\n')

def mark_sentiment(rating):
  if(rating<=3):
    return 0
  else:
    return 1
df['sentiment']=df['rating'].apply(mark_sentiment)

df.drop(['rating'],axis=1,inplace=True)

df.head()

df['sentiment'].value_counts()


def clean_reviews(review):  
    
    # 1. Removing html tags
    review_text = BeautifulSoup(review).get_text()
    
    # 2. Retaining only alphabets.
    review_text = re.sub("[^a-zA-Z]"," ",review_text)
    
    # 3. Converting to lower case and splitting
    word_tokens= review_text.lower().split()
    
    # 4. Remove stopwords
    le=WordNetLemmatizer()
    stop_words= set(stopwords.words("english"))     
    word_tokens= [le.lemmatize(w) for w in word_tokens if not w in stop_words]
    
    cleaned_review=" ".join(word_tokens)
    return cleaned_review

pos_df=df.loc[df.sentiment==1,:][:5000]
neg_df=df.loc[df.sentiment==0,:][:5000]
pos_df.head()
neg_df.head()
df=pd.concat([pos_df,neg_df],ignore_index=True)
print(df.shape)
df = df.sample(frac=1).reset_index(drop=True)
print(df.shape)
df.head()

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
sentences=[]
sum=0
for review in df['review']:
  sents=tokenizer.tokenize(review.strip())
  sum+=len(sents)
  for sent in sents:
    cleaned_sent=clean_reviews(sent)
    sentences.append(cleaned_sent.split()) # can use word_tokenize also.
print(sum)
print(len(sentences))
for te in sentences[:5]:
  print(te,"\n")

import gensim
w2v_model=gensim.models.Word2Vec(sentences=sentences,size=300,window=10,min_count=1)
w2v_model.train(sentences,epochs=10,total_examples=len(sentences))
w2v_model.wv.get_vector('like')
vocab=w2v_model.wv.vocab
print("The total number of words are : ",len(vocab))
w2v_model.wv.most_similar('like')
w2v_model.wv.similarity('good','like')
print("The no of words :",len(vocab))
vocab=list(vocab.keys())
word_vec_dict={}
for word in vocab:
  word_vec_dict[word]=w2v_model.wv.get_vector(word)
print("The no of key-value pairs : ",len(word_vec_dict)) # should come equal to vocab size
df['clean_review']=df['review'].apply(clean_reviews)
maxi=-1
for i,rev in enumerate(df['clean_review']):
  tokens=rev.split()
  if(len(tokens)>maxi):
    maxi=len(tokens)
print(maxi)
tok = Tokenizer()
tok.fit_on_texts(df['clean_review'])
vocab_size = len(tok.word_index) + 1
encd_rev = tok.texts_to_sequences(df['clean_review'])

max_rev_len=1565  # max lenght of a review
vocab_size = len(tok.word_index) + 1  # total no of words
embed_dim=300 # embedding dimension as choosen in word2vec constructor
pad_rev= pad_sequences(encd_rev, maxlen=max_rev_len, padding='post')
pad_rev.shape
embed_matrix=np.zeros(shape=(vocab_size,embed_dim))
for word,i in tok.word_index.items():
  embed_vector=word_vec_dict.get(word)
  if embed_vector is not None:
    embed_matrix[i]=embed_vector

Y=keras.utils.to_categorical(df['sentiment'])
x_train,x_test,y_train,y_test=train_test_split(pad_rev,Y,test_size=0.20,random_state=42)
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Dropout
model=Sequential()
model.add(Embedding(input_dim=vocab_size,output_dim=embed_dim,input_length=max_rev_len,embeddings_initializer=Constant(embed_matrix)))

model.add(Flatten())
model.add(Dense(16,activation='relu'))
model.add(Dropout(0.50))
# model.add(Dense(16,activation='relu'))
# model.add(Dropout(0.20))
model.add(Dense(2,activation='sigmoid'))
model.summary()
model.compile(optimizer=keras.optimizers.RMSprop(lr=1e-3),loss='binary_crossentropy',metrics=['accuracy'])
epochs=5
batch_size=32
modelm=model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,validation_data=(x_test,y_test))
c=model.evaluate(x_test,y_test,verbose=2)
print(c)
print("saving model...")
model.save("amazon.h5")