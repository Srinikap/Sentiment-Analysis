#!/usr/bin/env python
# coding: utf-8

# **Mounting google drive**

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# **Import Packages**

# In[ ]:


# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# to make this notebook's output identical/stable across runs
np.random.seed(7)

# set up numpy to display precision upto 3 decimal places and suppresses the use of scientific notation for small numbers
#np.set_printoptions(precision=3, suppress=True)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

pd.set_option('display.max_columns', None) # show all columns in a Pandas DataFrame
pd.set_option('display.max_rows', None)
pd.set_option('precision', 3)

# To plot pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
mpl.rcParams['savefig.dpi'] = 80
mpl.rcParams['figure.dpi'] = 80
plt.rcParams['image.cmap'] = "gray"

# Where to save the data and figures
PROJECT_ROOT_DIR = "."
IMAGES_DIR = "images"
DATA_DIR = "data"
MODELS_DIR = "models"
SUB_DIR = "ham-spam"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, IMAGES_DIR, SUB_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT_DIR, DATA_DIR, SUB_DIR)
MODELS_PATH = os.path.join(PROJECT_ROOT_DIR, MODELS_DIR, SUB_DIR)
os.makedirs(IMAGES_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)

# Function for saving figures
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Function for saving data downloaded from a URL
from six.moves import urllib
def save_data(file_url, file_name):
    path = os.path.join(DATA_PATH, file_name)
    print("Saving data file", file_name)
    urllib.request.urlretrieve(file_url, path)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")
#warnings.filterwarnings(action='ignore', category=DeprecationWarning, module='sklearn')
#warnings.filterwarnings(action='ignore', category=FutureWarning, module='sklearn')


# **Loading the data set**

# In[ ]:


get_ipython().run_line_magic('pwd', '')
from sklearn.datasets import load_files
SMS = pd.read_csv("/content/drive/My Drive/spam.csv", encoding='latin-1')


# In[ ]:


SMS.head()


# In[ ]:


print('Number of spam messages:', SMS[SMS['v1'] == 'spam']['v1'].count())
print('Number of ham messages:', SMS[SMS['v1'] == 'ham']['v1'].count())


# In[ ]:


SMS.info()


# In[ ]:


SMS.shape


# In[ ]:


SMS.columns


# **Data Preprocessing**

# **Dropping null values**

# In[ ]:


SMS.isnull().all()


# In[ ]:


SMS.isnull()


# In[ ]:


SMS.isnull().any()


# In[ ]:


SMS = SMS.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)


# In[ ]:


SMS


# **Checking for the duplicate values in the data, dropping them**

# In[ ]:


SMS.duplicated().sum()


# In[ ]:


SMS = SMS.drop_duplicates(keep='first')
SMS.duplicated().sum()


# **Renaming the columns**

# In[ ]:


SMS = SMS.rename({'v1':'Class','v2':'Text'},axis=1)
SMS


# In[ ]:


import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = stopwords.words('english')
print(STOPWORDS)


# In[ ]:


# Function to remove stop words
def remove_stopwords(tokenized_list):
    text = [word for word in tokenized_list if word not in STOPWORDS]
    return text

SMS['nosw_text'] = SMS['Text'].apply(lambda x: remove_stopwords(x))
SMS.head()


# **Visulazing the data**

# In[ ]:


# Barchart representing the number of spam & ham messages in the dataset
SMS.Class.value_counts().plot.bar(rot = 0)


# **Wordcloud to visualize spam words**

# In[ ]:


from wordcloud import WordCloud
wc = WordCloud(width=1500,height=1500,min_font_size=10,background_color='white')
spam_wc = wc.generate(SMS[SMS['Class'] == "spam"]['Text'].str.cat(sep=" "))


# In[ ]:


plt.figure(figsize=(15,6))
plt.imshow(spam_wc)


# **Wordcloud to visualize ham words**

# In[ ]:


ham_wc = wc.generate(SMS[SMS['Class'] == "ham"]['Text'].str.cat(sep=" "))
plt.figure(figsize=(15,6))
plt.imshow(ham_wc)


# **Bar chart to visualize top 30 ham and spam words**

# In[ ]:


# Top 30 Spam words
spam_corpus = []
for msg in SMS[SMS['Class'] == "spam"]['Text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)

len(spam_corpus)


# In[ ]:


from collections import Counter
sns.barplot(pd.DataFrame(Counter(spam_corpus).most_common(30))[0],pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# In[ ]:


# Top 30 ham words
ham_corpus = []
for msg in SMS[SMS['Class'] == "ham"]['Text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)

len(ham_corpus)        


# In[ ]:


from collections import Counter
sns.barplot(pd.DataFrame(Counter(ham_corpus).most_common(30))[0],pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# **Remove Punctuations**

# In[ ]:


import string
string.punctuation


# In[ ]:


# Funtion to remove punctuations
def remove_punct(text):
    text_nonpunct = "".join([char for char in text if char not in string.punctuation])
    return text_nonpunct

SMS['nopunc_text'] = SMS['Text'].apply(lambda x: remove_punct(x))
SMS.head()


# **Tokenization**

# In[ ]:


from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')


# In[ ]:


word_tokenize(SMS['Text'][0])


# In[ ]:


from nltk.tokenize import TreebankWordTokenizer
TreebankWordTokenizer().tokenize(SMS['Text'][0])


# In[ ]:


from nltk.tokenize import TweetTokenizer
tokens = TweetTokenizer().tokenize(SMS['Text'][0])
tokens


# In[ ]:


# Function to Tokenize words
import re
def tokenize(text):
    tokens = re.split('\W+', text)
    return tokens

SMS['token_text'] = SMS['nopunc_text'].apply(lambda x: tokenize(x.lower()))
SMS.head()


# **Remove Stopwords**

# In[ ]:


import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = stopwords.words('english')
print(STOPWORDS)


# In[ ]:


# Function to remove stop words
def remove_stopwords(tokenized_list):
    text = [word for word in tokenized_list if word not in STOPWORDS]
    return text

SMS['nosw_text'] = SMS['token_text'].apply(lambda x: remove_stopwords(x))
SMS.head()


# **Stemming**

# In[ ]:


#Porterstemmer
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[ ]:


# Function to perform stemming
def stemming(tokenized_text):
    text = [ps.stem(word) for word in tokenized_text]
    return text

SMS['stemmed_text'] = SMS['nosw_text'].apply(lambda x: stemming(x))
SMS.head()


# In[ ]:


#Snowball stemmer
from nltk.stem import SnowballStemmer
snowball = SnowballStemmer(language='english')


# In[ ]:


# Function to perform stemming by snowballstemmer
def sbstemming(tokenized_text):
    text = [snowball.stem(word) for word in tokenized_text]
    return text

SMS['sbstemmed_text'] = SMS['nosw_text'].apply(lambda x: sbstemming(x))
SMS.head()


# **Lemmatization**

# In[ ]:


import nltk
nltk.download('wordnet')
from nltk import WordNetLemmatizer
WN = nltk.WordNetLemmatizer()

# Function for performing lemmatization on the data
def lemmatizing(tokenized_text):
    text = [WN.lemmatize(word) for word in tokenized_text]
    return text

SMS['lemmatized_text'] = SMS['stemmed_text'].apply(lambda x: lemmatizing(x))
SMS.head()


# **Test Train Split**

# In[ ]:


X = SMS['Text']
y = SMS['Class']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[ ]:


print("X_train:\n{}".format(X_train[:10]))
print("=========================================")
print("X_test:\n{}".format(X_test[:10]))
print("=========================================")
print("y_train:\n{}".format(y_train[:10]))
print("=========================================")
print("y_test:\n{}".format(y_test[:10]))


# **Binarize the Class labels**

# In[ ]:


from sklearn.preprocessing import label_binarize

y_train = label_binarize(y_train, classes=['ham', 'spam']).ravel()
y_test = label_binarize(y_test, classes=['ham', 'spam']).ravel()


# In[ ]:


y_train[:20]


# In[ ]:


y_test[:20]


# In[ ]:


print("X_train:\n{}".format(X_train[:10]))
print("=========================================")
print("X_test:\n{}".format(X_test[:10]))
print("=========================================")
print("y_train:\n{}".format(y_train[:10]))
print("=========================================")
print("y_test:\n{}".format(y_test[:10]))


# **Data Vectorizing**

# **Applying TF-IDF transformation**

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer()


# In[ ]:


X_train_t = tfidf.fit_transform(X_train)
X_test_t = tfidf.transform(X_test)
print("X_train_t:\n{0}".format(X_train_t[:5]))
print("=========================================")
print("X_test_t:\n{0}".format(X_test_t[:5]))
print("=========================================")
print("y_train:\n{0}".format(y_train[:5]))
print("=========================================")
print("y_test:\n{0}".format(y_test[:5]))


# In[ ]:


X_train_t = tfidf.fit_transform(X_train).toarray()# fit and transform the train set
X_train_t.shape


# In[ ]:


X_test_t = tfidf.transform(X_test).toarray()# only transform the test set
X_test_t.shape


# **Building Model**

# In[ ]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()


# In[ ]:


# Gaussian NaiveBayes
gnb.fit(X_train_t,y_train)
y_pred1 = gnb.predict(X_test_t)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[ ]:


#Multinomial Naive Bayes
mnb.fit(X_train_t,y_train)
y_pred2 = mnb.predict(X_test_t)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


# In[ ]:


#Bernoyli NaiveBayes
bnb.fit(X_train_t,y_train)
y_pred3 = bnb.predict(X_test_t)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


# In[ ]:


lrc = LogisticRegression(solver='liblinear', penalty = 'l1')
svc = SVC(kernel='sigmoid', gamma=1.0)
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
knc = KNeighborsClassifier()
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)
xgb = XGBClassifier(n_estimators=50,random_state=2)


# In[ ]:


clfs = {
    'LR'  : lrc,
    'SVC' : svc,
    'NB'  : mnb,
    'DT'  : dtc,
    'KN'  : knc,
    'RF'  : rfc,
    'GBDT': gbdt,
    'xgb' : xgb
}


# In[ ]:


def train_classifier(clf,x_train,y_train,x_test,y_test):
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    
    return accuracy,precision

train_classifier(knc,X_train_t,y_train,X_test_t,y_test)    


# In[ ]:


accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    
    current_accuracy,current_precision = train_classifier(clf,X_train_t,y_train,X_test_t,y_test)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)


# In[ ]:


performance_df = pd.DataFrame({'Algorithm' : clfs.keys(),
              'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)


# In[ ]:


performance_df


# We got the highest accuracy of 98.3% for SVC algorithm followed by Randomforest with 98%.
# Considering Precision, the best model is BradientBoost Classifier with 96.6% accuracy and 100% precision followed by Naivebayes with 96.2% accuracy.

# In[ ]:


performance_df1 = pd.melt(performance_df, id_vars = "Algorithm")
performance_df1


# **Visualisation**

# In[ ]:


# Visualising
sns.catplot(x = 'Algorithm', y='value',hue = 'variable',data=performance_df1, kind='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation='vertical')
plt.show()


# **Conclusion:** For ham spam classification, the best model is chosen based on the precision value as this is the significant factor here than the accuracy. Hence the best model will be Gradient boost decision tree classifier with 96.6% but with 100% precision.
