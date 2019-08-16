import pandas as pd

# load the data
df = pd.read_csv('spam.csv', encoding='ISO-8859-1')

# check the data
df.head()

# check the shape of the data
df.shape

# check the info of the data
df.info()

# drop the other last 3 columns, because they have a lot of missing value
df = df[['v1','v2']]
df.head()

# using TF-IDF Vectorizer to process the text
from sklearn.feature_extraction.text import TfidfVectorizer
obj = TfidfVectorizer()

# define X and y
X = obj.fit_transform(df['v2'])
y = df['v1']

# make sure the X and y have the same rows
print(X.shape)
print(y.shape)

# split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# make sure the total for train and test data is the same
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# create the model
model_naive_bayes = MultinomialNB()
model_adaboost = AdaBoostClassifier()
model_random_forest = RandomForestClassifier()
model_svc = SVC()
model_knn = KNeighborsClassifier()

# fit the model into train data
model_naive_bayes.fit(X_train, y_train)
model_adaboost.fit(X_train, y_train)
model_random_forest.fit(X_train, y_train)
model_svc.fit(X_train, y_train)
model_knn.fit(X_train, y_train)

# print the classification rate
print('Classification rate for Naive Bayes: ', model_naive_bayes.score(X_test, y_test))
print('Classification rate for Adaboost: ', model_adaboost.score(X_test, y_test))
print('Classification rate for Random Forest: ', model_random_forest.score(X_test, y_test))
print('Classification rate for SVC: ', model_svc.score(X_test, y_test))
print('Classification rate for KNN: ', model_knn.score(X_test, y_test))

# cross validation 
from sklearn.model_selection import cross_val_score
score_nb = cross_val_score(model_naive_bayes, X_train, y_train, cv=10)
score_adaboost = cross_val_score(model_adaboost, X_train, y_train, cv=10)
score_rf = cross_val_score(model_random_forest, X_train, y_train, cv=10)
score_svc = cross_val_score(model_svc, X_train, y_train, cv=10)
score_knn = cross_val_score(model_knn, X_train, y_train, cv=10)

# print the average score
print('Average score of Naive Bayes:',  score_nb.mean())
print('Average score of Adaboost:',  score_adaboost.mean())
print('Average score of Random Forest:',  score_rf.mean())
print('Average score of SVC:',  score_svc.mean())
print('Average score of KNN:',  score_knn.mean())

# the best score is Adaboost

# see what is the word

from wordcloud import WordCloud
import matplotlib.pyplot as plt
def visualize(label):
    words = ''
    for msg in df[df['v1'] == label]['v2']:
        msg = msg.lower()
        words += msg + ' '
    wordcloud = WordCloud(width=600, height=400).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

visualize('spam')
visualize('ham')
