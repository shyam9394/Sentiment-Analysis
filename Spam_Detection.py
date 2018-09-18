
import pandas as pd

dataset = pd.read_csv('spam.csv', sep=',', engine='python')


# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,5572):
        review = re.sub('[^a-zA-Z]', ' ', dataset['v2'][i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)


        
#Creating a bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features = 2500)
X=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,0].values

#Encoding Categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
y=labelencoder.fit_transform(y)

#Splitting dataset into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.2, random_state =0)


#Random forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0, n_jobs = -1)
classifier.fit(X_train, y_train)

#Predicting the test set results
y_pred = classifier.predict(X_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred,y_test)

    

from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators': [1, 10, 100, 1000], 'criterion': ['entropy']}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_




