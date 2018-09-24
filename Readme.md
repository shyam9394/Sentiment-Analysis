# Spam or Ham
## Introduction 
The dataset for this project was taken from Kaggle.com. This code helps in detecting whether a text is spam or ham. Several operations are performed on the dataset and a bag of words is created (Sparse Matrix). Random forest classifier is used here to classify if a text is spam or ham. With the help of confusion matrix, an accuracy of 97.85% is obtained.

## Implementation
### Importing libraries

`import pandas as pd`  
Used for importing dataset

`import re`   
Used for text cleaning

`import nltk`  
Library used for natural language processing

`from nltk.corpus import stopwords`  
Used for importing stop words

`from nltk.stem.porter import PorterStemmer`  
Used for stemming words
  
`from sklearn.feature_extraction.text import CountVectorizer`   
Creates bag of words 

`from sklearn.preprocessing import LabelEncoder`  
Encodes categorical data

`from sklearn.model_selection import train_test_split`  
Used for splitting dataset

`from sklearn.ensemble import RandomForestClassifier`  
Random forest classifier

`from sklearn.metrics import confusion_matrix`  
Used for creating confusion matrix

`from sklearn.model_selection import GridSearchCV`  
Used for performance tuning

### Importing dataset  
`dataset = pd.read_csv('spam.csv', sep=',', engine='python')`

![](https://github.com/shyam9394/Spam-or-Ham/blob/master/Images/1.jpg?raw=true)

The dataset contains 5 columns v1, v2, unnamed:2, unnamed:3, unnamed:4. Only the columns v1 and v2 are used, where v1 is the dependent variable and v2 is the independent variable.

### Text cleaning 
`corpus = []`

![](https://github.com/shyam9394/Spam-or-Ham/blob/master/Images/2.jpg?raw=true)

A corpus is created which is a list that is initially empty. After performing text cleaning, the corpus contains a huge list of text without anything unnecessary.

`for i in range(0,5572):`  

Several operations are performed on text dataset. A for loop is applied, hence these operations are applied on the entire list

![](https://github.com/shyam9394/Spam-or-Ham/blob/master/Images/3.jpg?raw=true)


The last element of dataset initially looks like above

`review = re.sub('[^a-zA-Z]', ' ', dataset['v2'][i])` 



Removing all the elements but the alphabets ( lower and upper case).


`review = review.lower()`  

![](https://github.com/shyam9394/Spam-or-Ham/blob/master/Images/5.jpg?raw=true)  
Changing text to lower case.

 
`review = review.split()`  

![](https://github.com/shyam9394/Spam-or-Ham/blob/master/Images/6.jpg?raw=true)

Splitting sentence into seperate words so, unnecessary words can be removed easily

`ps = PorterStemmer()`  
initializing an object ps using PorterStemmer class

`review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]`   

![](https://github.com/shyam9394/Spam-or-Ham/blob/master/Images/7.jpg?raw=true)

Removing all the words that are in stopwords list


`review = ' '.join(review)`   

![](https://github.com/shyam9394/Spam-or-Ham/blob/master/Images/8.jpg?raw=true)

Joining the words to form a string of cleaned words.

`corpus.append(review)`

![](https://github.com/shyam9394/Spam-or-Ham/blob/master/Images/9.jpg?raw=true)


Appending the object review to corpus list. 


### Creating a bag of words  
`cv=CountVectorizer(max_features = 2500)`  
`X=cv.fit_transform(corpus).toarray()`  
`y=dataset.iloc[:,0].values`

![](https://github.com/shyam9394/Spam-or-Ham/blob/master/Images/10.jpg?raw=true)

Creating a bag of words (Sparse matrix)

### Encoding Categorical data  
`labelencoder=LabelEncoder()`  

![](https://github.com/shyam9394/Spam-or-Ham/blob/master/Images/16.jpg?raw=true)

It is important to encode categorical data. There are two categories in dependent variable, Ham and Spam.

`y=labelencoder.fit_transform(y)`


![](https://github.com/shyam9394/Spam-or-Ham/blob/master/Images/11.jpg?raw=true)

The label encoder encodes ham and spam to 0 and 1

### Splitting dataset into training and test set
`X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.2, random_state =0)`

![](https://github.com/shyam9394/Spam-or-Ham/blob/master/Images/17.jpg?raw=true)

Splitting X and y into X_train, X_test, y_train, y_test with test size = 20% 

### Applying Random forest classifier  

`classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0, n_jobs = -1)` 
`classifier.fit(X_train, y_train)`

![](https://github.com/shyam9394/Spam-or-Ham/blob/master/Images/12.jpg?raw=true)

n_estimators = 100 is similar to having 100 individual decision trees, entropy criterion is used and all the processor cores are used (n_jobs = -1)

### Predicting the test set results  

`y_pred = classifier.predict(X_test)`

![](https://github.com/shyam9394/Spam-or-Ham/blob/master/Images/13.jpg?raw=true)

Predictions of test set



### confusion matrix  
`cm = confusion_matrix(y_pred,y_test)`

![](https://github.com/shyam9394/Spam-or-Ham/blob/master/Images/14.jpg?raw=true)

The confusion matrix is used to calculate the accuracy of the predicted categories. It contains 948 correct and 1 incorrect Ham predictions, 142 correct and 24 incorrect spam predictions. An accuracy of 97.85% is obtained.

![](https://github.com/shyam9394/Spam-or-Ham/blob/master/Images/15.jpg?raw=true)

The accuracy is calculated by (sum of correct predictions)/ total no.of predictions.

### Parameter tuning   
`parameters = [{'n_estimators': [1, 10, 100, 1000], 'criterion': ['entropy']}] `   
Using different parameter values of random forest classifier

`grid_search = GridSearchCV(estimator = classifier,`  
                           `param_grid = parameters,`  
                           `scoring = 'accuracy',`  
                           `cv = 10,`  
                           `n_jobs = -1)`  

Using accuracy as the primary factor for evaluation.

`grid_search = grid_search.fit(X_train, y_train)`  

Fitting grid search object to X_train and y_train

`best_accuracy = grid_search.best_score_`  
`best_parameters = grid_search.best_params_`

![](https://github.com/shyam9394/Spam-or-Ham/blob/master/Images/18.jpg?raw=true)

best_accuracy gives the best accuracy with the given parameters, best parameters give the best values for different parameters.


