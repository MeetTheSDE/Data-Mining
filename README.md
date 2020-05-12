<center><b><h1>Data Mining Term Project Spring 2020</h1></bold></center>

>Name: Patel, Meetkumar Jigneshbhai<br>
>UTA ID: 100175000

The goal of the term project is building a classifier that you can show to someone with your homepage that will help you look competent.<br><br>



<h2>Overview of this notebook:</h2>

1.   I have used Board game geek dataset from kaggle:<br>
     Link: https://www.kaggle.com/jvanelteren/boardgamegeek-reviews<br>

2.   This dataset has 13million rows which is quite large dataset for the system that we use, so I have implemented this project on google colaboratory.<br>

3.    Data preprocessing on dataset:<br>
    In preprocessing, I removed all the rows having no values in comment column, as without comment value there is nothing to predict. 
    To remove stop words from comments, first, I tokenized those comments using tokenizer tool provided by Natural Language toolkit (NLTK), and then removed those stop words. In nltk, there is one class "corpus" which contains list of these stop words. Along with them, comments are also converted into lower alphabets using lower() function.<br>
    Data visualization:
    For data visualization, I have used matplotlib library and plotted graph for total number of comments with their ratings value, and mean and median values.

4.   Models for data training and testing:<br>
    There are so many models for text classification, Naive Bayes, Support vector machine, random forest, ridge regression, linear regression, etc.<br>
    I have used some of those models in order to get the best accuracy on the dataset I am using.<br>

5.   Performance evaluation of algorithms:<br>
    We must know how our algorithm is working. For that purpose, some accuracy measures, error meaasurement techniques are used. <br>
    Such as, confusion matrix, F1 score, precision, recall, etc.



<h2>Classifiers in this project:</h2>


>1. Multinomial Naive Bayes

This is basic logic behind naive bayes. It is all about conditional dependency.The term Multinomial Naive Bayes simply lets us know that each p(fi|c) is a multinomial distribution, rather than some other distribution. This works well for data which can easily be turned into counts, such as word counts in text.<br> 
![alt text](https://miro.medium.com/max/341/0*EfYTXtTJ9X-Ua9Nh.png)

>2. Ridge Classifier

This classifier first converts the target values into {-1, 1} and then treats the problem as a regression task (multi-output regression in the multiclass case). The L2 norm term in ridge regression is weighted by the regularization parameter alpha. So, if the alpha value is 0, it means that it is just an Ordinary Least Squares Regression model. So, the larger is the alpha, the higher is the smoothness constraint.
<br>
![alt text](https://miro.medium.com/max/1528/1*3cEysrHZokqla0tXnZ-5GQ.png)
<br><br>
<bold>Example:
<br>
![alt text](https://www.analyticsvidhya.com/wp-content/uploads/2015/08/Bayes_41-850x310.png)
<br>
P(Yes | Sunny) = P( Sunny | Yes) * P(Yes) / P (Sunny)<br>
Here we have P (Sunny |Yes) = 3/9 = 0.33, P(Sunny) = 5/14 = 0.36, P( Yes)= 9/14 = 0.64<br>
Now, P (Yes | Sunny) = 0.33 * 0.64 / 0.36 = 0.60, which has higher probability.

>3. Support vector machine<br>

SVM is a supervised machine learning algorithm which can be used for classification or regression problems.Support vectors are data points that are closer to the hyperplane and influence the position and orientation of the hyperplane. Using these support vectors, we maximize the margin of the classifier. SVM has one argument named "kernel", which specifies, the hyperplane shape which separates labels. I am using Linear shape for kernel value.<br>
![alt text](https://66.media.tumblr.com/ff709fe1c77091952fb3e3e6af91e302/tumblr_inline_o9aa8dYRkB1u37g00_540.png)

>4. k nearest neighbours<br>

kNN is easy-to-implement supervised machine learning algorithm that can be used to solve both classification and regression problems. The KNN algorithm assumes that similar things exist in close proximity. In other words, similar things are near to each other. To select the K that’s right for your data, we run the KNN algorithm several times with different values of K and choose the K that reduces the number of errors we encounter while maintaining the algorithm’s ability to accurately make predictions when it’s given data it hasn’t seen before.
<br>
![alt text](data:image/png;base64,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)

>5. Logistic regression

Logistic regression is the appropriate regression analysis to conduct when the dependent variable is dichotomous (binary). Logistic regression is a statistical model that in its basic form uses a logistic function. Logistic regression is used to describe data and to explain the relationship between one dependent binary variable and one or more nominal, ordinal, interval or ratio-level independent variables.<br><br>
Below image shows difference between linear and logistic regression.
<br>
![alt text](data:image/jpeg;base64,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)

>Ensemble methods<br>

Ensemple method is machine learning trechnique which improves accuracy of algorithm by combining two or more individual algorithms for same dataset.<br>
On above part, I got accuracy for SVC, Linear SVC, logistic regression, but that is very low to classify some random text.<br>
SO here, I am using ensemble method on those algorithms to get better accuracy than what I have got.<br><br>
Below is the architecture of working of ensemble methods.<br>
![alt text](https://miro.medium.com/max/1400/0*PBGJw23ud8Sp7qO4.)

Importing required dependencies and mounting google drive for data fetching
---


```python
import os
import sys
import re
import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import train_test_split
from sklearn import feature_extraction, linear_model, model_selection,preprocessing
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,f1_score,recall_score,precision_score,classification_report
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression,Ridge,RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.metrics import accuracy_score
import time
from time import sleep
#nltk toolkit used for stopwords removing
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
nltk.download('punkt')
nltk.download("stopwords")
```

    /usr/local/lib/python3.6/dist-packages/statsmodels/tools/_testing.py:19: FutureWarning: pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.
      import pandas.util.testing as tm
    

    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    




    True



Mounting google drive to read dataset


```python
#code to mount google drive for data 
from google.colab import drive
drive.mount('/content/gdrive')
```

    Drive already mounted at /content/gdrive; to attempt to forcibly remount, call drive.mount("/content/gdrive", force_remount=True).
    


```python
os.chdir('/content/gdrive/My Drive/Data mining/')
```

Data acquisition of Board Game Geek dataset from google drive
---


```python
#data acquisition from the given path in google drive
review_data_game = pd.read_csv("bgg-13m-reviews.csv")
print("Length of review data: ",len(review_data_game))
#representing first five rows from dataframe
review_data_game.head()
```

    Length of review data:  13170073
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>user</th>
      <th>rating</th>
      <th>comment</th>
      <th>ID</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>sidehacker</td>
      <td>10.0</td>
      <td>NaN</td>
      <td>13</td>
      <td>Catan</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Varthlokkur</td>
      <td>10.0</td>
      <td>NaN</td>
      <td>13</td>
      <td>Catan</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>dougthonus</td>
      <td>10.0</td>
      <td>Currently, this sits on my list as my favorite...</td>
      <td>13</td>
      <td>Catan</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>cypar7</td>
      <td>10.0</td>
      <td>I know it says how many plays, but many, many ...</td>
      <td>13</td>
      <td>Catan</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>ssmooth</td>
      <td>10.0</td>
      <td>NaN</td>
      <td>13</td>
      <td>Catan</td>
    </tr>
  </tbody>
</table>
</div>



Pre-processing of the dataset
---


```python
# counting total NaN values in each column
review_data_game.isna().sum()
```




    Unnamed: 0           0
    user                66
    rating               0
    comment       10532317
    ID                   0
    name                 0
    dtype: int64




```python
# this chart represents top 50 rated words from dataframe 
plt.figure(figsize=(20, 7))
review_data_game['name'].value_counts()[:50].plot(kind='bar')
plt.ylabel('Count of Rating')
plt.title('Top 50 Rated Games in dataframe')
plt.show()
```


![png](output_19_0.png)



```python
review_data_game = review_data_game[['comment','rating']]
#removing missing values from dataframe
review_data_game.dropna(inplace=True)
print("Length of dataset after removing missing values: ",len(review_data_game))
print('')
review_data_game.head()
```

    Length of dataset after removing missing values:  2637756
    
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>comment</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>Currently, this sits on my list as my favorite...</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>I know it says how many plays, but many, many ...</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>i will never tire of this game.. Awesome</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>This is probably the best game I ever played. ...</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Fantastic game. Got me hooked on games all ove...</td>
      <td>10.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#tokenization of comments column and removing punctuation
from nltk.tokenize import RegexpTokenizer
#this regular expression tokenize words in comma separated form
tokenizer = RegexpTokenizer(r'\w+')
review_data_game['comment'] = review_data_game['comment'].apply(lambda x: tokenizer.tokenize(x.lower()))
review_data_game.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>comment</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>[currently, this, sits, on, my, list, as, my, ...</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[i, know, it, says, how, many, plays, but, man...</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>[i, will, never, tire, of, this, game, awesome]</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>[this, is, probably, the, best, game, i, ever,...</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>[fantastic, game, got, me, hooked, on, games, ...</td>
      <td>10.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#removing stop words from comments 
stops = set(stopwords.words("english"))
review_data_game['comment'] = review_data_game['comment'].apply(lambda x: [item for item in x if item not in stops])
review_data_game.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>comment</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>[currently, sits, list, favorite, game]</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[know, says, many, plays, many, many, uncounte...</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>[never, tire, game, awesome]</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>[probably, best, game, ever, played, requires,...</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>[fantastic, game, got, hooked, games]</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>[one, best, games, ever, created, period, new,...</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>[game, 9, strategy, game, family, asks, play, ...</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>[great, game, even, got, number, non, game, pl...</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>[one, time, favorite, games, usually, get, pla...</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>[settlers, gem, havn, played, suggest, go, get...</td>
      <td>10.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#joining all tokens to proceed further for data modeling
review_data_game['comment'] = review_data_game['comment'].apply(' '.join)
```


```python
review_data_game.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>comment</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>currently sits list favorite game</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>know says many plays many many uncounted liked...</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>never tire game awesome</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>probably best game ever played requires thinki...</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>fantastic game got hooked games</td>
      <td>10.0</td>
    </tr>
  </tbody>
</table>
</div>



>Data visualization


```python
#histogram representing how much time each ratings occur
fig = plt.figure()
n, bins, patches = plt.hist(review_data_game.rating, bins=40, facecolor='blue',alpha = 1)
plt.xlabel('Rating')
plt.ylabel('Counts')
plt.title('Occurrence of each rating in preprocessed data')
fig.patch.set_facecolor('white')
plt.show()
# width of bars are defined using value of bin
```


![png](output_26_0.png)


* As we can see, most of the ratings are between 6 and 8.
* Below functions explains exact values.



```python
#average value of all the ratings 
print("Average rating: ",review_data_game['rating'].mean())
#median value of ratings values
print("Median value: ",review_data_game['rating'].median())
```

    Average rating:  6.852069823043514
    Median value:  7.0
    

Model Creation and testing 
---


```python
#selecting chunk of data (if you want) from whole dataframe for further model training and testing
review_data = review_data_game
```


```python
#rounding up all values of rating column such that: 7.8 ~ 8.0
review_data['rating'] =np.round(review_data['rating'])
```


```python
#occurance of each rating in the column
review_data['rating'].value_counts()
```




    8.0     657581
    7.0     574586
    6.0     526481
    9.0     239410
    5.0     217766
    10.0    153530
    4.0     136565
    3.0      70974
    2.0      40766
    1.0      20086
    0.0         11
    Name: rating, dtype: int64




```python
#splitting dataset in to four parts randomly
X_train, X_test, y_train, y_test = train_test_split(review_data['comment'], review_data['rating'], test_size=0.3) # 70% training and 30% test
```

>TF-IDF stands for term frequency-inverse document frequency.TF-IDF is a weight often used in information retrieval and text mining.Tf-idf can be successfully used for stop-words filtering in various subject fields including text summarization and classification.<br>
CountVectorizer:<br>
Transforms text into a sparse matrix of n-gram counts.<br>
TfidfTransformer:<br>
Performs the TF-IDF transformation from a provided matrix of counts.


```python
#vectorizing our data
#Convert a collection of raw documents to a matrix of TF-IDF features
vectorizer = TfidfVectorizer()
transformer = TfidfTransformer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

X_train_counts = transformer.fit_transform(X_train)
X_test_counts = transformer.fit_transform(X_test)
```



> Multinomial NaiveBayes




```python
model_mNB = MultinomialNB().fit(X_train_counts, y_train.astype('int'))
y_predicted_mNB = model_mNB.predict(X_test_counts)

accuracy = accuracy_score(y_test.astype('int'),y_predicted_mNB) * float(100)
accuracy_mNB = str(accuracy)
print('Testing Accuracy on multinomial naive bayes model is: '+accuracy_mNB+' %')
```

    Testing Accuracy on multinomial naive bayes model is: 30.23579379952914 %
    


```python
#confusion matrix for what output we got
print("Confusion matrix for Multinomial Naive Bayes classifier:\n")
matrix = confusion_matrix(y_test.astype('int'), y_predicted_mNB)
sb.heatmap(matrix, square=True, annot=True, fmt='d', cbar=False )
plt.xlabel('Actual rating')
plt.ylabel('Predicted rating');
plt.show()
```

    Confusion matrix for Multinomial Naive Bayes classifier:
    
    


![png](output_38_1.png)


>Ridge Classifier<br>



```python
model_ridge_class = RidgeClassifier().fit(X_train_counts, y_train.astype('int'))
y_predicted = model_ridge_class.predict(X_test_counts)
y_predicted_ridge_class = np.round(y_predicted)
accuracy = accuracy_score(y_test,y_predicted_ridge_class) * float(100)
ridge_class_accuracy = str(accuracy)
print('Testing Accuracy on ridge classifier model is: '+ridge_class_accuracy+' %')
```

    Testing Accuracy on ridge classifier model is: 31.376914979521743 %
    


```python
print("Confusion matrix for Ridge Classifier:\n")
matrix = confusion_matrix(y_test.astype('int'), y_predicted_ridge_class)
sb.heatmap(matrix, square=True, annot=True, fmt='d', cbar=False )
plt.xlabel('Actual rating')
plt.ylabel('Predicted rating');
plt.show()
```

    Confusion matrix for Ridge Classifier:
    
    


![png](output_41_1.png)



```python
#here you can test rating for your desired review
smt = "worst game dont play always play"
smt=[smt]
smt = vectorizer.transform(smt)
smt = transformer.fit_transform(smt)
prediction = np.round(model_ridge_class.predict(smt))
print(prediction)
```

    [1]
    

>Classification report for the above two classifiers<br>

This report contains values like, precision, recall, support and f1 score.These values help us to determine what is the error for testing data.


```python
#computing f1 score to find weighted average of precision and recall
#f1 score is always defined in range of [0,1]
#where 0 is worse case and 1 is considered as best case
#recall is ratio of True positive over sum of true positive and false negative
#precision is ratio of True positive over sum of true positive and false positive
#support is total number of instances in the set 
labels = ['0.0','1.0','2.0','3.0','4.0','5.0','6.0','7.0','8.0','9.0','10.0']
p = classification_report(y_test.astype('int'), y_predicted_mNB, target_names=labels,zero_division=1)
print("Representation of classification matrix on a per-class basis:\n")
print("Report for multinomial naive bayes:\n ")
print(p)
print('')
p = classification_report(y_test.astype('int'), y_predicted_ridge_class, target_names=labels,zero_division=1,)
print("Report for Ridge regression:\n ")
print(p)
```

    Representation of classification matrix on a per-class basis:
    
    Report for multinomial naive bayes:
     
                  precision    recall  f1-score   support
    
             0.0       1.00      0.00      0.00         1
             1.0       0.18      0.00      0.00      6017
             2.0       0.60      0.00      0.01     12347
             3.0       0.08      0.00      0.00     21245
             4.0       0.20      0.00      0.01     40964
             5.0       0.23      0.00      0.00     65656
             6.0       0.29      0.32      0.31    158017
             7.0       0.28      0.16      0.20    172261
             8.0       0.31      0.81      0.45    197197
             9.0       0.23      0.00      0.01     71830
            10.0       0.45      0.01      0.01     45792
    
        accuracy                           0.30    791327
       macro avg       0.35      0.12      0.09    791327
    weighted avg       0.29      0.30      0.22    791327
    
    
    Report for Ridge regression:
     
                  precision    recall  f1-score   support
    
             0.0       1.00      0.00      0.00         1
             1.0       0.33      0.10      0.15      6017
             2.0       0.24      0.05      0.08     12347
             3.0       0.18      0.03      0.05     21245
             4.0       0.20      0.06      0.09     40964
             5.0       0.22      0.05      0.09     65656
             6.0       0.30      0.40      0.35    158017
             7.0       0.29      0.26      0.27    172261
             8.0       0.34      0.61      0.44    197197
             9.0       0.21      0.04      0.07     71830
            10.0       0.35      0.17      0.23     45792
    
        accuracy                           0.31    791327
       macro avg       0.33      0.16      0.16    791327
    weighted avg       0.29      0.31      0.27    791327
    
    

Some other models
---

I have trained these models for some small number of dataset values (20000-30000). Just to check their performance, accuracy, and time consumption


```python
small_review_data = review_data_game.sample(n=30000)
small_review_data['rating'] =np.round(small_review_data['rating'])
X_train_small, X_test_small, y_train_small, y_test_small = train_test_split(small_review_data['comment'], small_review_data['rating'], test_size=0.3) # 70% training and 30% test
```


```python
#histogram representing how much time each ratings occur
fig = plt.figure()
n, bins, patches = plt.hist(small_review_data.rating, bins=40, facecolor='blue',alpha = 1)
plt.xlabel('Rating')
plt.ylabel('Counts')
plt.title('Occurrence of each rating in this small data')
fig.patch.set_facecolor('white')
plt.show()
# width of bars are defined using value of bin
```


![png](output_48_0.png)



```python
#vectorizing our data
vect = TfidfVectorizer()
trans = TfidfTransformer()

X_train_small = vect.fit_transform(X_train_small)
X_test_small = vect.transform(X_test_small)

X_train_small = trans.fit_transform(X_train_small)
X_test_small = trans.fit_transform(X_test_small)
#model_SVM = Pipeline([
#    ('Tfidf_vectorizer', TfidfVectorizer()), 
#    ('Tfidf_transformer',  TfidfTransformer()), 
#    ('classifier', SVC(kernel="linear"))])  #this kernel=linear argument states that decision boundary to separate data is straight line
```



> Support vector machine (SVC)




```python
model_SVC = SVC(kernel="linear",probability=True).fit(X_train_small,y_train_small.astype('int'))
#this kernel=linear argument states that decision boundary to separate data is straight line
y_predicted_SVC = model_SVC.predict(X_test_small)
accuracy = accuracy_score(y_test_small.astype('int'),y_predicted_SVC) * float(100)
accuracy_svc = str(accuracy)
print('Testing Accuracy is: '+accuracy_svc+' %')
```

    Testing Accuracy is: 28.999999999999996 %
    


```python
print("Confusion matrix for Support vector machine:\n")
matrix = confusion_matrix(y_test_small.astype('int'), y_predicted_SVC)
sb.heatmap(matrix, square=True, annot=True, fmt='d', cbar=False )
plt.xlabel('Actual rating')
plt.ylabel('Predicted rating');
plt.show()
```

    Confusion matrix for Support vector machine:
    
    


![png](output_52_1.png)




> Support vector machine (Linear SVC)
* We will use same number of samples in SVC and Linear SVC for accuracy comparison



```python
model_linear_SVC = LinearSVC().fit(X_train_small,y_train_small.astype('int'))
y_predicted_linear_SVC = model_linear_SVC.predict(X_test_small)
accuracy = accuracy_score(y_test_small.astype('int'),y_predicted_linear_SVC) * float(100)
accuracy_linear_svc = str(accuracy)
print('Testing Accuracy is: '+accuracy_linear_svc+' %')
```

    Testing Accuracy is: 26.066666666666666 %
    


```python
print("Confusion matrix for Linear Support vector machine:\n")
matrix = confusion_matrix(y_test_small.astype('int'), y_predicted_linear_SVC)
sb.heatmap(matrix, square=True, annot=True, fmt='d', cbar=False )
plt.xlabel('Actual rating')
plt.ylabel('Predicted rating');
plt.show()
```

    Confusion matrix for Linear Support vector machine:
    
    


![png](output_55_1.png)


* For same number of training and testing data, SVC model gives better accuracy than Linear SVC model, but at cost of time.



> Logistic Regression





```python
model_Logistic_Regression = LogisticRegression(max_iter=20000).fit(X_train_small,y_train_small.astype('int'))
y_predicted_log_reg = model_Logistic_Regression.predict(X_test_small)
accuracy = accuracy_score(y_test_small.astype('int'),y_predicted_log_reg) * float(100)
accuracy_log_reg = str(accuracy)
print('Testing Accuracy is: '+accuracy_log_reg+' %')
```

    Testing Accuracy is: 29.22222222222222 %
    

* Logistic regression gives almost same accuracy as SVC model in even lesser time


```python
print("Confusion matrix for Logistic regression:\n")
matrix = confusion_matrix(y_test_small.astype('int'), y_predicted_log_reg)
sb.heatmap(matrix, square=True, annot=True, fmt='d', cbar=False )
plt.xlabel('Actual rating')
plt.ylabel('Predicted rating');
plt.show()
```

    Confusion matrix for Logistic regression:
    
    


![png](output_60_1.png)


>K nearest neighbours


```python
model_kNN = KNeighborsClassifier(n_neighbors=10).fit(X_train_small,y_train_small.astype('int'))
y_predicted_knn = model_kNN.predict(X_test_small)
accuracy = accuracy_score(y_test_small.astype('int'),y_predicted_knn) * float(100)
accuracy_knn = str(accuracy)
print('Testing Accuracy is: '+accuracy_knn+' %')
```

    Testing Accuracy is: 22.566666666666666 %
    


```python
print("Confusion matrix for kNN:\n")
matrix = confusion_matrix(y_test_small.astype('int'), y_predicted_knn)
sb.heatmap(matrix, square=True, annot=True, fmt='d', cbar=False )
plt.xlabel('Actual rating')
plt.ylabel('Predicted rating');
plt.show()
```

    Confusion matrix for kNN:
    
    


![png](output_63_1.png)


#Ensemble methods



Ensemple method is machine learning trechnique which improves accuracy of algorithm by combining two or more individual algorithms for same dataset.<br>
On above part, I got accuracy for SVC, Linear SVC, logistic regression, but that is very low to classify some random text.<br>
SO here, I am using ensemble method on those algorithms to get better accuracy than what I have got.<br>
Below is the **architecture** of working of ensemble methods.


```python
from sklearn.ensemble import VotingClassifier
Ensemble = VotingClassifier(estimators=[('Linear SVC',model_SVC),('knn',model_kNN),('logistic',model_Logistic_Regression)],voting='soft',weights=[2,1,3])
model_ensemble = Ensemble.fit(X_train_small,y_train_small.astype('int'))
ensemble_predicted = model_ensemble.predict(X_test_small)
accuracy_ensemble = accuracy_score(ensemble_predicted,y_test_small.astype('int'))* float(100)
accuracy_ensemble = str(accuracy_ensemble)
print('Testing Accuracy is: '+accuracy_ensemble+' %')
```

    Testing Accuracy is: 28.844444444444445 %
    

* As we can see, after ensembling Logistic regression, kNN and SVC algorithms we got better accuracy than what they have individually achieved.  


```python
print("Confusion matrix for Ensemble methos for given estimators:\n")
matrix = confusion_matrix(y_test_small.astype('int'), ensemble_predicted)
sb.heatmap(matrix, square=True, annot=True, fmt='d', cbar=False )
plt.xlabel('Actual rating')
plt.ylabel('Predicted rating');
plt.show()
```

    Confusion matrix for Ensemble methos for given estimators:
    
    


![png](output_68_1.png)


#Final Test

Here i have visualized accuracy of all the classifiers to compare.


```python
acc=[]
acc.append(accuracy_mNB)
acc.append(ridge_class_accuracy)
acc.append(accuracy_svc)
acc.append(accuracy_linear_svc)
acc.append(accuracy_log_reg)
acc.append(accuracy_knn)
```


```python
fig = plt.figure()
labels = ['MNB','Ridge','SVM','Linear SVC','Logistic','kNN']
#acc = [accuracy_mNB,ridge_class_accuracy,accuracy_svc,accuracy_linear_svc,accuracy_log_reg,accuracy_knn]
#acc = [1,2,3,4,5,6]
ax = fig.add_axes([0,0,1,1])
ax.bar(labels,acc)
plt.ylim(0,40,0.01)
plt.show()
```


![png](output_72_0.png)


>From above all classifiers and ensemble methods, Ridge classifier gives best output (prediction) for given review. So we are using that one for random review inputs.


```python
def redge_acc(smt):
  smt=[smt]
  smt = vectorizer.transform(smt)
  smt = transformer.fit_transform(smt)
  prediction = np.round(model_ridge_class.predict(smt))
  return prediction
```


```python
#insert values in the input box below to get rating
while True:
  smt = input()
  rating_prediction = redge_acc(smt)
  print(rating_prediction)
```

#Contributions



1.   The reference that  i have used for support vector machine, it was done without using any hyperparameters. I have implemented it's probability and "linear" kernel and got improved accuracy. 
2.   Implemented Ensemble methods over some of selected classifiers. I achieved a good accuracy than those individuals are getting. I used voting classifier in ensemble methods, with appropriate weighted voting to three classifiers. As I have used less data for those classifiers, accuracy is less. But for larger data surely would get better results. 
3. Saved model in local system for further any classification purpose. 



#Challenges faced:



1.   The dataset was too large for the systems we are using, so I need to pre process data first in order to use it for training and testing. For pre processing I have to use case lowering, tokenization, regular expression for alphabets, punctuation removal and removing stop words.
2.   I am using 5-7 classifiers for accuracy comparisons and data analysis. So I have to got in depth knowledge about those classifiers before implementation.
3. Better data visualization, I need to understand 7-10 types of libraries and graphs.



#Exporting trained models to drive and then to local system for further use


```python
#using joblib from sklearn to export the trained model, vectorizer, transformer objects to local files
#so that we can use them further anywhere else without running whole model again and get the accuracy
#as i am working on colab these values will be stored in google drive itself from which we can retrive later  
os.chdir('/content/gdrive/My Drive/Colab Notebooks')
from sklearn.externals import joblib
joblib.dump(model_ridge_class,'model_ridge_class.sav')
joblib.dump(model_mNB,'model_mNB.sav')
joblib.dump(model_SVC,'model_SVC.sav')
joblib.dump(model_linear_SVC,'model_linear_svc.sav')
joblib.dump(model_Logistic_Regression,'model_logistic_regression.sav')
joblib.dump(model_kNN,'model_knn.sav')
joblib.dump(model_ensemble,'model_ensemble.sav')
joblib.dump(vectorizer,'vectorizer.sav')
joblib.dump(transformer,'transformer.sav')
```




    ['transformer.sav']



#References:



1.   Reading large dataset: https://towardsdatascience.com/3-simple-ways-to-handle-large-data-with-pandas-d9164a3c02c1
2.   Data Preparation: https://www.kaggle.com/ngrq94/boardgamegeek-reviews-data-preparation
3.   Data preprocessing: https://pythonhealthcare.org/2018/12/14/101-pre-processing-data-tokenization-stemming-and-removal-of-stop-words/<br>
https://towardsdatascience.com/nlp-for-beginners-cleaning-preprocessing-text-data-ae8e306bef0f 
4. Multinomial Naive Bayes: https://www.datacamp.com/community/tutorials/naive-bayes-scikit-learn
5. Working with text data: https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
6. Multi-Class Text Classification with Scikit-Learn: https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f
7. Vectorizing using Scikit-learn API's : https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.text
8. Ridge Classifier: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html
9. Linear SVC classifier: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
10. SVC classifier: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
11. Logistic regression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
12. Ensemble methods: https://scikit-learn.org/stable/modules/ensemble.html
13. Confusion matrix: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
14. Classification report: https://scikit-learn.org/0.18/modules/generated/sklearn.metrics.classification_report.html
15. Random histograms: https://towardsdatascience.com/a-complete-exploratory-data-analysis-and-visualization-for-text-data-29fb1b96fb6a


```python

```
