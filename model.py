import pandas as pd
import pickle


data = pd.read_csv('cleaned_data.csv')
data = data.sample(frac = 1, random_state = 0)

print('class labels and counts')
print(data['labels'].value_counts())


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(data['text_cleaned'], data['labels'], test_size = 0.2, random_state = 0, stratify= data['labels'])


from sklearn.feature_extraction.text import TfidfVectorizer

'''
tfidf_train_val = TfidfVectorizer(sublinear_tf = True , min_df = 5, ngram_range = (1, 2), max_df = 0.2)
#hopefully keeping mindf will remove eroor in spellings
#maxdf will remove terms which occures in more than 50 percent documents
X_train_features = tfidf_train_val.fit_transform(X_train).toarray()
X_val_features = tfidf_train_val.transform(X_val)
'''
#As we are using k cross validation we will use only this tfidf vectorizer


#################### Use this when only for testing train contaings train aand validation data################################
tfidf_train_test = TfidfVectorizer(max_features=8000,sublinear_tf = True , min_df = 5, ngram_range = (1, 2), max_df = 0.4)




train_df_features = tfidf_train_test.fit_transform(X_train)
test_df_features = tfidf_train_test.transform(X_val)
'''
train_df['labels'] = y_train
test_df['labels'] = y_val
'''
print(train_df_features.shape)
print(test_df_features.shape)

from sklearn.feature_selection import chi2
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import f1_score

model = LogisticRegression(random_state=0)

model.fit(train_df_features, y_train)

print('Tried Simple Logistic Model FIrst')

y_train_pred = model.predict(train_df_features)
y_test_pred = model.predict(test_df_features)
print()
print(f"Train F1 Micro Average score {f1_score(y_train, y_train_pred, average='micro')}")



print(f"Test F1 Micro Average score {f1_score(y_val, y_test_pred, average='micro')}")


models = [
    RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0)
]

# 5 Cross-validation
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))

entries = []
for model in models:
  
  model_name = model.__class__.__name__
  print(model_name)
  accuracies = cross_val_score(model, train_df_features, y_train, scoring='accuracy', cv=CV)
  print(accuracies)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
    
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])



mean_accuracy = cv_df.groupby('model_name').accuracy.mean()
std_accuracy = cv_df.groupby('model_name').accuracy.std()

acc = pd.concat([mean_accuracy, std_accuracy], axis= 1, 
          ignore_index=True)
acc.columns = ['Mean Accuracy', 'Standard deviation']

print(acc)

print('linear svc performed better')



model = LinearSVC()

model.fit(train_df_features, y_train)

print('Linear SVC')

y_train_pred = model.predict(train_df_features)
y_test_pred = model.predict(test_df_features)


print(f"Train F1 Micro Average score {f1_score(y_train, y_train_pred, average='micro')}")



print(f"Test F1 Micro Average score {f1_score(y_val, y_test_pred, average='micro')}")


