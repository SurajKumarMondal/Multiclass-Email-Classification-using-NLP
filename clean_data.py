import pandas as pd
import pickle



with open('text_data_labels.pkl', 'rb') as f:
    text_file_data, text_file_labels = pickle.load(f)

data = pd.DataFrame(list(zip(text_file_data, text_file_labels)),
               columns =['text', 'labels'])



print(data['text'][0])


import re
import string
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

re_url = re.compile(r'(?:http|ftp|https)://(?:[\w_-]+(?:(?:\.[\w_-]+)+))(?:[\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?')
re_email = re.compile('(?:[a-z0-9!#$%&\'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&\'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])')



def clean_header(text):
    text = re.sub(r'(From:\s+[^\n]+\n)', '', text)
    text = re.sub(r'(Subject:[^\n]+\n)', '', text)
    text = re.sub(r'(([\sA-Za-z0-9\-]+)?[A|a]rchive-name:[^\n]+\n)', ' ', text)
    text = re.sub(r'(Last-modified:[^\n]+\n)', '', text)
    text = re.sub(r'(Version:[^\n]+\n)', '', text)

    return text


data['text_cleaned'] = data['text'].apply(clean_header)


def clean_text(text):        
    text = text.lower()
    text = text.strip()
    text = re.sub(re_url, '', text)
    text = re.sub(re_email, '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    text = re.sub(r'(\d+)', ' ', text)
    text = re.sub(r'(\s+)', ' ', text)
    
    return text

data['text_cleaned'] = data['text'].apply(clean_text)


stop_words = stopwords.words('english')
data['text_cleaned'] = data['text_cleaned'].str.split().apply(lambda x: ' '.join([word for word in x if word not in stop_words]))


import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

from nltk.stem import WordNetLemmatizer
  
lemmatizer = WordNetLemmatizer()


def apply_lemmatization(text):
    new_string = []
    s = ' '
    for i in text.split():
        #print(i)
        if(len(i) > 2):
            #print(i)
            new_string.append(  lemmatizer.lemmatize(i) )
    return s.join(new_string)


data['text_cleaned'] = data['text_cleaned'].apply(apply_lemmatization)


print(data['text_cleaned'][0])
data.to_csv('./cleaned_data.csv', index=False)
