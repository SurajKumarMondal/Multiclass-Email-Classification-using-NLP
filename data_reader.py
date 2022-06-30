import pickle
import matplotlib.pyplot as plt
import pandas as pd
import os
  
# Folder Path
path = "./enron_with_categories"
  
listOfFiles = list()
for (dirpath, dirnames, filenames) in os.walk(path):
    listOfFiles += [os.path.join(dirpath, file) for file in filenames]
#print(len(listOfFiles))



text_file_names = list(filter(lambda x: x[-4:] == '.txt', listOfFiles))
#print(txt_files)
#print(len(text_file_names))


#print(text_file_names[0].split('/'))
#exit()
#we can get labels from here

#Now get text data from files in txt files:

def read_text_file(file_path):
    with open(file_path, 'r') as f:
        text = f.read()
    return text
text_file_path = []
text_file_data = []
text_file_labels = []

for file_name in text_file_names:
    text = read_text_file(file_name)
    if(int(file_name.split('/')[2]) != 7 and int(file_name.split('/')[2]) != 8):
        text_file_path.append(file_name)
        text_file_data.append(text)
        text_file_labels.append( int(file_name.split('/')[2]) )


print(len(text_file_data))
print(text_file_path[0], text_file_data[0])
print( text_file_labels[0])


with open('text_data_labels.pkl', 'wb') as f:
    pickle.dump((text_file_data, text_file_labels), f)





