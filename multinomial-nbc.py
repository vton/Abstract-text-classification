import sklearn
import pandas as pd
import math
import numpy as np
import string
from nltk.corpus import stopwords


data = pd.read_csv('trg.csv', header = 0)
df = pd.DataFrame(data)
        
# Removing special characters from the data
# Alphanumeric characters will enhance the prediction accuracy
def removePunctuation(value):
    result = ''
    for a in value:
        if a not in string.punctuation:
            result += a
    return result

# Creating a list of common words that will not be used to predict the domain
stopwords_list = []
for stopword in stopwords.words('english'):
    stopword = removePunctuation(stopword)
    stopwords_list.append(stopword)
    
words = []

# Generate a list of all common words
for i in range(0, df.shape[0]):
    abstract = df["abstract"][i] # Split when there is a space
    cleaned_abstract = removePunctuation(abstract)
    words_abstract = cleaned_abstract.split(" ")
    for word in words_abstract:
        if word not in stopwords_list:
            words.append(word)

# Remove duplicate words and sort them alphabetically
words = list(np.unique(np.sort(words)))


# Creating a frequency table with the counted number of each word for each abstract
# Please note that this section will take a long time to run - use myframe.pkl instead

text_boolean_df = pd.DataFrame( np.zeros((df.shape[0], len(words) + 1), dtype = int) , columns = words + ["theclass"])
text_boolean_df["theclass"] = df["class"]

for i in range(0, df.shape[0]):
    abstract = df["abstract"][i]
    clean_abstract = remove_punctuation(abstract)
    words_abstract = clean_abstract.split(" ")
    
    for j in range(0, len(words_abstract)):
        word = words_abstract[j]
        if word not in stopwords_list:
            text_boolean_df[word][i] += 1     

# Saving the data frame for future use
text_boolean_df.to_pickle('myframe.pkl')

#text_boolean_df = pd.read_pickle('myframe.pkl')
#text_boolean_df.head()

# Setting up the dictionary which includes the conditional probabilities for each word in a given class
probabilities = {'B':{},'A':{},'E':{},'V':{}}

# Calculating the priors
num_b = text_boolean_df['theclass'][text_boolean_df['theclass'] == 'B'].count()
num_a = text_boolean_df['theclass'][text_boolean_df['theclass'] == 'A'].count()
num_e = text_boolean_df['theclass'][text_boolean_df['theclass'] == 'E'].count()
num_v = text_boolean_df['theclass'][text_boolean_df['theclass'] == 'V'].count()
total_rows = text_boolean_df['theclass'].count()
p_b = num_b/total_rows
p_a = num_a/total_rows
p_e = num_e/total_rows
p_v = num_v/total_rows

# Calculating all the unique words in the vocabulary
vocab_size = len(list(np.unique(np.sort(words))))

# The total number of word occurrences in a given class
b_size = 0
a_size = 0
e_size = 0
v_size = 0

def sumOfCounts(data,colname,target):
    newSum = 0
    newSum = data[colname][data['theclass'] == target]
    sumTotals = sum(newSum)
    return sumTotals

for word in text_boolean_df.columns[:-1]:
    count_b = sumOfCounts(text_boolean_df,word,'B')
    b_size += count_b
    count_a = sumOfCounts(text_boolean_df,word,'A')
    a_size += count_a
    count_e = sumOfCounts(text_boolean_df,word,'E')
    e_size += count_e
    count_v = sumOfCounts(text_boolean_df,word,'V')
    v_size += count_v
    
# The number of times the word occurs in a given class
# Calculating all the conditonal probabilities
for word in text_boolean_df.columns[:-1]:
        probabilities['B'][word] = {}
        probabilities['A'][word] = {}
        probabilities['E'][word] = {}
        probabilities['V'][word] = {}
        
        count_ct_0 = sumOfCounts(text_boolean_df,word,'B')
        probabilities['B'][word] = (count_ct_0 + 1) / (b_size + vocab_size)
        count_ct_1 = sumOfCounts(text_boolean_df,word,'A')
        probabilities['A'][word] = (count_ct_1 + 1) / (a_size + vocab_size)
        count_ct_2 = sumOfCounts(text_boolean_df,word,'E') 
        probabilities['E'][word] = (count_ct_2 + 1) / (e_size + vocab_size)
        count_ct_3 = sumOfCounts(text_boolean_df,word,'V')
        probabilities['V'][word] = (count_ct_3 + 1) / (v_size + vocab_size)

def classify(abstracts):
    print("Processing the test abstracts...")
    result = ''
    results_list = []
    for row in abstracts:
        # Log the probabilities because they are getting too small
        pr_b = math.log(p_b)
        pr_a = math.log(p_a)
        pr_e = math.log(p_e)
        pr_v = math.log(p_v)
        cleaned_test_abstract = removePunctuation(row)
        row_abstract = cleaned_test_abstract.split(" ")
        # Calculating the posterior probabilities
        for w in row_abstract:
            if w in probabilities['B'] and w not in stopwords_list:
                pr_b += math.log(probabilities['B'][w])
            if w in probabilities['A'] and w not in stopwords_list:
                pr_a += math.log(probabilities['A'][w])
            if w in probabilities['E'] and w not in stopwords_list:
                pr_e += math.log(probabilities['E'][w])
            if w in probabilities['V'] and w not in stopwords_list:
                pr_v += math.log(probabilities['V'][w])
        
            if w not in probabilities['B'] and w not in stopwords_list:
                pr_b += math.log(1/(b_size + vocab_size))
                
            if w not in probabilities['A'] and w not in stopwords_list:
                pr_a += math.log(1/(a_size + vocab_size))
            
            if w not in probabilities['E'] and w not in stopwords_list:
                pr_e += math.log(1/(e_size + vocab_size))
                
            if w not in probabilities['V'] and w not in stopwords_list:
                pr_v += math.log(1/(v_size + vocab_size))
        
        print("Classifying the test abstracts...")
        if pr_b > pr_a and pr_b > pr_e and pr_b > pr_v:
            result = 'B'
        elif pr_a > pr_e and pr_a > pr_v:
            result = 'A'
        elif pr_e > pr_v:
            result = 'E'
        else:
            result = 'V'
        print(pr_b, pr_a, pr_e, pr_v, result)
        results_list.append(result)
    return results_list

# Load in the test set .csv
test_set = pd.read_csv("tst.csv")

# Apply the model to the test set
test_set_class_predictions = classify(test_set["abstract"])
test_set["class"] = test_set_class_predictions

# Write the test set classifications to a .csv so it can be submitted to Kaggle
test_set.drop(["abstract"], axis = 1).to_csv("tst_kaggle.csv", index=False)
