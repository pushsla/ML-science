from keras.models import Sequential
from keras.layers import Dense, ActivityRegularization, Dropout
import pandas as ps
import numpy as np
import pickle
#import re
#from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords


def clean_ht(row):
    if row['category_name'] in ['Магазины', 'Бренды']:
        row['category_name'] = None
    return row

def toid(row):
    global name2id
    name = row['category_name']
    row['category_name'] = name2id[name]
    return row

def toname(row):
    global id2name
    id = row['category_name']
    row['category_name'] = id2name[id]
    return row

def nn_batch_generator(X_data, y_data, batch_size):
    samples_per_epoch = X_data.shape[0]
    number_of_batches = samples_per_epoch/batch_size
    counter=0
    index = np.arange(np.shape(y_data)[0])
    while 1:
        index_batch = index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X_data[index_batch,:].todense()
        y_batch = y_data[index_batch]
        counter += 1
        yield np.array(X_batch),y_batch
        if (counter > number_of_batches):
            counter=0


df = ps.read_csv('all_posts_with_eda.csv')[['processed_text','category_name']]
df = df.apply(clean_ht, axis=1).dropna()
cats = df['category_name'].unique()
name2id = dict(zip(cats, range(len(cats))))
keys = list(name2id.keys())
id2name = dict(zip([name2id[k] for k in keys], keys))
df = df.apply(toid, axis=1)

dftest = df.sample(frac=0.1, random_state=8821)
dfwork = df.drop(dftest.index)
dftrain = dfwork.sample(frac=0.85, random_state=92212)
dfvalid = dfwork.drop(dftrain.index)

stops = stopwords.words('english') + stopwords.words('russian')
document = dfwork['processed_text'].values
vcr = TfidfVectorizer(max_features=30000, stop_words=stops, min_df=10, max_df=0.9)
vcr.fit(document)

document = dftrain['processed_text'].values
X_tr = vcr.transform(document)
document = dfvalid['processed_text'].values
X_va = vcr.transform(document)
X_va = X_va.toarray()

document_te = dftest['processed_text'].values
X_te = vcr.transform(document_te)
X_te = X_te.toarray()
cats_te = dftest['category_name']
catset_te = list(set(cats_te))
cat_ints_te = [catset_te.index(i) for i in cats_te]

y_tr = to_categorical(dftrain['category_name'])
y_va = to_categorical(dfvalid['category_name'])
y_te = to_categorical(dftest['category_name'])


msl = X_tr.shape[1]
ncl = y_tr.shape[1]
model = Sequential()
model.add(Dense(100, input_shape=(msl,), activation='tanh')) ## may use relu
model.add(Dropout(0.2))
model.add(Dense(86, activation='relu'))
model.add(Dense(6, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
m_hist = model.fit_generator(nn_batch_generator(X_tr, y_tr, 60), epochs=5, steps_per_epoch=X_tr.shape[0]//60, validation_data=(X_va, y_va))


model.save('model.h5')
with open('tfidf.sav', 'wb') as f:
    pickle.dump(vcr, f, pickle.HIGHEST_PROTOCOL)
with open('id2name.sav', 'wb') as f:
    pickle.dump(id2name, f, pickle.HIGHEST_PROTOCOL)
with open('name2id.sav', 'wb') as f:
    pickle.dump(name2id, f, pickle.HIGHEST_PROTOCOL)
np.save('X_tr.npy', X_tr)
np.save('y_tr.npy', y_tr)
np.save('X_te.npy', X_te)
np.save('y_te.npy', y_te)
np.save('X_va.npy', X_va)
np.save('y_va.npy', y_va)
