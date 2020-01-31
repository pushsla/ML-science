from keras.models import load_model
from keras import backend as K
import pickle
import numpy as np
from pymorphy2 import MorphAnalyzer
import re
import pandas as pd
import tensorflow as tf
from collections import Counter
import time
print("""
your_name = KeLoss()
your_name.predict(msg_text, wlen=min_word_len(default 3)) to get class of msg.
    will return (class_name, class_id) tuple
your_name.remorph(msg_text, wlen=min_word_len(default 3)) to get lemmatized msg_text
    useable for w2v ranging =)
""")


class KeLoss:
    def __init__(self, model='bin/models/model.h5',
                 vectr='bin/models/tfidf.sav',
                 # vec300='bin/models/tfidf_for_w2v_300.sav',
                 id2nam='bin/dat/id2name.sav',
                 nam2id='bin/dat/name2id.sav',
                 w2v_f='bin/dat/word2vec300-16-new.npy',
                 avg_w2v_f='bin/dat/avg_w2v300-16-new.npy', updw2v=False, newlog=True):
        self.model = load_model(model)
        self.model._make_predict_function()
        # with open(vec300, 'rb') as f:
        #    self.vec300 = pickle.load(f)
        with open(vectr, 'rb') as f:
            self.vectr = pickle.load(f)
            print("Loaded vectorizer")
        with open(id2nam, 'rb') as f:
            self.id2nam = pickle.load(f)
            print("Loaded id2name")
        with open(nam2id, 'rb') as f:
            self.nam2id = pickle.load(f)
            print("Loaded name2id")
        self.morph = MorphAnalyzer()
        self.word2index = {}
        with open('word_indices.txt', 'r', encoding='utf-8') as fout:
            for line in fout:
                self.word2index[line.split()[0]] = line.split()[1]
        print("Loaded word2index")

        self.WORD2VEC = np.load(w2v_f)
        print("Loaded WORD2VEC")
        self.df = pd.read_csv("all_posts_with_eda.csv")
        self.df = self.df[(self.df['category_name'] != 'Магазины') & (self.df['category_name'] != 'Бренды')]
        self.df = self.df.reset_index()
        sself.mainshape = self.WORD2VEC.shape[1]
        if newlog:
            self.log = pd.DataFrame({'date': [], 'request': [], 'resp_group_id': [], 'resp_post_id': [], 'response': []})
        else:
            self.log = pd.read_csv('chat_log.csv', encoding='utf-8')
        print("DBG:MAINSHAPE:", self.mainshape)
        print("Loaded DataFrame")
        if updw2v:
            self.AVG_W2V = np.empty((self.df.shape[0], self.mainshape))
            #i = 0
            for (i, row) in self.df.iterrows():
                lof = row['processed_text'].split(' ')
                #need_words = Counter(lof)
                #need = sorted(dict(need_words).items(), key=lambda x: x[1]) [::-1]
                #r = [i[1] for i in need[:10]]
                k = 0
                c_mean = np.zeros((self.mainshape,))
                for word in lof:
                    if word in self.word2index:
                        c_mean += self.WORD2VEC[int(self.word2index[word])]
                        k += 1

                self.AVG_W2V[i] = c_mean / k
                #i += 1
            np.save(avg_w2v_f, self.AVG_W2V)
        else:
            self.AVG_W2V = np.load(avg_w2v_f)
            print("Loaded AVG_W2V")

    def remorph(self, text, wlen=3):
        text = text.lower()
        text = re.sub('[^a-zа-я ]', ' ', text)
        print("DBG:PRDCT:REMORPH:TEXT", text)
        text = ' '.join([i for i in text.lower().split(' ') if len(i) > wlen])
        print("DBG:PRDCT:REMORPH:TEXT", text)
        return ' '.join([self.morph.parse(i)[0].normal_form for i in text.split(' ')])

    def predict(self, text, wlen=3):
        text0 = self.remorph(text, wlen)
        print(text0)
        text = self.vectr.transform([text0]).toarray()
        print("DBG:PRDCT:VECTORIZED")
        # K.clear_session()
        #predicted = self.id2nam[np.argmax(self.model.predict(text))]

        # return (predicted, self.nam2id[predicted])
        predicted = self.model.predict(text)
        print("DBG:PRDCT:", predicted)
        a = dict(zip(self.nam2id.keys(), predicted[0]))
        b = dict(zip(predicted[0], self.nam2id.keys()))
        return (a, b, text0)

    def get_post(self, s):
        #s = self.remorph(s)
        name_to_per, per_to_name, cute_text = self.predict(s)
        #category = per_to_name[sorted(per_to_name.keys(), reverse=True)]
        category = sorted(name_to_per.keys(), key=lambda x: name_to_per[x], reverse=True)[0:2]
        print("DBG:PERCENTS:0:", name_to_per[category[0]])
        print("DBG:PERCENTS:1:", name_to_per[category[1]])
        print("DBG:CATEGORY:PRE:", category)
        if name_to_per[category[0]] * 100 - name_to_per[category[1]] * 100 > 5:
            category = [category[0], category[0]]
        print("DBG:category", category)

        #word2index = {}
        # with open('word_indices.txt', 'r', encoding='utf-8') as fout:
        #    for line in fout:
        #        word2index[line.split()[0]] = line.split()[1]

        #WORD2VEC = np.load('word2vec100.npy')

        #df = pd.read_csv('all_posts_with_eda.csv')
        #df = df[(df['category_name'] != 'Магазины') & (df['category_name'] != 'Бренды')]
        # AVG_W2V = np.empty((self.df.shape[0], 100)
        k = 0
        c_mean = np.zeros((self.mainshape,))
        for word in cute_text.split(' '):
            if word in self.word2index:
                c_mean += self.WORD2VEC[int(self.word2index[word])]
                k += 1
        res = c_mean / k
        df_now = self.df[self.df['category_name'] == category[0]]
        df_now = pd.concat([df_now, self.df[self.df['category_name'] == category[1]]])
        distance = np.empty((df_now.shape[0], 1))
        now_i = 0
        print("DBG:W2V_SHAPE:", self.AVG_W2V.shape)
        print("DBG:DF_SHAPE:", self.df.shape)
        print("DBG:DF_SHAPE:", df_now.shape)
        now_i = 0
        for (i, r) in df_now.iterrows():
            # dist = (self.AVG_W2V[i] - res) ** 2 ## vector
            distance[now_i] = np.linalg.norm(self.AVG_W2V[i] - res)
            now_i += 1
        j = np.argmin(distance)
        print("DBG:J", j)
        toret = (df_now['group_id'].iloc[j], df_now['post_id'].iloc[j])
        log_write = pd.DataFrame({'date': [time.ctime()], 'request': [cute_text], 'resp_group_id': [int(toret[0])], 'resp_post_id': [int(toret[1])], 'response': [df_now['processed_text'].iloc[j]]})
        self.log = pd.concat([self.log, log_write])
        self.log.to_csv('chat_log.csv', encoding='utf-8', index=False)
        # del(j)
        #text  = (df_now['text'].iloc[j], df_now['group_name'])
        #
        # Возвращается  tuple из трех элементов. Первый -- text
        # Второй -- словарь вероятностей. Его нужно включить в вывод gui обязательно!
        #Третий -- group_id, post_id
        #
        percents = 'Ваш запрос "' + cute_text + '" классифицирован:<br>'
        for key in name_to_per.keys():
            percents += str(key) + ": " + str(round(name_to_per[key] * 100)) + "%<br>"
        return ('', percents, toret)

        # MODULE DISTANCE
        #k = 0
        #c_mean = np.zeros((100,))
        # for word in cute_text.split(' '):
        #    if word in self.word2index:
        #        k += 1
        #        c_mean += self.WORD2VEC[int(self.word2index[word])]
        # res = c_mean / k # mean request vector
        #df_now = self.df[self.df['category_name'] == category]
        #distance = np.empty((df_now.shape[0], 1))
        #now_i = 0
        # for i in range(self.AVG_W2V.shape[0]):
        #    if self.df['category_name'].iloc[i] == category:
        #        distance[now_i] = np.linalg.norm(self.AVG_W2V[i] - res)
        #        now_i += 1
        #j = np.argmin(distance)
        #print("DBG:J", j)
        #toret = (df_now['group_id'].iloc[j], df_now['post_id'].iloc[j])

        # COSINE DISTANCE
        # for (i, r) in df_now.iterrows():
        #    #distance[now_i] = np.linalg.norm(self.AVG_W2V[i] - res) -- module distance
        #    scalar = np.sum(self.AVG_W2V[i] * res)
        #    mvec = np.sqrt(np.sum(self.AVG_W2V[i] ** 2)) * np.sqrt(np.sum(res ** 2)) # cosinus distance
        #    distance[now_i] = scalar/mvec
        #    now_i += 1
        #j = np.argmin(distance)
        #print("DBG:J", j)

        # AVG COLCULATING USE MEAN
        #i = 0
        # for (_, row) in self.df.iterrows():
        #    k = 0
        #    lof = row['processed_text'].split(' ')
        #    need_words = Counter(lof)
        #    need = sorted(dict(need_words).items(), key=lambda x: x[1]) [::-1]
        #    r = [i[1] for i in need[:10]]
        #    c_mean = np.zeros((100,))
        #    for word in lof:
        #        if word in self.word2index:
        #            k += 1
        #            c_mean += self.WORD2VEC[int(self.word2index[word])]#

        #    self.AVG_W2V[i] = c_mean / k
        #    i += 1
