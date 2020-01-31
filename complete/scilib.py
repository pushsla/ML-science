"""Docstring."""

import numpy as np


def remorph_text(self, text, wlen=3):
    """Doc."""
    import re
    from pymorphy2 import MorphAnalyzer

    morph = MorphAnalyzer()

    text = text.lower()
    text = re.sub('[^a-zа-я ]', ' ', text)
    # print("DBG:PRDCT:REMORPH:TEXT", text)
    text = ' '.join([i for i in text.lower().split(' ') if len(i) > wlen])
    # print("DBG:PRDCT:REMORPH:TEXT", text)
    return ' '.join([morph.parse(i)[0].normal_form for i in text.split(' ')])


class NewModel:
    """Docs."""

    from keras.models import Sequential
    from keras.layers import Dense, ActivityRegularization, Dropout
    import pandas as ps
    import numpy as np
    import pickle
    # import re
    # from keras.preprocessing.text import Tokenizer

    def __init__(self, dataframe, read_mode='f',
                 arg_series_xy=['processed_text', 'category_name'],
                 bool_defaults=True):
        if read_mode[0] == 'v':
            if read_mode[1] == 'c':
                self.df = dataframe.copy()
            else:
                self.df = dataframe
        elif read_mode[0] == 'f':
            self.df = self.ps.read_csv(dataframe)[arg_series_xy]
        else:
            raise KeyError('`read_mode` value out of range')

        self.df = self.df.apply(self.clean_ht, axis=1).dropna()
        self.cats = self.df[arg_series_xy[1]].unique()
        self.name2id = dict(zip(self.cats, range(len(self.cats))))
        self.keys = list(self.name2id.workeys())
        self.id2name = dict(zip([self.name2id[k] for k in self.keys], self.keys))
        self.df = self.df.apply(self.toid, axis=1)

        self.model = self.Sequential()

        if bool_defaults:
            self.make_batches()
            self.model.add(self.Dense(100, input_shape=(self.msl,), activation='tanh'))  # or use relu
            self.model.add(self.Dropout(0.2))
            self.model.add(self.Dense(86, activation='relu'))
            self.model.add(self.Dense(self.ncl, activation='softmax'))

    def model_train(self, arg_metrics=['acc'], arg_optimizer='rmsprop',
                    arg_loss='categorical_crossentropy',
                    arg_epochs=10, arg_aplpha=60):
        self.model.compile(optimizer=arg_optimizer,
                           loss=arg_loss,
                           metrics=arg_metrics
                           )
        self.hist = self.model.fit_generator(
            self.nn_batch_generator(self.x_tr, self.y_tr, 60),
            epochs=arg_epochs,
            steps_per_epoch=self.x_tr.shape[0] // arg_aplpha,
            validation_data=(self.x_va, self.y_va)
        )

    def save(self):
        self.model.save('model.h5')
        with open('tfidf.sav', 'wb') as f:
            self.pickle.dump(self.vcr, f, self.pickle.HIGHEST_PROTOCOL)
        with open('id2name.sav', 'wb') as f:
            self.pickle.dump(self.id2name, f, self.pickle.HIGHEST_PROTOCOL)
        with open('name2id.sav', 'wb') as f:
            self.pickle.dump(self.name2id, f, self.pickle.HIGHEST_PROTOCOL)
        np.save('x_tr.npy', self.x_tr)
        np.save('y_tr.npy', self.y_tr)
        np.save('x_te.npy', self.x_te)
        np.save('y_te.npy', self.y_te)
        np.save('x_va.npy', self.x_va)
        np.save('y_va.npy', self.y_va)

    def make_batches(self, bool_remstops=True, arg_custom_stops=[]):

        from sklearn.feature_extraction.text import TfidfVectorizer
        from nltk.corpus import stopwords
        from keras.utils import to_categorical

        dftest = self.df.sample(frac=0.1, random_state=8821)
        dfwork = self.df.drop(dftest.index)
        dftrain = dfwork.sample(frac=0.85, random_state=92212)
        dfvalid = dfwork.drop(dftrain.index)

        if bool_remstops:
            stops = stopwords.words('english') + stopwords.words('russian') + arg_custom_stops
        else:
            stops = []

        document = dfwork['processed_text'].values
        self.vcr = TfidfVectorizer(max_features=30000, stop_words=stops, min_df=10, max_df=0.9)
        self.vcr.fit(document)

        document = dftrain['processed_text'].values
        self.x_tr = self.vcr.transform(document)
        document = dfvalid['processed_text'].values
        self.x_va = self.vcr.transform(document)
        self.x_va = self.x_va.toarray()

        document_te = dftest['processed_text'].values
        self.x_te = self.vcr.transform(document_te)
        self.x_te = self.x_te.toarray()
        cats_te = dftest['category_name']
        catset_te = list(set(cats_te))
        self.cat_ints_te = [catset_te.index(i) for i in cats_te]

        self.y_tr = to_categorical(dftrain['category_name'])
        self.y_va = to_categorical(dfvalid['category_name'])
        self.y_te = to_categorical(dftest['category_name'])

        self.msl = self.x_tr.shape[1]
        self.ncl = self.y_tr.shape[1]

    def clean_ht(self, row):
        if row['category_name'] in ['Магазины', 'Бренды']:
            row['category_name'] = None
        return row

    def toid(self, row):
        name = row['category_name']
        row['category_name'] = self.name2id[name]
        return row

    def toname(self, row):
        id = row['category_name']
        row['category_name'] = self.id2name[id]
        return row

    def nn_batch_generator(self, x_data, y_data, batch_size):
        samples_per_epoch = x_data.shape[0]
        number_of_batches = samples_per_epoch / batch_size
        counter = 0
        index = np.arange(np.shape(y_data)[0])
        while 1:
            index_batch = index[batch_size * counter:batch_size * (counter + 1)]
            x_batch = x_data[index_batch, :].todense()
            y_batch = y_data[index_batch]
            counter += 1
            yield np.array(x_batch), y_batch
            if (counter > number_of_batches):
                counter = 0


class DataProcess:
    """Docstring."""

    import collections
    import random
    import re

    import pandas as ps

    def __init__(self, dataframe, read_mode='f',
                 bool_dropnan=True, text_series='text',
                 processed_series='processed_text', length_series='length',
                 group_series='group_name', cat_series='category_name'):
        """
        The __init__ of DataProcess class object.
           Parameters:
            required:
             filename(string): csv file with text data to precess.
            optional important:
             text_series(string, 'text'): serie name where texts are stored.
            optional:
             bool_dropnan(bool, def.True): drop rows without text.
             processed_series(string, 'processed_text'): serie to store processed texts.
             length_series(string, 'length'): serie to store original texts length.
             --Needed by DataProcess.normalize()
             group_series(string, 'group_name'): serie name where text`s owners names are.
             --Needed by DataProcess.cleantext() for optional unrequired vars
             cat_series(string, 'category_name'): serie name where text`s class names are.
        """
        if read_mode[0] == 'v':
            if read_mode[1] == 'c':
                self.df = dataframe.copy()
            else:
                self.df = dataframe
        elif read_mode[0] == 'f':
            csv = self.ps.read_csv(dataframe, encoding='utf-8')
        else:
            raise KeyError('`read_mode` value out of range')

        try:
            # Test if dataframe contents text_series series.
            test_data = csv.dropna(subset=[text_series])[text_series]
            test_string = test_data[len(test_data) // 2]
            del(test_string)
            print("MLSKIT:DataProcessing:__init__:CSV:\n\t{}".format(csv[:0]))
        except KeyError:
            raise KeyError('''MLSKIT:DataProcessing:__init__:
                Data series {} for {} does not exists'''.format(
                text_series, csv[:0])
            )

        # Drop empty rows if specefied in __init__.
        if bool_dropnan:
            csv = csv.dropna(subset=[text_series])

        # self.file = filename
        self.dropnan = bool_dropnan
        self.text_series = text_series
        self.processed_series = processed_series
        self.length_series = length_series
        self.group_series = group_series
        self.cat_series = cat_series
        self.origin = csv
        self.current = csv.copy()
        # self.bow = {}  # Deprecated from __init__. See them in makewords().
        # self.freqw = []
        self.stopwords = []
        self.row_pr_posts = ''
        self.all_text = ''
        self.all_pubs = ''

    def remshort(self, row, minlen=7):
        """
        Remove rows with too short PROCESSED texts from dataframe.

        Developed specially for pandas.apply() usage.
        CANT be used upon the class exemplar.
        Parameters:
            required:
                row(pandas dataframe row)
            optional:
                minlen(int, 7): minimal length of text in words
                               this variable CANT be set in cleantext().
        """
        if len(row[self.processed_series].split(' ')) >= minlen:
            return row
        else:
            row[self.processed_series] = None
            return row

    def lemmatize(self, bool_isself=False):
        """
        Lemmatize every word in PROCESSED texts.

        Developed for manual usage with stores in object data.
        Used by cleantext() in automatical mode to cut your time.
        Parameters:
            bool_isself(bool, False): marker to create MorphAnalyzer.
                                     If you call it manually set False.
        """
        from pymorphy2 import MorphAnalyzer

        if not bool_isself:
            morph = MorphAnalyzer()
        serie = []
        # Deprecated from lemmatize() because of useless.
        # total = len(self.current[self.processed_series])
        i = 1
        for text in self.current[self.processed_series]:
            serie.append(
                ' '.join([morph.parse(i)[0].normal_form for i in text.split(' ')])
            )
            i += 1
        self.current[self.processed_series] = serie

    def normalize(self):
        """
        Equalize the amount of data by class, if there are several.

        Developed for manual usage with stored in object data.
        Used by cleantext() in automatical mode to cut your time.
        Parameters:
            nop.
        """
        # MAKE AVOIDNESS OF `cat_series` less dataframes!!!
        sizes = dict(self.current[self.cat_series].value_counts())
        print(sizes)
        fres = sorted(sizes.keys(), key=lambda x: sizes[x])
        print(fres)

        if len(fres) > 1:  # Aviod of IndexError with one class in dataframe.
            n_size = (sizes[fres[0]] + sizes[fres[1]]) // 2
        else:
            n_size = sizes[fres[0]]
        print(n_size)

        rdy = self.ps.DataFrame(columns=list(self.current.keys()))
        for cat in sizes.keys():
            df = self.current[self.current[self.cat_series].isin([cat])]
            df.reindex(np.random.permutation(df.index))
            df = df[:n_size]
            rdy = self.ps.concat([rdy, df])
        self.current = rdy.copy()
        del(rdy)

    def delstops(self, arg_stopwords, texts, textis):
        """Docstring."""
        from nltk.corpus import stopwords

        print("MLSKIT:DataProcessing:cleantext:Make stopwords...")
        self.post_list = list(self.current[self.text_series])
        self.row_posts = ' '.join(self.post_list)
        for i in arg_stopwords:
            self.stopwords += stopwords.words(i)
        self.stopwords += ['это', 'который', 'который',
                           'которая', 'которые', 'которых']

        print("MLSKIT:DataProcessing:cleantext:Delete stopwords...")
        for text in textis:
            texts.append([i for i in text if i not in self.stopwords])
        self.current[self.processed_series] = [' '.join(i) for i in texts]
        self.current[self.length_series] = [len(i.split(' '))
                                            for i in self.current[self.processed_series]
                                            ]

    def cleantext(self, arg_regexp=r'', arg_stopwords=['russian'],
                  bool_lemm=True, bool_normalize=True,
                  bool_remshort=True, bool_delstops=True):
        """
        Clean text data in dataframe.

        Optionally performs:
            lemmatize(), normalize(), remshort(), delstops()
        In automatical mode.

        Parameters:
            required:
                nop
            optional:
                arg_regexp(string, r''): if you want to delete custom regular
                                          expression sequences from texts
                arg_stopwords(list, ['russian']): list of languages to delete
                                                   stop - words stored in its
                                                   dictionaries
                bool_lemm(bool, True): make lemmatization of words in texts
                bool_normalize(bool, True): normalize the amount of data
                                             by class if any
                bool_remshort(bool, True): remove texts with length <= 7
                bool_delstops(bool, True): delete stop - words from texts
        """
        print('MLSKIT:DataProcessing:cleantext:Current DF lenght: {}'
              .format(len(self.current.index))
              )
        regexp = arg_regexp
        regexp += r'\\n|\n|"|-|:|–|\/|—|\#|#|«|»|\?|\+|\-|°|\!|\(|\)|_\w*'
        regexp += r"|\[[^\]]*\]|'"  # all in []
        regexp += r'|\x96|\x85|\n|https?\:\/\/[\S]*[\r\n]*'
        regexp += r'|<.*>'  # html tags, times, geoips
        regexp += r'|\\[en].+'  # escape symbols
        regexp += r'|[0-9]+:[0-9]+'  # timecodes
        regexp += r'|[\!\.]'  # i`ve forgot...
        regexp += r'|:[\(\)]'  # smiles
        regexp += r'|\S*@\S+.\S*|@'  # @user definitions and e-mails
        regexp += r'|\*.*\*'  # starts: smiles, ital. fonts
        regexp += r'|[^а-я А-Я]+'

        try:  # Check if group_series exists. If they are -- make sender_list var
            self.sender_list = list(self.current[self.group_series])
        except KeyError:
            self.sender_list = []

        # if bool_delstops:  # Make list of stop-words if specefied.

        print('MLSKIT:DataProcessing:cleantext:Delete some trash...')

        self.origin_series = self.current.dropna(subset=[self.text_series])
        self.origin_series = self.origin_series[self.text_series]
        self.current[self.processed_series] = [self.re.sub(regexp, ' ', i,
                                                           flags=self.re.MULTILINE).lower()
                                               for i in self.origin_series
                                               ]
        self.current[self.processed_series] = [self.re.sub(r'[^a-zA-Zа-яА-Я0-9]',
                                                           ' ',
                                                           i,
                                                           flags=self.re.MULTILINE).lower()
                                               for i in self.current[self.processed_series]
                                               ]
        textis = [list(i.split()) for i in self.current[self.processed_series]]
        texts = []
        # Remove stop-words stored in previously generated list.
        if bool_delstops:
            self.delstops(arg_stopwords, texts, textis)

        # Remove short texts from text_series if specefied.
        if bool_remshort:
            print("MLSKIT:DataProcessing:cleantext:Remove too short texts...")
            self.current = self.current.apply(self.remshort, axis=1)
            if self.dropnan:

                self.current = self.current.dropna(subset=[self.processed_series])
                self.current = self.current.drop_duplicates(subset=[self.processed_series])

        # Lemmatize every word for every text in text_series if specefied.
        if bool_lemm:
            print("MLSKIT:DataProcessing:cleantext:Make some lemmatization...")
            # Make morphAnalyzer for self.lemmatize() function.
            self.lemmatize()

        # Normalize size of classes to mean value if specefied.
        if bool_normalize:
            print("MLSKIT:DataProcessing:cleantext:Make some class normalize...")
            self.normalize()

        self.current = self.current.reset_index(drop=True)

        print('MLSKIT:DataProcessing:cleantext:Current DF length: {}'
              .format(len(self.current.index))
              )

    def save(self, arg_name=str(random.randint(123, 24872929))):
        """
        Save stored in object dataframe to file.

        Options:
            optional:
                arg_name(string, random_int_as_str): filename to save csv
        """
        self.current.to_csv(arg_name, index=False, encoding='utf-8')

    def makewords(self, bool_ret=False):
        """
        Make BagOfWords dist and word_by_frequency list.

        Options:
            optional:
                bool_ret(bool, False): set True if you want this method to
                                        return BOW and WBF as variables.
                                        In other case, they will be only
                                        saved as self.bow and self.freqw
        DEPRECATED FOR NOW BECAUSE OF UNFIXABLE BUG
        """
        self.row_pr_posts = ' '.join(list(self.current[self.text_series]))
        self.bow = self.collections.Counter(self.re.findall('\w+', self.row_pr_posts))
        self.freqw = sorted(list(self.bow.keys()),
                            key=lambda x: self.bow[x],
                            reverse=True
                            )
        self.freqw = [i for i in self.freqw if len(i) > 3]
        self.bow = dict(zip(self.freqw, [self.bow[i] for i in self.freqw]))

        #  Return values if specefied.
        if bool_ret:
            return self.bow, self.freqw
