#!/usr/bin/python

"""This module is the base of ml-scince toolkit."""

import collections
import csv
import getpass
import os
import random
import re

from nltk.corpus import stopwords

import numpy as np

import pandas as ps

from pymorphy2 import MorphAnalyzer

import vk_api

print('sir_liba.help()')


def help():
    """Help function."""
    page = """
        Hello. This is sir_liba. Best friend of data sciencer, which loves vk
        So, what this liba can? Vk parsing and processing EDA with parsed data.
        liba have two classes:
            VkParse
                will parse vk.com for you. can save some data to csv files.
                use VkParse.help() for more info.
            EdaMaker
                will process EDA with loaded by VkParse data.
                VkParse creates compatible csv.
                If you want process EDA with other csv,
                        you have to change your csv for
                make it compatible.
                use EdaMaker.help() for more info.

        P.S. you have to install libraries:
            pandas
            numpy
            vk_api
            nltk(and data for it)
            pymorphy2 and pymorphy2-dict-ru

        P.P.S. It is assumed that you are familiar with the basics
            of data analysis and the python language.
        This library is NOT a high-level data analysis tool.
        It is designed to speed up the writing of the code for the project
            'sirius/call-center',
        or any project related to the records in the vkontakte groups,
        provided that you understand what is happening inside of this library.

        """
    print(page)


class Word2vec:
    """Class to work with W2V."""

    def __init__(self, w2v_origin, dataframe_origin, arg_dropnan=True,
                 arg_text_series='processed_text', ):
        """Initialise."""

        self.w2v_origin = np.load(w2v_origin)
        self.text_data = ps.read_csv(dataframe_origin)[arg_text_series]


class EdaMaker:
    """Clean text data and perform some DataAnalysis."""

    def help(self):
        """Help for EdaMaker."""

        page = """
            Hello. This is EDAMAKER part of sir_liba.
            It will make some EDA on your vkontakte dataframe.
            It will be wonderfull,
            if your dataframe was created by sir_liba too.
            So, your dataframe MUST have these columns:
                'group_name' -- name of group, from which row of dataframe
                'text' -- text of publication
                When you make object of this class, you will see construction.
                You MUST check out columns names.

            So, this class have only three useable for you methods:
                cleantext(arg_regexp,  arg_stopwords)
                    will make some EDA on your dataframe.
                    will remove many trash from texts
                    will remove 'stopwords' from texts

                save(arg_name)
                    will save last modification of your dataframe

                makewords(ret=False)
                    will make BagOfWords(BOW) for dataframe, stored in memory.
                    will make WordByFrequency(WBF) for dataframe, stored.
                    WILL NOT clean dataframe.
                    if 'ret' was set as True, will return BOW and WBF

                __init__(filename, text_series='text') (when you create object)
                    will initiale EDA maker with filename. default 'text'.

            How to use it?
            ok, you have dataframe. Lets make class object:
                eda = EdaMaker(your_csv_to_eda)
                    you will see configuration of dataframe.
                    you MUST CHECK IT TI HAVE 'group_name' and 'text'

                eda.cleantext()
                    will clean your texts

                bagofwords, wordbyfreq = eda.makewords(ret=True)
                       or
                eda.makewords()
                bagofwords, wordbyfreq = eda.bow, eda.freqw
                    will return you BOW and WBF

                eda.save(where_to_save)
                    will save new, better dataframe to disk

            Tips:
                if you working under jupyter :), you can check your dataframes:
                    eda.origin -- first, not modified version of dataframe.
                    eda.current -- current version of dataframe
                    eda.stopwords -- list of stopwords, for cleantext()
                     If you want more, see code))))
            """
        print(page)

    def __init__(self, filename, arg_dropnan=True, text_series='text',
                 processed_series='processed_text', length_series='length',
                 group_series='group_name', cat_series='category_name'):
        """Initialise variables."""
        csv = ps.read_csv(filename, encoding='utf-8')

        try:
            # if text_series column exists
            test_data = csv.dropna(subset=[text_series])[text_series]
            test_string = test_data[len(test_data) // 2]
            if test_string.isnumeric():
                print("WARN: your test data seems to NOT be plain text")
        except KeyError:
            print("ERR: column '", text_series, "'does not exists.")
            print("YOUR CSV:", csv[:0])
            return None

        self.file = filename
        self.dropnan = arg_dropnan
        self.text_series = text_series
        self.processed_series = processed_series
        self.length_series = length_series
        self.group_series = group_series
        self.cat_series = cat_series
        if self.dropnan:
            csv = csv.dropna(subset=[text_series])
        print("YOUR CSV:", csv[:0])
        self.origin = csv
        self.current = csv.copy()
        self.bow = {}
        self.freqw = []
        self.stopwords = []
        self.row_pr_posts = ''
        self.all_text = ''
        self.all_pubs = ''

    def remshort(self, row, minlen=7):
        """Remove too short texts."""
        if len(row[self.processed_series].split(' ')) >= minlen:
            return row
        else:
            row[self.processed_series] = None
            return row

    def lemm(self):
        """Lemmatize words"""
        morph = MorphAnalyzer()
        serie = []
        # total = len(self.current[self.processed_series])
        i = 1
        for text in self.current[self.processed_series]:
            serie.append(' '.join([morph.parse(i)[0].normal_form for i in text.split(' ')]))
            i += 1
        self.current[self.processed_series] = serie

    def normalize(self):
        """Make equal texts in every category"""
        sizes = dict(self.current[self.cat_series].value_counts())
        print(sizes)
        fres = sorted(sizes.keys(), key=lambda x: sizes[x])
        print(fres)
        n_size = (sizes[fres[0]] + sizes[fres[1]]) // 2
        print(n_size)
        rdy = ps.DataFrame(columns=list(self.current.keys()))
        for cat in sizes.keys():
            df = self.current[self.current[self.cat_series].isin([cat])]
            df.reindex(np.random.permutation(df.index))
            df = df[:n_size]
            rdy = ps.concat([rdy, df])
        self.current = rdy.copy()
        del(rdy)

    def cleantext(self, arg_regexp=r'', arg_stopwords=['russian'],
                  lemm=True, normalize=True, remshort=True, delstops=True):
        """Clean text from trash and stopwords."""
        print('Len of DataFrame was:', len(self.current.index))
        regexp = arg_regexp
        regexp += r'\\n|\n|"|-|:|–|\/|—|#|«|»|\?|\+|\-|°|\!|\(|\)|_\w*'  # \n symbols, kavichki`s, other trash symbols.
        regexp += r"|\[[^\]]*\]|'"  # all in []
        regexp += r'|\x96|\x85|\n|https?\:\/\/[\S]*[\r\n]*'  # \x85,\x96 symbol and urls
        regexp += r'|<.*>'  # html tags, times, geoips
        regexp += r'|\\[en].*'  # escape symbols
        regexp += r'|[0-9]+:[0-9]+'  # timecodes
        regexp += r'|[\!\.]'  # i`ve forgot...
        regexp += r'|:[\(\)]'  # smiles
        regexp += r'|\S*@\S+.\S*|@'  # @user definitions and e-mails
        regexp += r'|\*.*\*'  # starts: smiles, ital. fonts
        regexp += r'[^a-z,0-9]'

        try:
            self.sender_list = list(self.current[self.group_series])
        except KeyError:
            self.sender_list = []

        if delstops:
            print("Make stopwords...")

            self.post_list = list(self.current[self.text_series])
            self.row_posts = ' '.join(self.post_list)
            for i in arg_stopwords:
                self.stopwords += stopwords.words(i)
            self.stopwords += ['это', 'который', 'который',
                               'которая', 'которые', 'которых']

        print('Delete some trash...')

        self.origin_series = self.current.dropna(subset=[self.text_series])
        self.origin_series = self.origin_series[self.text_series]
        self.current[self.processed_series] = [re.sub(regexp, ' ', i, flags=re.MULTILINE).lower() for i in self.origin_series]
        self.current[self.processed_series] = [re.sub(r'[^a-zA-Zа-яА-Я0-9]', ' ', i, flags=re.MULTILINE).lower() for i in self.current[self.processed_series]]
        textis = [list(i.split()) for i in self.current[self.processed_series]]
        texts = []
        if delstops:
            print("Delete stopwords...")
            for text in textis:
                texts.append([i for i in text if i not in self.stopwords])
            self.current[self.processed_series] = [' '.join(i) for i in texts]
            self.current[self.length_series] = [len(i.split(' ')) for i in self.current[self.processed_series]]

        if remshort:
            print("Remove too short texts...")
            self.current = self.current.apply(self.remshort, axis=1)
            if self.dropnan:

                self.current = self.current.dropna(subset=[self.processed_series])
                self.current = self.current.drop_duplicates(subset=[self.processed_series])

        if lemm:
            print("Make some lemmatzation...")
            self.lemm()

        if normalize:
            print("Make some class normalize...")
            self.normalize()

        self.current = self.current.reset_index(drop=True)

        print('Len of DataFrame become:', len(self.current.index))

    def save(self, arg_name=str(random.randint(123, 24872929))):
        """Save current csv to file."""
        self.current.to_csv(arg_name, index=False, encoding='utf-8')

    def makewords(self, ret=False):
        """Make BagOfWords abd list of words by frequency."""
        self.row_pr_posts = ' '.join(list(self.current[self.origin_series]))
        self.bow = collections.Counter(re.findall('\w+', self.row_pr_posts))
        self.freqw = sorted(list(self.bow.keys()), key=lambda x: self.bow[x], reverse=True)
        self.freqw = [i for i in self.freqw if len(i) > 3]
        self.bow = dict(zip(self.freqw, [self.bow[i] for i in self.freqw]))
        if ret:
            return self.bow, self.freqw

    def makecloud(self, arg_frame=''):
        """Make wordcloud. DEPRECATED."""
        # self.all_text = ' '.join(list(arg_frame['text']))
        # self.all_pubs = ' '.join([str(i) for i in list(arg_frame['group_name'])])
        print("were working hard for this method...")


class VkParse:
    """Parse VK and get data."""

    def help(self):
        """Help."""
        page = """
            Hello! This is VKPARSE part of sir_liba. It will make for you some dirty jobs of vk.com parsing.
            It will be amazing, if you will make EDA by sir_liba too.
            So, what can this class do? ooooh.... many dirty things ;)

            This class have tho types of methods:
                asm_* -- these methods often useless, if you want just to parse vk and save data to csv.
                other -- very useable methods for easy (i mean, with less, less code writing) vk parsing.

            There are some regular methods you can use:
                categoriesinfo()
                    the most easy to understand function. Will return you types of publics and their number.
                    if you have already started, it will help you to choose, ehat type of publics you
                    will parse
                makegrouptable(category_name, topit=True)
                    next method. When you`ve choose tye of publics, set category_name as it. F.E "Блогеры"
                    'topit' set by default as True. If you will set is as False, makegrouptable will use ALL
                    publics with category_name type. If it is True, by default, the least popular publics will
                    not be used. If you call it twice, WILL NOT overwrite existing group table. So, you
                    can call it with different group type names to parse them all.
                savegrouptable(arg_name)
                    if you want to save group table, use this function. by default arg_name will be set like
                    category_name value. It will be saved as arg_name.csv
                scrapposts(total_count=False, init=True, filename=False)
                    when group table WAS CREATED by makegrouptable() method, you can start to download posts.
                    method will download total_count of posts, where from each public will be loaded near
                    total_count//publics_count posts. F.E total_count=100, in table 100 publics. from each -- 1 post
                    init, filename -- you don`t need to change its values actually. If init was set to False,
                    csv file with posts will be NOT saved to your computer. It can be usefull, if you want to process
                    some extra operations. filename usefull in case of init=True. It`s filename to save.
                    if init=True, file will be appended.
                saveposts(filename, writemode)
                    will save scrapped posts to csv 'filename', which will be opened in 'writemode' mode.
                    avialable mods:
                        'a' -- appending
                        'w' -- rewriting

            If you want to use asm_* methods, see the code. Else they will be useless.

            Some examples:

                vk = VkParse(your_vk_login, your_vk_token)
                vk.categoriesinfo()
                    Рекомендации 250
                    VK Fest 24
                    Новости 17
                    Спорт 33
                vk.makegrouptable('Спорт')
                vk.scrapposts()

                sfter that, you will have Спорт_posts.csv. with construction:
                    group_name, group_id, post_id, text, n_likes, category_name
                        ...        ...       ...    ..     ...         ....
                        ...        ...       ...    ..     ...         ....

            Also, there are some usefull variables of class for you:
                self.login -- login
                self.token -- vk token
                self.api -- you can get manually acess to vk api session
                self.table -- you can work with created group table
                self.acc_res -- loaded posts from vk.com.

            """
        print(page)

    def __init__(self, login, token, init=True):
        """Initialise."""
        self.login = login
        self.token = token
        print('enter your password:(will not be stored)')
        self.api = vk_api.VkApi(login=self.login, password=getpass.getpass(), token=self.token)
        self.cat_ids = []
        self.cat_names = []
        self.categories = []
        self.id2cat = {}
        self.cat2id = {}

        try:
            self.api.auth(token_only=True)
        except vk_api.AuthError as msg:
            print("ERR:", msg)

        if init:
            self.asm_initcategories()

    def asm_initcategories(self, subs=0, ext=1):
        """Under-table function to make categories table."""
        with vk_api.VkRequestsPool(self.api) as pool:
            self.categories_info = pool.method('groups.getCatalogInfo',
                                               values={
                                                   'subcategories': subs,
                                                   'extended': ext
                                               })
        self.categories = self.categories_info.result['categories']
        self.cat_ids = list(map(lambda x: x['id'], self.categories))
        self.cat_names = list(map(lambda x: x['name'], self.categories))

        self.id2cat = dict(zip(self.cat_ids, self.cat_names))
        self.cat2id = dict(zip(self.cat_names, self.cat_ids))

        self.table = ps.DataFrame()
        self.oldtable = ps.DataFrame()
        self.table['category_name'] = []
        self.table['category_id'] = []
        self.table['group_name'] = []
        self.table['group_id'] = []
        self.table['n_subs'] = []

    def asm_wallget(self, wall_id=-1, count=10, offset=0):
        """Under-table function to get writings from wall VK."""
        with vk_api.VkRequestsPool(self.api) as pool:
            posts = pool.method(
                'wall.get', values={
                    'owner_id': wall_id,
                    'count': count,
                    'filter': 'owner',
                    'offset': offset
                }
            )
        if posts.ok and posts.ready:
            self.last_type = 0
            self.last = posts.result
            return (True, posts.result)
        else:
            return (False, {})

    def asm_getwalls(self, wall_list, total_count=500, offset=0):
        """Under-table function to get writings from wall list. Uses asm_wallget."""
        # if total_count > 3000:
        #    print('INFO: set total_count as 3000 because of vk api')
        #    total_count = 3000
        result = {}
        per_wall = total_count // len(wall_list) + 1
        for wall in wall_list:
            if type(wall) != int:
                print("ERR:", wall, "Must be integer")
                continue
            elif wall > 0:
                print("WARN:", wall, "Must be less then zero!\n     Try to get publications for", wall * -1)
                wall = wall * -1
            res = self.asm_wallget(wall_id=wall, count=per_wall, offset=offset)
            if res[0]:
                result[wall] = res[1]
            else:
                result[wall] = {}
                print("ERR:", wall, "Bad responce!")
        self.last_type = 1
        self.last = result
        return (True, result)

    def asm_save(self, name=str(random.randint(123, 1198379))):
        """Under-table function to save data."""
        if self.last_type == 0:
            ids = []
            froms = []
            texts = []
            likes = []
            types = []
            reposts = []
            for pub in self.last['items']:
                ids.append(pub['id'])
                froms.append(pub['from_id'])
                texts.append(pub['text'])
                likes.append(pub['likes']['count'])
                types.append(pub['post_type'])
                reposts.append(pub['reposts']['count'])
            frame = ps.DataFrame()
            frame['id'] = ids
            frame['from_id'] = froms
            frame['text'] = texts
            frame['likes'] = likes
            frame['reposts'] = reposts
            frame['type'] = types
        elif self.last_type == 1:
            data_ready = []
            for kee in list(self.last.keys()):
                data_ready.append(self.last[kee])
            ids = []
            froms = []
            texts = []
            likes = []
            types = []
            reposts = []
            for wall in data_ready:
                plen = len(wall['items'])
                ids += [wall['items'][i]['id'] for i in range(plen)]
                froms += [wall['items'][i]['from_id'] for i in range(plen)]
                texts += [wall['items'][i]['text'] for i in range(plen)]
                likes += [wall['items'][i]['likes']['count'] for i in range(plen)]
                types += [wall['items'][i]['post_type'] for i in range(plen)]
                reposts += [wall['items'][i]['reposts']['count'] for i in range(plen)]
            frame = ps.DataFrame()
            frame['id'] = ids
            frame['from_id'] = froms
            frame['text'] = texts
            frame['likes'] = likes
            frame['reposts'] = reposts
            frame['type'] = types

        frame[['id', 'from_id', 'text', 'likes', 'reposts', 'type']].to_csv(name, encoding='utf-8')

    def categoriesinfo(self):
        """Get info about categories."""
        n_groups = 0
        for name in self.cat2id.keys():
            with vk_api.VkRequestsPool(self.api) as pool:
                groups = pool.method('groups.getCatalog', values={
                    'category_id': self.cat2id[name]
                })
            print(name, groups.result['count'])
            n_groups += groups.result['count']
        print('Всего', n_groups)

    def makegrouptable(self, catname, topit=True, alpha=1, gr_num=16):
        """Make table of groups by category."""
        self.catname = catname
        with vk_api.VkRequestsPool(self.api) as pool:
            groups = pool.method('groups.getCatalog', values={
                'category_id': self.cat2id[catname]
            })

        self.g_ids = list(map(lambda x: x['id'], groups.result['items']))[:gr_num]
        self.g_nam = list(map(lambda x: x['screen_name'], groups.result['items']))[:gr_num]
        self.id2group = dict(zip(self.g_ids, self.g_nam))
        self.group2id = dict(zip(self.g_nam, self.g_ids))
        for idd in self.g_ids:
            self.id2cat[idd] = catname

        print('Число подписчиков:')
        self.n_subs = []
        for name in self.group2id.keys():
            with vk_api.VkRequestsPool(self.api) as pool:
                subs = pool.method('groups.getMembers', values={
                    'group_id': self.group2id[name]
                })
            print(name, subs.result['count'])
            self.n_subs += [(self.group2id[name], subs.result['count'])]
        self.dictsubs = dict(self.n_subs)

        if topit:
            tops = np.array(list(self.dictsubs.values())).mean() * alpha
        else:
            tops = 0
        self.selected_groups = list(filter(lambda x: self.dictsubs[x] >= tops, sorted(list(self.id2group.keys()), key=lambda x: self.dictsubs[x], reverse=True)))

        self.oldtable = self.table.copy()

        del(self.table)
        self.table = ps.DataFrame()
        self.table['category_name'] = [catname for i in range(len(self.selected_groups))]
        self.table['category_id'] = [int(self.cat2id[catname]) for i in range(len(self.selected_groups))]
        names = [self.id2group[i] for i in self.selected_groups]
        self.table['group_name'] = names
        self.table['group_id'] = [int(self.group2id[i]) for i in names]
        self.table['n_subs'] = [int(self.dictsubs[i]) for i in self.selected_groups]

        self.table = ps.concat([self.oldtable, self.table])
        self.table = self.table.reset_index(drop=True)

    def cleangrouptable(self):
        """Clean current table of groups."""
        self.table = ps.DataFrame()

    def savegrouptable(self, arg_name=False):
        """Save table of groups to csv."""
        if not arg_name:
            arg_name = self.catname + "_groups.csv"
        # filename = arg_name
        self.table.to_csv(arg_name, index=False)

    def scrapposts(self, total_count=False, init=True, filename=False, offset=0):
        """Get posts from VK."""
        ids = []
        nams = []
        if not total_count:
            total_count = int(len(self.table)) * 2
        if not filename:
            filename = self.catname + "_posts.csv"
        for index in list(self.table.index):
            ex_id, ex_name = self.table['group_id'][index], self.table['group_name'][index]
            ids.append(int(-ex_id))
            nams.append(ex_name)
        self.id2name = dict(zip(ids, nams))
        totalres = []
        result = self.asm_getwalls(wall_list=ids, total_count=total_count, offset=offset)
        totalres.append(result[1])
        acc_res = []
        for dct in totalres:
            grps = list(dct.keys())
            for grp in grps:
                acc_res.append(dct[grp])

        self.acc_res = acc_res

        if init:
            self.saveposts(filename)

    def saveposts(self, filename=False, write_mode='a'):
        """Save scrapper from VK posts."""
        if not filename:
            filename = self.catname + "_posts.csv"
        csvfile = open(filename, write_mode, encoding='utf-8')
        csvwriter = csv.writer(csvfile)

        # Если он пустой, записываем названия столбцов
        if os.stat(filename).st_size == 0:
            csvwriter.writerow(['group_name', 'group_id', 'post_id', 'text', 'n_likes', 'category_name'])

        # Записываем каждый пост в таблицу
        for i in range(len(self.acc_res)):
            for post in self.acc_res[i]['items']:
                csvwriter.writerow([
                    self.id2name[int(post['owner_id'])], post['from_id'], post['id'],
                    post['text'], post['likes']['count'], self.id2cat[int((-1) * post['owner_id'])]
                ])
        csvfile.close()


def sdelat_horosho():
    """Make goosd."""
    login = input('your vk login? ')
    token = input('your vk token? ')
    print('your password:')
    parser = VkParse(login, token)
    parser.makegrouptable(random.choice(parser.cat_names))
    parser.savegrouptable()
    parser.scrapposts()

    eda = EdaMaker(parser.catname + "_posts.csv")
    eda.cleantext()
    eda.save()


def wtf_is_this():
    """Just fun."""
    print("""
                Pandas -- for data-kings under their coachww
                Re -- for expression-lords in their holes of words
                Vk_api -- for mortal men doomed to parse
                Liba -- for Vk Lord by his computer
                In the land of Linux, where the daemons runs

                One lib to view them all
                One lib to find them
                One lib to bring them all
                And by the method parse them

                In the land of Linux, where the daemons runs
    """)
