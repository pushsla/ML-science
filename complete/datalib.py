"""Docstring."""

import numpy as np


# class UseModel:
#     """Docstring."""

#     from keras.models import load_model
#     from keras import backend as K
#     import pickle
#     import numpy as np
#     from pymorphy2 import MorphAnalyzer
#     import re
#     import pandas as ps
#     import tensorflow as tf
#     from collections import Counter
#     import time

#     def __init__(self, vectorizer, model, word2vec, avg_w2v, dataframe, read_mode='f'):
#         if read_mode[0] == 'v':
#             self.vcr = vectorizer
#             self.model = model
#             self.w2v = word2vec
#             self.aw2v = avg_w2v
#             self.df = dataframe
#         elif read_mode[0] == 'f':
#             self.model = load_model(model)
#             with open(vectorizer, 'rb') as vcr:
#                 self.vcr = pickle.load(vectorizer)
#             self.w2v = np.load(word2vec)
#             self.aw2v = np.load(avg_w2v)
#             self.df = self.ps.read_csv(dataframe)
#         else:
#             raise KeyError('`read_mode` value out of range')

#         self.model._make_predict_function()
#         self.df = self.df.reset_index()
#         self.mainshape = self.w2v.shape[1]

#     def lemmatize(self):
#         pass


class WordToVec:
    """Docstring."""

    import pandas as ps

    def __init__(self):
        """Docstring."""
        self.origin = None

    def average(self, dataframe, read_mode='f',
                bool_reload=False, arg_filename_w2v='', arg_filename_indices='',
                arg_serie_to_compile='processed_text',
                arg_serie_to_exclude='category_name', arg_excluded_series=[]):
        """Doc."""
        if bool_reload:
            self.origin = np.load(arg_filename_w2v)
            self.word2index = np.load(arg_filename_indices)
            self.mainshape = self.origin.shape[1]

        if read_mode[0] == 'v':
            if read_mode[1] == 'c':
                self.df = dataframe.copy()
            else:
                self.df = dataframe
        elif read_mode[0] == 'f':
            csv = self.ps.read_csv(dataframe, encoding='utf-8')
        else:
            raise KeyError('`read_mode` value out of range')

        # df = df[(df[arg_serie_to_exclude] not in arg_excluded_series)]
        df = df.reset_index()

        self.average = np.empty((df.shape[0], self.mainshape))
        # i = 0
        for (i, row) in df.iterrows():
            lof = row[arg_serie_to_compile].split(' ')
            # need_words = Counter(lof)
            # need = sorted(dict(need_words).items(), key=lambda x: x[1]) [::-1]
            # r = [i[1] for i in need[:10]]
            k = 0
            c_mean = np.zeros((self.mainshape,))
            for word in lof:
                pos = np.where(self.word2index == word)
                if len(pos[0]) != 0:
                    c_mean += self.origin[pos[0][0]]
                    k += 1

            self.average[i] = c_mean / k
            # i += 1

    def save(self, filename):
        np.save(arg_saveto, self.average)


class VkPosts:
    """Dosctring."""

    import getpass
    import vk_api
    import os
    import pandas as ps
    import csv
    import random

    def __init__(self, login, token, init=True):
        """Initialise."""
        self.login = login
        self.token = token
        print('vk password:')
        self.api = self.vk_api.VkApi(login=self.login,
                                     password=self.getpass.getpass(),
                                     token=self.token
                                     )
        self.cat_ids = []
        self.cat_names = []
        self.categories = []
        self.id2cat = {}
        self.cat2id = {}

        # Try to authorize in VK.
        try:
            self.api.auth(token_only=True)
        except self.vk_api.AuthError as msg:
            # print("ERR:", msg)
            raise self.vk_api.AuthError("MLSKIT:DataScrap:DataScrap:\
                                    __init__:VKAUTH:\n\t{}"
                                        .format(msg)
                                        )
        # Initiate basic variables for data processing if specefied.
        if init:
            self.asm_initcategories()

    def asm_initcategories(self, subs=0, ext=1):
        """Docstring."""
        with self.vk_api.VkRequestsPool(self.api) as pool:
            self.categories_info = pool.method(
                'groups.getCatalogInfo',
                values={
                    'subcategories': subs,
                    'extended': ext
                }
            )
        self.categories = self.categories_info.result['categories']
        self.cat_ids = list(map(lambda x: x['id'], self.categories))
        self.cat_names = list(map(lambda x: x['name'], self.categories))

        self.id2cat = dict(zip(self.cat_ids, self.cat_names))
        self.cat2id = dict(zip(self.cat_names, self.cat_ids))

        self.table = self.ps.DataFrame()
        self.oldtable = self.ps.DataFrame()
        self.table['category_name'] = []
        self.table['category_id'] = []
        self.table['group_name'] = []
        self.table['group_id'] = []
        self.table['n_subs'] = []

    def asm_wallget(self, wall_id=-1, count=10, offset=0):
        """Under-table function to get writings from wall VK."""
        with self.vk_api.VkRequestsPool(self.api) as pool:
            posts = pool.method(
                'wall.get',
                values={
                    'owner_id': wall_id,
                    'count': count,
                    'filter': 'owner',
                    'offset': offset
                }
            )
        if posts.ok and posts.ready:
            self.last_type = 0  # This variable will be used during save.
            self.last = posts.result
            return (True, posts.result)
        else:
            return (False, {})

    def asm_getwalls(self, wall_list, total_count=500, offset=0):
        """Under-table function to get writings from wall list. Uses asm_wallget."""
        # Check if specefied value isnt overhead vk_api.
        # if total_count > 3000:
        #    print('INFO: set total_count as 3000 because of vk api')
        #    total_count = 3000
        result = {}
        per_wall = total_count // len(wall_list) + 1
        for wall in wall_list:
            # Some safety tests.
            if type(wall) != int:
                print("MLSKIT:DataScrap:asm_getwalls:ERR:", wall, "Must be integer")
                continue
            elif wall > 0:
                print("MLSKIT:DataScrap:asm_getwalls:WARN:", wall, "Must be < 0")
                print("\tTry to get publications for", wall * -1)
                wall = wall * -1
            res = self.asm_wallget(wall_id=wall, count=per_wall, offset=offset)
            if res[0]:
                result[wall] = res[1]
            else:
                result[wall] = {}
                raise ValueError("MLSKIT:DataScrap:asm_getwalls:\
                                 {} has returned bad responce"
                                 .format(wall)
                                 )
        self.last_type = 1  # This variable will be used during save.
        self.last = result
        return (True, result)

    def asm_save(self, name=str(random.randint(123, 1198379))):
        """Under-table function to save data."""
        # Check if there was 'wallget' or 'getwalls'.
        # These ^ sunctions returns different data types.
        if self.last_type == 0:
            # If it was 'wallget', so data type is pandas dataframe.
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
            frame = self.ps.DataFrame()
            frame['id'] = ids
            frame['from_id'] = froms
            frame['text'] = texts
            frame['likes'] = likes
            frame['reposts'] = reposts
            frame['type'] = types
        elif self.last_type == 1:
            # If it was 'getwalls', so data types is list.
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
            frame = self.ps.DataFrame()
            frame['id'] = ids
            frame['from_id'] = froms
            frame['text'] = texts
            frame['likes'] = likes
            frame['reposts'] = reposts
            frame['type'] = types

        frame[['id',
               'from_id',
               'text',
               'likes',
               'reposts',
               'type'
               ]].to_csv(name, encoding='utf-8')

    def categoriesinfo(self):
        """Get info about categories."""
        n_groups = 0
        for name in self.cat2id.keys():
            with self.vk_api.VkRequestsPool(self.api) as pool:
                groups = pool.method('groups.getCatalog', values={
                    'category_id': self.cat2id[name]
                })
            print(name, groups.result['count'])
            n_groups += groups.result['count']
        print('Всего', n_groups)

    def makegrouptable(self, catname, topit=True, alpha=1, gr_num=16):
        """Make table of groups by category."""
        self.catname = catname
        with self.vk_api.VkRequestsPool(self.api) as pool:
            groups = pool.method(
                'groups.getCatalog',
                values={'category_id': self.cat2id[catname]}
            )

        self.g_ids = list(map(lambda x: x['id'], groups.result['items']))[:gr_num]
        self.g_nam = list(map(lambda x: x['screen_name'], groups.result['items']))[:gr_num]
        self.id2group = dict(zip(self.g_ids, self.g_nam))
        self.group2id = dict(zip(self.g_nam, self.g_ids))
        for idd in self.g_ids:
            self.id2cat[idd] = catname

        print('MLSKIT:DataScrap:makegrouptable:Subscribers:--')
        self.n_subs = []
        for name in self.group2id.keys():
            with self.vk_api.VkRequestsPool(self.api) as pool:
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
        self.selected_groups = list(filter(lambda x: self.dictsubs[x] >= tops,
                                           sorted(list(self.id2group.keys()),
                                                  key=lambda x: self.dictsubs[x],
                                                  reverse=True
                                                  )
                                           )
                                    )

        self.oldtable = self.table.copy()

        del(self.table)
        self.table = self.ps.DataFrame()
        self.table['category_name'] = [catname for i in range(len(self.selected_groups))]
        self.table['category_id'] = [int(self.cat2id[catname])
                                     for i in range(len(self.selected_groups))
                                     ]
        names = [self.id2group[i] for i in self.selected_groups]
        self.table['group_name'] = names
        self.table['group_id'] = [int(self.group2id[i]) for i in names]
        self.table['n_subs'] = [int(self.dictsubs[i]) for i in self.selected_groups]

        self.table = self.ps.concat([self.oldtable, self.table])
        self.table = self.table.reset_index(drop=True)

    def cleangrouptable(self):
        """Clean current table of groups."""
        self.table = self.ps.DataFrame()

    def savegrouptable(self, arg_name=False):
        """Save table of groups to csv."""
        if not arg_name:
            arg_name = self.catname + "_groups.csv"
        # filename = arg_name
        self.table.to_csv(arg_name, index=False)

    def scrapposts(self, total_count=None, bool_autosave=False, filename=None, offset=0):
        """Get posts from VK."""
        ids = []
        nams = []
        if not total_count:
            total_count = int(len(self.table)) * 2
        if not filename:
            filename = self.catname + "_posts.csv"
        for index in list(self.table.index):
            ex_id = self.table['group_id'][index]
            ex_name = self.table['group_name'][index]
            ids.append(int(-ex_id))
            nams.append(ex_name)
        self.id2name = dict(zip(ids, nams))
        totalres = []

        result = self.asm_getwalls(wall_list=ids,
                                   total_count=total_count,
                                   offset=offset
                                   )
        totalres.append(result[1])
        acc_res = []
        for dct in totalres:
            grps = list(dct.keys())
            for grp in grps:
                acc_res.append(dct[grp])

        self.acc_res = acc_res

        if bool_autosave:
            self.save(filename)

    def save(self, filename=False, write_mode='a'):
        """Save scrapper from VK posts."""
        if not filename:
            filename = self.catname + "_posts.csv"
        # edecsvfile = open(filename, write_mode, encoding='utf-8')
        with open(filename, write_mode, encoding='utf-8') as csvfile:
            csvwriter = self.csv.writer(csvfile)

            # Check if filename.csv is empty.
            if self.os.stat(filename).st_size == 0:
                # If so, write series names.
                csvwriter.writerow(['group_name',
                                    'group_id',
                                    'post_id',
                                    'text',
                                    'n_likes',
                                    'category_name'
                                    ])

            # And write every post in dataframe
            for i in range(len(self.acc_res)):
                for post in self.acc_res[i]['items']:
                    csvwriter.writerow([
                        self.id2name[int(post['owner_id'])],
                        post['from_id'],
                        post['id'],
                        post['text'],
                        post['likes']['count'],
                        self.id2cat[int((-1) * post['owner_id'])]
                    ])
        # csvfile.close()
