#!/usr/bin/python
# -*- coding: utf-8 -*-

import vk_api
import getpass
import random
import pandas as ps

def help():
    page = """
    How to use this library? Its very easy =)
    first you should create object of class Vkth, like this:
            parser = Vkth(your_access_token, your_phone_number)
    after that, it will ask your password for vk.com. Please type it in.
    Awesome! Your new object have two methods:
        parser.wallget
        parser.getawlls
        parser.save
    With them you can get last publications of one wall, or last publications for list of walls, or even save last loaded data.
    
    result = parser.wallget(wall_id, count)
        will return you last "count" publications of "wall_id" public. wall_id must be int, that less then zero!!!
        result will looks like (bool, publications) tuple.
        if all is ok, result[0] is True, else result[0] is False.
        result[1] is what you want. In standart type for vk_api.getwall

    result = parser.getwalls(walls_list, count)
        will return you totally "count" publications, where each wall from walls_list returns you "count//len(wall_list)"
        wall_list must be python list of integers less then zero.
        if all is ok, result[0] is True, else result[0] is False.
        result[1] is what you want. In standart type for vk_api.getwall
    
    parser.save(name)
        will save LAST downloaded data as csv file with "name" to current folder.
        Ohm.... hm.... it will be unused column in this csv... but you can easy delete it =)))

    Examples:
        parser = vk_parser.Vkth(token, login)
        result = parser.wallget(wall_id, count)
            result[0] -> True
            result[1] -> {vk_api}
        result = parser.getwalls(walls_list, count)
            result[0] -> True
            result[1] -> {wall:{vk_api}, wall:{vk_api}.....}
        parser.save('hello.csv')
            will save last loaded data from vk to csv.


    DEPS:
        vk_api(need to be installed)
        pandas(need to be installed)
        getpass(already included to your python)
        random(already included to your python)
    """
    print(page)

class Vkth:
    def __init__(self, token='', login=''):
        self.token = token
        self.login = login
        self.vk_session = vk_api.VkApi(login=self.login, token=self.token, password=getpass.getpass())
        try:
            self.vk_session.auth(token_only=True)
        except vk_api.AuthError as msg:
            print('ERR:',msg)
    def __str__(self):
        return self.login
    
    def wallget(self, wall_id=-1, count=10):
        with vk_api.VkRequestsPool(self.vk_session) as pool:
            posts = pool.method(
            'wall.get', values={
            'owner_id':wall_id, 'count': count, 'filter':'owner'
            }
        )
        if posts.ok and posts.ready:
            self.last_type = 0
            self.last = posts.result
            return (True, posts.result)
        else:
            return (False, {})
    def getwalls(self, wall_list, total_count=100):
        if total_count > 1000:
            print('INFO: set total_count as 1000 because of vk api')
            total_count = 1000
        result = {}
        per_wall = total_count//len(wall_list)+1
        for wall in wall_list:
            if type(wall) != int:
                print("ERR:",wall, "Must be integer")
                continue
            elif wall > 0:
                print("WARN:",wall,"Must be less then zero!\n     Try to get publications for",wall*-1)
                wall = wall*-1
            res = self.wallget(wall_id=wall, count=per_wall)
            if res[0]:
                result[wall] = res[1]
            else:
                result[wall] = {}
                print("ERR:", wall, "Bad responce!")
        self.last_type = 1
        self.last = result
        return (True, result)
    
    def save(self, name=str(random.randint(123,1198379))):
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
        
        frame[['id','from_id','text','likes','reposts','type']].to_csv(name, encoding='utf-8')



