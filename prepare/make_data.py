import sir_liba  # uor library depends on vk_api
#import vk_api
import pandas as ps
import time

# PARSE VK

parser = sir_liba.VkParse('89097237917', '68229a1068229a1068229a10a66847b08a6682268229a1033675e956fcebd9147ed8039')
parser.cleangrouptable()
for cat in ['Игры и киберспорт', 'Наука и технологии', 'Новости', 'Развлечения', 'Культура и искусство', 'Спорт']:
    parser.makegrouptable(cat, alpha=0.4)

i = 0
while i <= 200:
    print("offset ", i)
    try:
        parser.scrapposts(filename='all_NEW', offset=i, total_count=1000)
        i += 100
    except:
        time.sleep(15 * 60)
        continue

# CLEAR DATAFRAME

# eda = sir_liba.EdaMaker('all_posts.csv', text_series='text')
# eda.cleantext()
# eda.save(arg_name="edaed_posts.csv")

# Read sir_liba.help() for more info...
