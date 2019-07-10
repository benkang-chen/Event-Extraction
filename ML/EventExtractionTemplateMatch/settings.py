# 配置文件
# author: luohuagang
# version: 0.0.1
# init: 6/26/2019
# last: 6/26/2019

stanfordnlp_host = 'http://localhost'
stanfordnlp_port = 9000
stanfordnlp_language = 'zh'

test_data_path = './data/test_fire_news'

trigger_fire = ['火灾', '爆炸']
time_fire = ['DATE', 'TIME', 'NUMBER', 'MISC']
location_fire = ['LOCATION', 'FACILITY', 'STATE_OR_PROVINCE']
organization_fire = ['ORGANIZATION']
