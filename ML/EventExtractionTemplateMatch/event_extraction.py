# usage: import
# author: luohuagang
# version: 0.0.1
# init: 6/25/2019
# last: 6/26/2019

import re
from ML.EventExtractionTemplateMatch import settings
from stanfordcorenlp import StanfordCoreNLP


class EventExtraction:
    def __init__(self, context):        
        # 初始化事件字典，包含触发词，事件类型
        # 时间，地点，救援组织，事故原因，事故损失
        self.event = {}
        # 使用stanfordnlp工具分析事件
        nlp = StanfordCoreNLP(path_or_host=settings.stanfordnlp_host, lang=settings.stanfordnlp_language, port=settings.stanfordnlp_port)
        nlp_result = nlp.ner(context)
        self.having_event(nlp_result)

        if self.event['events'] == '火灾':
            # 提取时间、地点、救援组织
            self.event['time'] = ",".join(self.taking_time(nlp_result))
            self.event['location'] = self.taking_location(nlp_result)
            self.event['organization'] = self.taking_organization(nlp_result)

            # 定义事故原因模板
            self.cause_patterns = self.pattern_cause()
            # 定义事故损失模板
            self.lose_patterns = self.pattern_lose()
            # 匹配事故原因和事故损失
            self.pattern_match(context)
            self.event['cause'] = "".join(self.cause)
            self.event['lose'] = self.lose
        else:
            self.event['trigger'] = None
            self.event['events'] = None
        nlp.close()

    def having_event(self, nlp_result):
        for item in nlp_result:
            if item[1] == 'CAUSE_OF_DEATH':
                if item[0] in settings.trigger_fire:
                    self.event['trigger'] = item[0]
                    self.event['events'] = '火灾'
    
    def taking_time(self, nlp_result):
        i = 0
        state = False
        only_date = False
        time_fire = ""
        result = []
        while i < len(nlp_result):
            if nlp_result[i][1] == 'DATE':
                time_fire += nlp_result[i][0]
                if state == False:
                    state = True
                    only_date = True
            else:
                if state == True:
                    if (nlp_result[i][1] == 'TIME'
                        or nlp_result[i][1] == 'NUMBER'
                        or nlp_result[i][1] == 'MISC'):
                        time_fire += nlp_result[i][0]
                        only_date = False
                    else:
                        if only_date == False:
                            result.append(time_fire)
                        time_fire = ""
                        state = False
            i += 1
        if state == True:
            result.append(time_fire)
        
        result = list(set(result))

        return result

    def taking_location(self, nlp_result):
        i = 0
        state = False
        location = ""
        result = []
        while i < len(nlp_result):
            if (nlp_result[i][1] == 'LOCATION' or
                    nlp_result[i][1] == 'FACILITY' or
                    nlp_result[i][1] == 'CITY'):
                location += nlp_result[i][0]
                if state == False:
                    state = True
            else:
                if state == True:
                    result.append(location)
                    location = ""
                    state = False
            i += 1
        if state == True:
            result.append(location)
        
        result = list(set(result))

        return result

    def taking_organization(self, nlp_result):
        i = 0
        state = False
        organization = ""
        result = []
        while i < len(nlp_result):
            if nlp_result[i][1] == 'ORGANIZATION':
                organization += nlp_result[i][0]
                if state == False:
                    state = True
            else:
                if state == True:
                    result.append(organization)
                    organization = ""
                    state = False
            i += 1
        if state == True:
            result.append(organization)
        
        result = list(set(result))

        return result

    def pattern_cause(self):
        patterns = []
        
        key_words = ['起火', '事故', '火灾']
        pattern = re.compile('.*?(?:{0})原因(.*?)[,.?:;!，。？：；！]'.format('|'.join(key_words)))
        patterns.append(pattern)

        return patterns


    def pattern_lose(self):
        patterns = []
        
        key_words = ['伤亡','损失']
        pattern = re.compile('.*?(未造成.*?(?:{0}))[,.?:;!，。？：；]'.format('|'.join(key_words)))
        patterns.append(pattern)

        patterns.append(re.compile('(\d+人死亡)'))
        patterns.append(re.compile('(\d+人身亡)'))
        patterns.append(re.compile('(\d+人受伤)'))
        patterns.append(re.compile('(\d+人烧伤)'))
        patterns.append(re.compile('(\d+人坠楼身亡)'))
        patterns.append(re.compile('(\d+人遇难)'))
        
        return patterns
        

    def pattern_match(self, news):
        self.cause = []
        self.lose = []
        
        for pattern in self.cause_patterns:
            match_list = re.findall(pattern, news)
            if match_list:
                self.cause.append(match_list[0])
        
        for pattern in self.lose_patterns:
            match_list = re.findall(pattern, news)
            if match_list:
                self.lose.append(match_list[0])


if __name__=='__main__':
    # 读入数据    
    with open(settings.test_data_path, 'r') as f:
        test_data = f.readlines()
    
    news = test_data[1]
    print(news)
    events = EventExtraction(news)
    print(events.event)
    '''
    with open('test_result', 'a') as f:
        f.write(news)
        f.write(str(events.event))
        f.write('\n')
'''
