"""
# 搭建rest服务
# author: luohuagang
# version: 0.0.1
# date: 6/25/2019
"""
import time
from flask import Flask, jsonify, request
from event_extraction import EventExtraction
from data_to_graph import DataToGraph


SERVICE = Flask(__name__)


@SERVICE.route('/mas/rest/eventextraction/v1',
               methods=['GET', 'POST'])
def eventextraction():
    ''' 事件提取
    '''
    result = {}

    result['code'] = 'OK'
    result['msg'] = '调用成功'
    result['timestamp'] = str(int(time.time()))

    json_data = request.get_json()
    news = json_data['body']['text']
    event = EventExtraction(news)

    result['body'] = {}
    result['body']['graph'] = DataToGraph(event).graph
    result['body']['event_extraction'] = event.event

    return jsonify(result)


def main():
    ''' main 函数
    '''
    SERVICE.config['JSON_AS_ASCII'] = False
    SERVICE.run(
        host='0.0.0.0',
        port=5001,
        debug=True
    )


if __name__ == '__main__':
    main()
