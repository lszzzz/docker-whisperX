import pytest
import http.client
import json
import requests


def test_get():
    connection = http.client.HTTPConnection('localhost', 7010, timeout=10)
    connection.request("GET", "/")
    response = connection.getresponse()

    print("Status:{}, reason:{}, body:{}".format(response.status, response.reason, response.read().decode()))
    connection.close()


def test_post():
    connection = http.client.HTTPConnection('localhost', 7010, timeout=10)
    headers = {'Content-type': 'application/json'}

    request_params = {
        'model': 'small',
        'compute_type': 'int8',
    }
    json_data = json.dumps(request_params)

    connection.request('POST', '/asr', json_data, headers)
    response = connection.getresponse()

    print("Status:{}, reason:{}, body:{}".format(response.status, response.reason, response.read().decode()))
    connection.close()


def test_post_file_en():
    url = 'http://localhost:7010/asr'
    files = {'file': open('/Users/samlee/Documents/sample/asr/en/cv-corpus-18.0-delta-2024-06-14/en/clips/common_voice_en_40187648.mp3', 'rb')}
    values = {
        'model': 'small',
        'compute_type': 'int8',
        'device': 'cpu'
    }
    response = requests.post(url, files=files, data=values)
    print(response.json())


def test_post_file_zh():
    url = 'http://localhost:7010/asr'
    files = {'file': open('/Users/samlee/Documents/sample/asr/cn/vad_example.wav', 'rb')}
    values = {
        'model': 'large-v3',
        'compute_type': 'int8',
        'language': 'zh',
        'initial_prompt': '该转录涉学卡的活动机会，通过报名畅学卡后面可以参加专门的活动降低试错成本',
        'device': 'cpu'
    }
    response = requests.post(url, files=files, data=values)
    print(response.json())


def test_post_file_yue():
    url = 'http://localhost:7010/asr'
    files = {'file': open('/Users/samlee/Documents/sample/asr/yue/SAMPLE01_S001.wav', 'rb')}
    values = {
        'model': 'large-v3',
        'compute_type': 'int8',
        'language': 'yue',
        'align_model': 'CAiRE/wav2vec2-large-xlsr-53-cantonese',
        'device': 'cpu'
    }
    response = requests.post(url, files=files, data=values)
    print(response.json())