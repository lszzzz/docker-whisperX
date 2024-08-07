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


def test_post_file():
    url = 'http://localhost:7010/asr'
    files = {'file': open('/Users/samlee/Documents/sample/asr/en/cv-corpus-18.0-delta-2024-06-14/en/clips/common_voice_en_40187648.mp3', 'rb')}
    values = {
        'model': 'small',
        'compute_type': 'int8',
    }
    response = requests.post(url, files=files, data=values)
    print(response.json())
