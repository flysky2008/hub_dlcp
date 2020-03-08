# Version python3.6
# -*- coding: utf-8 -*-
# @Time    : 2019/12/17 10:41
# @Author  : HENRY
# @Email   : mogaoding@163.com
# @File    : doc-extractor
# @Project : dlcp_hub
# @Software: PyCharm

import json
from wsgiref import simple_server
import falcon
from falcon import RequestOptions
import sys


class ThingsResource(object):

    def __init__(self):
        pass
        # self.ner_service = NerService()

    def on_get(self, req, resp):
        resp.body = '{"message": "GET is not support!"}'

    def on_post(self, req, resp):
        resp.body = '{"message": "POST METHOD"}'
        self._handle(req, resp)

    def _handle(self, req, resp):
        try:
            content_str = req.get_param('content') or ''  # 该图片字节数组
            print('content:{}'.format(content_str))
            #  result = self.ner_service.parse_company_name(content_str, ckpt_file, config, model)
        except Exception as ex:
            print(ex)
            description = 'Some Error happened, please check your request.'
            raise falcon.HTTPServiceUnavailable(
                'Service Outage',
                description,
                30)
        resp.status = falcon.HTTP_200
        resp.body = result


app = falcon.API()
ro = RequestOptions()
ro.auto_parse_form_urlencoded = True
app.req_options = ro

app.add_route('/nlp/test', ThingsResource())
print("start sucessed ")

if __name__ == '__main__':
    httpd = simple_server.make_server('192.168.1.10', 8901, app)
    print("start ....")
    httpd.serve_forever()
    # ckpt_file, config, model = ParameterServer.ParameterServer().get_init_parameter()
    # print('ckpt_file:{}, config:{}, model:{}'.format(ckpt_file, config, model))
    print("running...... ")

