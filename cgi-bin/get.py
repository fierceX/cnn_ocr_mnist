#!/usr/bin/env python
# coding:utf-8
import cgi,cgitb
form = cgi.FieldStorage()
import sys
sys.path.append("..")
from cnn_ocr_mnist.infer_mnist import GetMnistPredict
from cnn_ocr_mnist.infer_captcha import GetCaptchaPredict
from cnn_ocr_mnist.infer_plate import GetPlatePredict


print "Content-Type: text/html"

print 
form = cgi.FieldStorage()

site = form.getvalue('mnist')

line=''

if site == 'mnist':
    line = GetMnistPredict()
if site == 'captcha':
    line = GetCaptchaPredict()
if site == 'plate':
    line = GetPlatePredict()
print '<meta charset="utf-8" http-equiv="Refresh" content="0;URL=../index.html?'+str(line)+'">'
