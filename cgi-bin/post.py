# coding:utf-8
import cgi,cgitb
form = cgi.FieldStorage()
import sys
sys.path.append("..")
from cnn_ocr_mnist.infer_cnn_ocr_mnist import predict as premnist
from cnn_ocr_mnist.infer_cnn_ocr_mnist import SetImage
from cnn_ocr_mnist.infer_cnn_ocr import predict as percap
from cnn_ocr_mnist.infer_cnn_ocr import GetCaptcha,GetImage

# Output to stdout, CGIHttpServer will take this as response to the client
print "Content-Type: text/html"     # HTML is following

# Start of content
form = cgi.FieldStorage()

site = form.getvalue('mnist')

line=''
if site == 'mnist':
    line = premnist(SetImage())
if site == 'captcha':
    line = percap(GetImage(GetCaptcha('../Ubuntu-M.ttf'),4))
print '<meta http-equiv="Refresh" content="0;URL=../index.html?'+str(line)+'">'
