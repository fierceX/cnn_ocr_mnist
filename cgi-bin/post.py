# coding:utf-8
import cgi
form = cgi.FieldStorage()
import sys
sys.path.append("..")
from cnn_ocr_mnist.infer_cnn_ocr_mnist import predict, SetImage

# Output to stdout, CGIHttpServer will take this as response to the client
print "Content-Type: text/html"     # HTML is following
print                               # blank line, end of headers
# Start of content
print "<body bgcolor=\"#F5F5DC\">"
print "<p>Handwritten numeral recognition based on convolutional neural network </p>"
print '''
<IMG src="../img.jpg"/>
<form name="input" action="post.py" method="post">
<input type="submit" value="Next">
'''
print "<p>" + repr(predict(SetImage())) + "</p>"

for i in range(32):
    print "<IMG src=\"../p4"+str(i)+".jpg\"/>"
print "</body>"
