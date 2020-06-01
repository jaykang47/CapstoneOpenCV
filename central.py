import cv2
import numpy as np
import urllib.request
import detect_blinks
import detect_mouth
import SMS_Sender

cam1 = 'http://192.168.86.87/picture/1/current/'
cam2 = 'http://192.168.86.87/picture/2/current/'
cam3 = 'http://192.168.86.87/picture/3/current/'

def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

currentIMG = url_to_image(cam1)
if detect_mouth.main(currentIMG):
    #SMS_Sender.main('1234567890', 'Your mouth is open')
    print("test mouth")
if not detect_blinks.main(currentIMG):
    #SMS_Sender.main('1234567890', 'Your eyes are closed')
    print("test eye")

