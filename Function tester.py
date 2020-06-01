import cv2
from Xml_formatter import image_resize

image = cv2.imread('Dataset2/train/image/d97b64fdc4b7d76a.jpg')
image = image_resize(image, width=512)
cv2.imwrite('Dataset2/train/image/d97b64fdc4b7d76a2.jpg', image)
