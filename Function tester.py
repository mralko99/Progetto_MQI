import cv2
from function import image_resize

# Codice per testare le funzioni di Xml_formatter
image = cv2.imread('000360253ca71b80.jpg')
image = image_resize(image, 1024)
cv2.imwrite('000360253ca71b802.jpg', image)
