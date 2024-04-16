import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
from labtask3 import histogram

InputImage = "landscape.png"


erlang_pdf = np.empty([256])
erlang_cdf = np.empty([256])

def hist_matching(img, cdf1, cdf2):
    Marray = np.zeros((256))
    for idx in range(0,256):
        minimum = 1e6
        ind = -25
        for idx2 in range(0,256):
            if minimum > abs(cdf1[idx]-cdf2[idx2]):
                minimum = abs(cdf1[idx]-cdf2[idx2])
                ind = idx2
        Marray[idx] = ind
    width, height = img.shape
    new_output_image = np.zeros((width,height))

    Marray = np.round(Marray).astype(np.uint8)

    print("================================")

    for i in range(width):
        for j in range(height):
            new_output_image[i][j] = Marray[img[i][j]]

    return new_output_image

def find_value(x,k,mu):
    return (math.pow(x,k-1) * math.exp((-x)/mu))/(math.pow(mu,k) * math.factorial(k-1))    

print("Enter the value of Shape Parameter. (k)")
shape = int(input())
print("Enter the value of Scale Parameter. (mu)")
scale = float(input())

for i in range(0,256):
    erlang_pdf[i] = find_value(i,shape, scale)

erlang_cdf[0] = erlang_pdf[0]

for i in range(1,256):
    erlang_cdf[i] =  erlang_pdf[i] + erlang_cdf[i-1]

img = cv2.imread(InputImage,cv2.IMREAD_GRAYSCALE)
cv2.imshow("Input Gray Scale image", img)
cv2.waitKey(0)

image_cdf = histogram(img,CDF_Pass=False)
image_pdf = histogram(img,PDF_Pass=False)
image_histogram, _ = np.histogram(img,256,[0,256]) 


plt.figure(1,figsize=(15, 8))
plt.subplot(2, 3, 1)
plt.plot(erlang_pdf,color = 'b')
plt.title('Erlang PDF')
plt.subplot(2, 3, 2)
plt.plot(erlang_cdf,color = 'g')
plt.title('Erlang CDF')
plt.show()

new_output_image = hist_matching(img, image_cdf, erlang_cdf)
new_output_image = np.round(new_output_image).astype(np.uint8)


cv2.imshow("Output Gray image", new_output_image)
cv2.waitKey(0)

output_image_cdf = histogram(new_output_image,CDF_Pass=False)
output_image_pdf = histogram(new_output_image,PDF_Pass=False)
output_image_histogram, _ = np.histogram(new_output_image,256,[0,256]) 



plt.figure(2,figsize=(15, 8))
plt.subplot(2, 3, 1)
plt.plot(image_pdf,color = 'b')
plt.title('Input Image PDF')
plt.subplot(2, 3, 2)
plt.plot(image_cdf,color = 'g')
plt.title('Input Image CDF')
plt.subplot(2, 3, 3)
plt.plot(image_histogram,color = 'r')
plt.title('Input Image Histogram')
plt.subplot(2, 3, 4)
plt.plot(output_image_pdf,color = 'b')
plt.title('Output Image PDF')
plt.subplot(2, 3, 5)
plt.plot(output_image_cdf,color = 'g')
plt.title('Output Image CDF')
plt.subplot(2, 3, 6)
plt.plot(output_image_histogram,color = 'r')
plt.title('Output Image Histogram')
plt.show()


