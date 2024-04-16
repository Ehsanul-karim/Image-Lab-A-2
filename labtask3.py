import numpy as np
import cv2
import matplotlib.pyplot as plt

sendondImage = "E:\\4th year, 1st semester\\Image Lab\\color_img.jpg"
InputImage = "E:\\4th year, 1st semester\\Image Lab\\col.jpg"
# InputImage = "my_picture.jpg"



# def normalize(image):
#     cv2.normalize(image,image, 0, 255, cv2.NORM_MINMAX)
#     image = np.round(image).astype(np.uint8)
#     return image

def histogram(image,CDF_Pass=True, PDF_Pass=True):
    width, height = image.shape
    array = np.zeros((256))
    intesity = np.zeros((256))
    cumilative =  np.zeros((256))


    for i in range(width):
        for j in range(height):
            value = image[i][j]
            if(array[value] == 0):
                array[value] = 1
                intesity[value] = intesity[value]+1
            else:
                intesity[value] = intesity[value]+1
    for i in range(256):
        intesity[i] = intesity[i] / (width * height)

    if PDF_Pass == False:
        return intesity
    
    cumilative[0] = intesity[0]
    for i in range(1,256):
        cumilative[i] = intesity[i] + cumilative[i-1]

    if CDF_Pass == False:
        return cumilative
    
    for i in range(256):
        cumilative[i] = cumilative[i] * 255

    cumilative = np.round(cumilative).astype(np.uint8)


    for i in range(width):
        for j in range(height):
            value = image[i][j]
            ans = cumilative[value]
            image[i][j] = ans

    return image

def Main():
        color = ('b','g','r')
        img = cv2.imread(InputImage)
        cv2.imshow("input", img)  
        cv2.waitKey(0)           
        b1, g1, r1 = cv2.split(img)
        bo1 = histogram(b1)
        go1 = histogram(g1)
        ro1 = histogram(r1)
        merged = cv2.merge((bo1, go1, ro1))
        # out = normalize(merged)
        out = merged
        print (out)
        cv2.imshow("output", out)
        cv2.waitKey(0)  

        plt.figure(1,figsize=(20, 10))
        for i,col in enumerate(color):   
            plt.subplot(2, 3, i+1)
            histr, _ = np.histogram(img[:,:,i],256,[0,256])
            plt.plot(histr,color = col)  #Add histogram to our plot 
            plt.title('Channel'+str(i+1))
        plt.show()

        plt.figure(2,figsize=(20, 10))
        for i,col in enumerate(color):   
            plt.subplot(2, 3, i+1)
            histr, _ = np.histogram(out[:,:,i],256,[0,256])
            plt.plot(histr,color = col)
            plt.title('Channel'+str(i+1))
        plt.show()

# ---------------------------------------------------------
        img = cv2.imread(InputImage)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cv2.imshow("input in HSV formet", img)  
        cv2.waitKey(0)  
        h1, s1, v1 = cv2.split(img)
        vo1 = histogram(v1)
        merged = cv2.merge((h1, s1, vo1))
        # out = normalize(merged)
        out = merged
        print (out)
        cv2.imshow("Merged output HSV picture", out)
        cv2.waitKey(0)

        plt.figure(3,figsize=(20, 10))
        for i,col in enumerate(color):   
            plt.subplot(2, 3, i+1)
            histr, _ = np.histogram(img[:,:,i],256,[0,256])
            plt.plot(histr,color = col)  #Add histogram to our plot 
            plt.title('Channel'+str(i+1))
        plt.show()

        plt.figure(4,figsize=(20, 10))
        for i,col in enumerate(color):   
            plt.subplot(2, 3, i+1)
            histr, _ = np.histogram(out[:,:,i],256,[0,256])
            plt.plot(histr,color = col)
            plt.title('Channel'+str(i+1))
        plt.show()

        BacktoRGB = cv2.cvtColor(out, cv2.COLOR_HSV2BGR)
        # out = normalize(BacktoRGB)
        out = BacktoRGB
        cv2.imshow("HSV brought back to RGB,(HSV space)", out)
        cv2.waitKey(0)
        # cv2.imwrite("my_output_image.jpg", out) 


    

