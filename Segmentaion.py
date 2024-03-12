import numpy as np
import cv2

LinaImage = "E:\\4th year, 1st semester\\Image Lab\\lena.jpg"


def normalize(image):
    cv2.normalize(image,image, 0, 255, cv2.NORM_MINMAX)
    image = np.round(image).astype(np.uint8)
    return image


def recursive_split(image, threshold, min_size=2):
    height, width = image.shape

    if height <= min_size and width <= min_size:
        mean = np.mean(image)
        for l in range(image.shape[0]):
            for j in range(image.shape[1]):
                image[l][j] = mean
        return
    
    if(np.std(image) > threshold):
        mid_h = height // 2
        mid_w = width // 2
        quadrants = [
            image[:mid_h, :mid_w],  
            image[:mid_h, mid_w:], 
            image[mid_h:, :mid_w],  
            image[mid_h:, mid_w:]   
        ]
    
        for quadrant in quadrants:
            recursive_split(quadrant, threshold, min_size)
    else:
        mean = np.mean(image)
        for l in range(image.shape[0]):
            for j in range(image.shape[1]):
                image[l][j] = mean

def Main():
    while True:
        print("Enter the value of Threshold: ")
        threshold = float(input())
        img = cv2.imread(LinaImage, cv2.IMREAD_COLOR)
        cv2.imshow("input", img)  
        cv2.waitKey(0)
        print(img.shape)
        b1, g1, r1 = cv2.split(img)
        recursive_split(b1, threshold)
        recursive_split(g1, threshold)
        recursive_split(r1, threshold)
        merged = cv2.merge((b1, g1, r1))
        out = normalize(merged)
        print (out)
        cv2.imshow("output", out)
        cv2.waitKey(0)       
        cv2.destroyAllWindows()
Main()