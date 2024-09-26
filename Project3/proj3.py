import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
import random
    

def filter_process(img, alpha):
    value = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            value.append(img[i, j])
    value.sort()
    sum = 0
    for i in range(math.floor(alpha/2.), len(value) - math.floor(alpha/2.)):
        sum += value[i]
    sum /= len(value) - 2 * math.floor(alpha/2.)
    return int(sum)

if __name__ == '__main__':

    img_ori = cv2.imread('Kid2 degraded.tiff',0)
    
    kernel_size = 5
    alpha = 16
    
    side = math.floor(kernel_size/2.)
    height, width = img_ori.shape
    padded_img = np.zeros((height+2*side, width+2*side))
    de_noise_img = np.zeros((height, width))
    padded_img[2:height+2, 2:width+2] = img_ori
    for i in range(side, height+side):
        for j in range(side, width+side):
            tmp_img = padded_img[i - side:i + side + 1, j - side:j + side + 1]
            de_noise_img[i-2, j-2] = filter_process(tmp_img, alpha)
    
    padding = cv2.copyMakeBorder(de_noise_img, 0, 800, 0, 800, cv2.BORDER_CONSTANT)
    g = np.fft.fft2(padding)
    G = np.fft.fftshift(g)

    D0 = 250
    n = 3
    M = padding.shape[0]
    N = padding.shape[1]
    GLPF = np.zeros((M,N), dtype=np.float32) # Gaussian LPF
    BLPF = np.zeros((M,N), dtype=np.float32) # Butterworth LPF
    k = 0.00025
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
            # D =(u-M/2)**2 + (v-N/2)**2
            BLPF[u,v] = 1 / (1 + (D/D0)**n)
            GLPF[u,v] = np.exp(-D**2/(2*D0*D0))
            

    # final = G*BLPF/GLPF +  G*(1-BLPF)
    final = G*GLPF/BLPF
    final = np.abs(np.fft.ifft2(np.fft.ifftshift(final)))
    final = final[0:800,0:800]
    
    # show the result and calculate the noise model parameter
    hist_ori = cv2.calcHist([img_ori.astype(np.uint8)], [0], None, [256], [0, 256])/(800**2)
    hist_noise = cv2.calcHist([de_noise_img.astype(np.uint8)], [0], None, [256], [0, 256])/(800**2)
    hist_diff = hist_ori - hist_noise
    
    plt.subplot(311), plt.plot(hist_ori)
    plt.subplot(312), plt.plot(hist_noise)
    plt.subplot(313), plt.plot(hist_diff)
    plt.show()
    
    print("Pa = ", hist_diff[0])
    print("Pb = ", hist_diff[255])

    # plot the image
    plt.subplot(221), plt.imshow(img_ori, cmap='gray')
    plt.subplot(222), plt.imshow(de_noise_img, cmap='gray')
    plt.subplot(223),plt.imshow(final, cmap='gray')
    plt.show()
    
    # save the image
    de_noise_save = Image.fromarray(de_noise_img.astype(np.uint8))
    de_noise_save.save("img/de_noise.png",dpi = (200,200))
    final_save = Image.fromarray(final.astype(np.uint8))
    final_save.save("img/D0_250.png",dpi = (200,200))
    