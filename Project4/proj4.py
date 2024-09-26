import cv2
import numpy as np
from PIL import Image

img = cv2.imread("LovePeace rose.tif")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# get RGB and HSI components
B = img[:,:,0]
G = img[:,:,1]
R = img[:,:,2]
H = hsv[:,:,0]
S = hsv[:,:,1]
I = hsv[:,:,2]

# show all the components
cv2.imshow("B", B)
cv2.imshow("G", G)
cv2.imshow("R", R)
cv2.imshow("H", H)
cv2.imshow("S", S)
cv2.imshow("I", I)

# RGB sharpening and HSI sharpening
kernel = np.array([[-1, -1, -1],
                   [-1, 9,-1],
                   [-1, -1, -1]])
rgb_sharp = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
hsv_sharp = cv2.filter2D(src=hsv, ddepth=-1, kernel=kernel)
after_hsv = cv2.cvtColor(hsv_sharp, cv2.COLOR_HSV2BGR)
diff = cv2.subtract(rgb_sharp, after_hsv)

cv2.imshow("img", img)
cv2.imshow("after rgb sharpening", rgb_sharp)
cv2.imshow("after hsv sharpening", after_hsv)
cv2.imshow("difference", diff)
cv2.imshow("hsv", hsv)
cv2.waitKey()

R_save = Image.fromarray(R.astype(np.uint8))
R_save.save("img/R.png",dpi = (200,200))
G_save = Image.fromarray(G.astype(np.uint8))
G_save.save("img/G.png",dpi = (200,200))
B_save = Image.fromarray(B.astype(np.uint8))
B_save.save("img/B.png",dpi = (200,200))
H_save = Image.fromarray(H.astype(np.uint8))
H_save.save("img/H.png",dpi = (200,200))
S_save = Image.fromarray(S.astype(np.uint8))
S_save.save("img/S.png",dpi = (200,200))
I_save = Image.fromarray(I.astype(np.uint8))
I_save.save("img/I.png",dpi = (200,200))

rgb_sharp = cv2.cvtColor(rgb_sharp, cv2.COLOR_BGR2RGB)
rgb_sharpen_save = Image.fromarray(rgb_sharp.astype(np.uint8))
rgb_sharpen_save.save("img/rgb_sharpen.png",dpi = (200,200))

after_hsv = cv2.cvtColor(after_hsv, cv2.COLOR_BGR2RGB)
hsi_sharpen_save = Image.fromarray(after_hsv.astype(np.uint8))
hsi_sharpen_save.save("img/hsi_sharpen.png",dpi = (200,200))

diff = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)
diff_save = Image.fromarray(diff.astype(np.uint8))
diff_save.save("img/diff1.png",dpi = (200,200))