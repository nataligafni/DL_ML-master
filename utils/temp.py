import cv2
import matplotlib.pyplot as plt
import os
Path=r'C:\Users\MorBen-Nun\Documents\University\Deep_learning_medical\Data\classification\CT_COVID_MASKS'
I=cv2.imread(Path+'\\'+'._2019-novel-Coronavirus-severe-adult-respiratory-dist_2020_International-Jour-p3-89%0.png')
plt.imshow(I)

import matplotlib.image as mpimg
# for img in os.listdir("/content/train"):
full_path=os.path.join(Path,'2019-novel-Coronavirus-severe-adult-respiratory-dist_2020_International-Jour-p3-89.png')
image = mpimg.imread(full_path)
plt.imshow(image)
plt.show()