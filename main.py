from utils.map_texture import *
from utils.tools import load_config
import cv2
import matplotlib.pyplot as plt
if __name__ == '__main__':
    print('Running')
    config = load_config(r'configs.yaml')
    map_t = MapTexture(config)
    im = cv2.imread('demo_im.jpg')
    iuv = cv2.imread('demo_im_IUV.png')
    im=np.zeros(iuv.shape)
    out=map_t.transfer_texture(im,iuv)
    plt.imshow(out)
    plt.show()