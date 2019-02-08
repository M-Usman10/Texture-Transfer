from utils.texture import *
from utils.tools import *
import cv2
import matplotlib.pyplot as plt
import skimage.transform as transform
print(os.listdir(os.path.abspath('')))
config = load_config(r'configs.yaml')
map_t = Texture(config)
with Cap(path=r'C:\Users\MuhammadUsman\Downloads\4imgs.mp4',step_size=0) as cap:
    images=cap.read_all()
iuv=np.load(r'C:\Users\MuhammadUsman\Downloads\result_IUV.npy')
print(iuv.shape,images[0].shape)
out=map_t.extract_texture_from_video(images,iuv,'Texture_Data/experiment.jpg')
plt.imshow(out)
plt.show()