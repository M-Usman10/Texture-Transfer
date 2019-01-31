import numpy
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
import cv2
class MapTexture:
    def __init__(self,config):
        self.mode='read_from_file'
        self.texture_path=config['texture_img']
        self.Grid_Pixels=config['grid_pixels']
        self.read_texture()

    def read_texture(self):
        texture_img=cv2.imread(self.texture_path)[:,:,::-1]/255.
        self.TextureIm = np.zeros([24, 200, 200, 3])
        for i in range(4):
            for j in range(6):
                self.TextureIm[(6 * i + j), :, :, :] = \
                    texture_img[(self.Grid_Pixels * j):(self.Grid_Pixels * j + self.Grid_Pixels),
                    (self.Grid_Pixels * i):(self.Grid_Pixels * i + self.Grid_Pixels), :]

    def transfer_texture(self, im, IUV):
        TextureIm = self.TextureIm
        generated_image = im.copy()
        print(generated_image.max())
        for PartInd in range(1, 25):  ## Set to xrange(1,23) to ignore the face part.
            tex = TextureIm[PartInd - 1, :, :, :].squeeze()  # get texture for each part.
            u_current_points = IUV[..., 1][IUV[:, :, 0] == PartInd]  # Pixels that belong to this specific part.
            v_current_points = IUV[..., 2][IUV[:, :, 0] == PartInd]
            mask = ((255 - v_current_points) * 199. / 255.).astype(int), (
                    u_current_points * 199. / 255.).astype(int)
            tex_to_rep = tex[mask][..., ::-1]
            print(tex_to_rep.max())
            generated_image[IUV[:, :, 0] == PartInd] = (tex_to_rep * 255).astype(np.uint8)
        return generated_image
