import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from utils.map_texture import *
from utils.tools import load_config
from utils.tools import make_flask_app
from utils.tools import create_new_folder
from flask import url_for, send_from_directory, request
from werkzeug import secure_filename
import skimage.io as io
import cv2


config = load_config(r'configs.yaml')
map_t = MapTexture(config)
app = make_flask_app(config)
@app.route('/', methods = ['POST'])
def api_root():
    print("heloo")
    if request.method == 'POST':
        app.logger.info(app.config['UPLOAD_FOLDER'])
        img = request.files['image']
        img_name = secure_filename(img.filename)
        create_new_folder(app.config['UPLOAD_FOLDER'])
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
        app.logger.info("saving {}".format(saved_path))
        img.save(saved_path)
        im=cv2.imread(saved_path)
        os.system(
            "sudo nvidia-docker run --rm -v {}:/denseposedata -t densepose:c2-cuda9-cudnn7-wdata python2 tools/infer"
            "_simple.py --cfg configs/DensePose_ResNet101_FPN_s1x-e2e.yaml --output-dir DensePoseData/output_results/"
            " --image-ext jpg --wts DensePoseData/weights/weights.pkl DensePoseData/input_imgs/{}".format(
                config['inference_dir'], img_name))
        iuv = cv2.imread(os.path.join(config['output_dir'],img_name[:-4]+'_IUV.png'))
        out = map_t.transfer_texture(im,iuv)
        io.imsave('Output_Data/1.jpg',out[...,::-1])
        return send_from_directory(app.config['UPLOAD_FOLDER'],img_name, as_attachment=True)
    else:
        return "Where is the image?"

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=9090, debug=False)