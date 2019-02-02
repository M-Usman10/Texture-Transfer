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


def process_video(saved_path,video_name):
    os.system(
        "sudo nvidia-docker run --rm -v {}:/denseposedata -t densepose:c2-cuda9-cudnn7-wdata python2 tools/infer"
        "_simple.py --cfg configs/DensePose_ResNet101_FPN_s1x-e2e.yaml --output-dir DensePoseData/output_results/"
        " --image-ext jpg --wts DensePoseData/weights/weights.pkl DensePoseData/input_imgs/{}".format(
            config['inference_dir'], video_name))
    im = cv2.imread(saved_path)
    iuv = cv2.imread(os.path.join(config['output_dir'], video_name[:-4] + '_IUV.png'))
    out = map_t.transfer_texture(im, iuv)
    result_save_file = os.path.join(app.config['UPLOAD_FOLDER'], "texture_result.jpg")
    io.imsave(result_save_file, out[..., ::-1])


@app.route('/', methods = ['POST'])
def api_root():
    if request.method == 'POST' and request.files['video']:
        app.logger.info(app.config['UPLOAD_FOLDER'])
        video = request.files['video']
        video_name = secure_filename(video.filename)
        create_new_folder(app.config['UPLOAD_FOLDER'])
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], video_name)
        app.logger.info("saving {}".format(saved_path))
        video.save(saved_path)
        process_video(saved_path,video_name)
        return send_from_directory(config['send_from'],'texture_result.jpg', as_attachment=True)
    else:
        return "Where is the image?"

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=9090, debug=False)
