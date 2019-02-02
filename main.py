import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from utils.texture import *
from utils.tools import *
from flask import url_for, send_from_directory, request
from werkzeug import secure_filename
import skimage.io as io
import cv2
config = load_config(r'configs.yaml')
map_t = Texture(config)
app = make_flask_app(config)


def process_video(saved_path,video_name):
    os.system(
        "sudo nvidia-docker run --rm -v {}:/denseposedata -t densepose:c2-cuda9-cudnn7-wdata python2 tools/infer"
        "_simple.py --cfg configs/DensePose_ResNet101_FPN_s1x-e2e.yaml --output-dir DensePoseData/output_results/"
        " --image-ext jpg --wts DensePoseData/weights/weights.pkl DensePoseData/input_imgs/{}".format(
            config['inference_dir'], video_name))
    cap=Cap(saved_path,step_size=1)
    with cap as cap:
        images = cap.read_all()
    cap = Cap(os.path.join(config['output_dir'], video_name[:-4] + '_IUV.mp4'), step_size=1)
    with cap as cap:
        iuvs = cap.read_all()
    out=[]
    for i in range(len(images)):
        out.append(map_t.transfer_texture(images[i], iuvs[i]))
    result_save_file = os.path.join(app.config['UPLOAD_FOLDER'], "texture_result.mp4")
    save_video(out,result_save_file)


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
        return send_from_directory(config['send_from'],'texture_result.mp4', as_attachment=True)
    else:
        return "Where is the image?"

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=9090, debug=False)
