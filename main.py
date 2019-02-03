import os
import sys
import glob
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


def process_video(saved_path,video_name,flag=0):
    os.system(
        "sudo nvidia-docker run --rm -v {}:/denseposedata -v {}:/denseposetools "
        "-t densepose:c2-cuda9-cudnn7-wdata-movie python2 tools/infer"
        "_video.py --cfg configs/DensePose_ResNet101_FPN_s1x-e2e.yaml --output-dir DensePoseData/output_results/"
        " --image-ext jpg --wts DensePoseData/weights/weights.pkl DensePoseData/input_imgs/{}".format(
            config['inference_dir'],config['tool_dir'],video_name))
    video_base_name=os.path.basename(video_name).split(".")[0]
    IUV_save_path=os.path.join(config['output_dir'],video_base_name+'.avi')
    cap = Cap(saved_path, step_size=1)
    with cap as cap:
        images = cap.read_all()
    cap = Cap(IUV_save_path, step_size=1)
    with cap as cap:
        iuvs=cap.read_all()
    # iuvs=read_images_sorted(IUV_save_path,key=iuv_files_sort)
    print ("IUVS found {}".format(len(iuvs)))
    if flag==0:
        result_save_file = os.path.join(app.config['UPLOAD_FOLDER'], "texture_result.mp4")
        out=map_t.transfer_texture_on_video(images,iuvs)
        print ("saving video at {}".format(result_save_file))
        save_video(out,result_save_file)

    else:
        result_save_file = os.path.join(app.config['UPLOAD_FOLDER'], "texture_result.jpg")
        map_t.extract_texture_from_video(images,iuvs,result_save_file)

@app.route('/', methods = ['POST'])
def api_root():
    if request.method == 'POST' and request.files['extract']:
        app.logger.info(app.config['UPLOAD_FOLDER'])
        video = request.files['extract']
        video_name = secure_filename(video.filename)
        create_new_folder(app.config['UPLOAD_FOLDER'])
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], video_name)
        app.logger.info("saving {}".format(saved_path))
        video.save(saved_path)
        process_video(saved_path,video_name,flag=1)
        return send_from_directory(app.config['UPLOAD_FOLDER'],'texture_result.jpg', as_attachment=True)
    elif request.method == 'POST' and request.files['transfer']:
        app.logger.info(app.config['UPLOAD_FOLDER'])
        video = request.files['transfer']
        video_name = secure_filename(video.filename)
        create_new_folder(app.config['UPLOAD_FOLDER'])
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], video_name)
        app.logger.info("saving {}".format(saved_path))
        video.save(saved_path)
        process_video(saved_path, video_name)
        return send_from_directory(app.config['UPLOAD_FOLDER'], 'texture_result.mp4', as_attachment=True)
    else:
        return "Where is the image?"

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=9090, debug=False)
