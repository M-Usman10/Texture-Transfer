import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from utils.texture import *
from utils.tools import *
from flask import send_from_directory, request
from werkzeug import secure_filename

config = load_config(r'configs.yaml')
map_t = Texture(config)
app = make_flask_app(config)


def process_video(saved_path, video_name, result_filename='', flag=0):
    os.system(
        "sudo nvidia-docker run --rm -v {}:/denseposedata -v {}:/denseposetools "
        "-t densepose:c2-cuda9-cudnn7-wdata-movie python2 tools/infer"
        "_video.py --cfg configs/DensePose_ResNet101_FPN_s1x-e2e.yaml --output-dir DensePoseData/output_results/"
        " --step-size 0 --image-ext jpg --wts DensePoseData/weights/weights.pkl DensePoseData/input_imgs/{}".format(
            config['inference_dir'],config['tool_dir'],video_name))
    video_base_name=os.path.basename(video_name).split(".")[0]
    IUV_save_path=os.path.join(config['output_dir'],video_base_name)
    IUV_save_path=os.path.join(IUV_save_path,"result_IUV.npy")

    with Cap(saved_path,step_size=0) as cap:
        images = cap.read_all()

    # with Cap(IUV_save_path, step_size=1) as cap:
    #     iuvs=cap.read_all()
    # iuvs=read_images_sorted(IUV_save_path,key=iuv_files_sort)
    iuvs=np.load(IUV_save_path)

    assert len(iuvs)==len(images),"Number of frames of IUV video and sent video not equal"

    if flag==0:
        result_filename="texture_result.mp4" if len(images)>1 else "texture_result.jpg"
        result_save_file = os.path.join(app.config['UPLOAD_FOLDER'],result_filename)
        out=map_t.transfer_texture_on_video(images,iuvs)
        if len(images)>1:
            print ("saving video at {}".format(result_save_file))
            save_video(out,result_save_file)
        else:
            cv2.imwrite(result_save_file,out[0])
        return result_filename

    else:
        result_filename = result_filename + '.jpg'
        result_save_file = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        map_t.extract_texture_from_video(images,iuvs,result_save_file)
        return result_filename


@app.route('/retreive_texture', methods = ['POST'])
def retreive_texture():
    print ("request for texture retreival received")
    app.logger.info(app.config['UPLOAD_FOLDER'])
    create_new_folder(app.config['UPLOAD_FOLDER'])
    img1 = request.files['img1']
    img2 = request.files['img2']
    img3 = request.files['img3']
    img4 = request.files['img4']
    video = [img1, img2, img3, img4]
    name = request.form['texture_filename']
    names = []
    for i in video:
        names.append(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(i.filename)))
        app.logger.info("saving {}".format(names[-1]))
        i.save(names[-1])
    images_to_video(names, app.config['UPLOAD_FOLDER'] + '/video.mp4')
    print(name, type(name))
    filename = process_video(app.config['UPLOAD_FOLDER'] + '/video.mp4', 'video.mp4', result_filename=name, flag=1)
    return send_from_directory(app.config['UPLOAD_FOLDER'], name + '.jpg', as_attachment=True)

@app.route('/transfer_texture', methods = ['POST'])
def transfer_texture():
    print ("request for texture transfer received")
    app.logger.info(app.config['UPLOAD_FOLDER'])
    video = request.files['transfer']
    name = request.files['texture_filename']
    map_t.texture_path = os.path.join(app.config['UPLOAD_FOLDER'], name)
    map_t.read_texture()
    video_name = secure_filename(video.filename)
    print ("Video/image to process is {}".format(video_name))
    create_new_folder(app.config['UPLOAD_FOLDER'])
    saved_path = os.path.join(app.config['UPLOAD_FOLDER'], video_name)
    app.logger.info("saving {}".format(saved_path))
    video.save(saved_path)
    filename = process_video(saved_path, video_name)
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route("/",methods=['POST','GET'])
def index_fn():
    return "hello world"

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=9090, debug=False)
