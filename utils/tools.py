import yaml
from flask import Flask
import logging, os
import cv2


def iuv_files_sort(name):
    return int(name[:-8])

class Cap:
   def __init__(self, path, step_size=1):
       self.path = path
       self.step_size = step_size
       self.curr_frame_no = 0
   def __enter__(self):
       self.cap = cv2.VideoCapture(self.path)
       return self
   def read(self):
       success, frame = self.cap.read()
       if not success:
           return success, frame
       for _ in range(self.step_size):
           s, f = self.cap.read()
           if not s:
               break
       return success, frame
   def read_all(self):
       frames_list = []
       while True:
           success, frame = self.cap.read()
           if not success:
               return frames_list
           frames_list.append(frame)
           for _ in range(self.step_size-1):
               s, f = self.cap.read()
               if not s:
                   return frames_list
   def __exit__(self, a, b, c):
       self.cap.release()
       cv2.destroyAllWindows()

def save_video(images,path,fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height,width=images[0].shape[:2]
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for image in images:
        out.write(image)
    out.release()
    cv2.destroyAllWindows()

def create_new_folder(local_dir):
    newpath = local_dir
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath

def make_flask_app(config):
    app = Flask(__name__)
    file_handler = logging.FileHandler('server.log')
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
#     PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
#     UPLOAD_FOLDER = '{}/{}/'.format(PROJECT_HOME,config['input_dir'])
    UPLOAD_FOLDER = config['input_dir']
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    return app

def load_config(file):
    with open(file, 'r') as stream:
        dict_ = yaml.load(stream)
    return dict_
