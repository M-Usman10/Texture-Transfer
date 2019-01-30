import yaml
import os
from flask import Flask
import logging, os


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
