import argparse
import codecs
import logging
import os
import os.path as osp
import sys
import yaml
import time

from qtpy import QtCore
from qtpy import QtWidgets

from labelme import __appname__
from labelme import __version__
from labelme_app import MainWindow
from labelme.config import get_config
#from labelme.logger import logger
from labelme.utils import newIcon
import multiprocessing_logging as mpl

import pickle

from importlib import reload

def main(parent_class = None):
    
    # Hardcodes
    label_file = r'C:\Tmp\Work1\labels.yaml'
    flag_file = r'C:\Tmp\Work1\labelflags.yaml'
    #'TEMP': 'C:\\Users\\mherzo\\AppData\\Local\\Temp'
    input_dir = osp.join(os.getenv('TEMP'),'tissue_vision')
    output_dir = input_dir
    
    if not osp.exists(input_dir):
        os.mkdir(input_dir)
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--version', '-V', action='store_true', help='show version'
    )
    parser.add_argument(
        '--reset-config', action='store_true', help='reset qt config'
    )
    parser.add_argument(
        '--logger-level',
        #default='info',
        default='debug',
        choices=['debug', 'info', 'warning', 'fatal', 'error'],
        help='logger level',
    )
    parser.add_argument(
        'filename', 
        nargs='?', 
        help='image or label filename',
        default=input_dir)
    parser.add_argument(
        '--output',
        '-O',
        '-o',
        default=output_dir,
        help='output file or directory (if it ends with .json it is '
             'recognized as file, else as directory)'
    )
    default_config_file = os.path.join(os.path.expanduser('~'), '.labelmerc')
    parser.add_argument(
        '--config',
        dest='config',
        help='config file or yaml-format string (default: {})'.format(
            default_config_file
        ),
        default=default_config_file,
    )
    # config for the gui
    parser.add_argument(
        '--nodata',
        dest='store_data',
        action='store_false',
        help='stop storing image data to JSON file',
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--autosave',
        dest='auto_save',
        action='store_true',
        help='auto save',
        #default=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--nosortlabels',
        dest='sort_labels',
        action='store_false',
        help='stop sorting labels',
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--flags',
        help='comma separated list of flags OR file containing flags',
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--labelflags',
        dest='label_flags',
        help='yaml string of label specific flags OR file containing json '
             'string of label specific flags (ex. {person-\d+: [male, tall], '
             'dog-\d+: [black, brown, white], .*: [occluded]})',  # NOQA
        # default=argparse.SUPPRESS,
        default=flag_file,
    )
    parser.add_argument(
        '--labels',
        help='comma separated list of labels OR file containing labels',
        # default=argparse.SUPPRESS,
        default=label_file,
    )
    parser.add_argument(
        '--validatelabel',
        dest='validate_label',
        choices=['exact'],
        help='label validation types',
        # default=argparse.SUPPRESS,
        # action='store_true',
        default='exact'
    )
    parser.add_argument(
        '--keep-prev',
        action='store_true',
        help='keep annotation of previous frame',
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--epsilon',
        type=float,
        help='epsilon to find nearest vertex on canvas',
        default=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.version:
        print('{0} {1}'.format(__appname__, __version__))
        sys.exit(0)

    logging.basicConfig(filename=r'c:\tmp\labelme_root.log', filemode='w', level=getattr(logging, args.logger_level.upper()))
    # TODO Does mpl work?
    mpl.install_mp_handler()
    # TODO Set up root logger in __main__, not this procedure.  Use labelme.logger instead of manually setting up logger.
    logger = logging.getLogger()
    logger.info('Logger initialized')

    if hasattr(args, 'flags'):
        if os.path.isfile(args.flags):
            with codecs.open(args.flags, 'r', encoding='utf-8') as f:
                args.flags = [l.strip() for l in f if l.strip()]
        else:
            args.flags = [l for l in args.flags.split(',') if l]

    if hasattr(args, 'labels'):
        if os.path.isfile(args.labels):
            with codecs.open(args.labels, 'r', encoding='utf-8') as f:
                args.labels = [l.strip() for l in f if l.strip()]
        else:
            args.labels = [l for l in args.labels.split(',') if l]

    if hasattr(args, 'label_flags'):
        if os.path.isfile(args.label_flags):
            with codecs.open(args.label_flags, 'r', encoding='utf-8') as f:
                args.label_flags = yaml.safe_load(f)
        else:
            args.label_flags = yaml.safe_load(args.label_flags)

    config_from_args = args.__dict__
    config_from_args.pop('version')
    reset_config = config_from_args.pop('reset_config')
    filename = config_from_args.pop('filename')
    output = config_from_args.pop('output')
    config_file_or_yaml = config_from_args.pop('config')
    config = get_config(config_file_or_yaml, config_from_args)

    if not config['labels'] and config['validate_label']:
        logger.error('--labels must be specified with --validatelabel or '
                     'validate_label: true in the config file '
                     '(ex. ~/.labelmerc).')
        sys.exit(1)

    output_file = None
    output_dir = None
    if output is not None:
        if output.endswith('.json'):
            output_file = output
        else:
            output_dir = output

    translator = QtCore.QTranslator()
    translator.load(
        QtCore.QLocale.system().name(),
        osp.dirname(osp.abspath(__file__)) + '/translate'
    )
    
    # https://stackoverflow.com/a/53387775/11262633
    app = QtWidgets.QApplication.instance()
    if app is None:
        # if it does not exist then create QApplication 
        app = QtWidgets.QApplication(sys.argv)

    #app.setApplicationName(__appname__)
    #app.setWindowIcon(newIcon('icon'))
    app.installTranslator(translator)
    win = MainWindow(
        config=config,
        filename=filename,
        output_file=output_file,
        output_dir=output_dir,
        parent_class=parent_class,
    )

    if reset_config:
        logger.info('Resetting Qt config: %s' % win.settings.fileName())
        win.settings.clear()
        sys.exit(0)

    win.show()
    win.raise_()
    
    print(f'Window:  Pos {win.pos().x()}:{win.pos().y()}, Dim ({win.width()},{win.height()})')

    # TODO Get from win
    if not parent_class is None:
        #Maximized window (QT - x,y): 960,493 -- 1920, 986
        #parent_class.set_targ_rect([[0,0],[986, 1920]])
        parent_class.set_targ_rect()
        
    return app, win

class TestClass():
    def __init__(self):
        test_images_pickle = r'c:\tmp\video_input.pickle'
        if osp.exists(test_images_pickle):
            with open(test_images_pickle,'rb') as f:
                image_list = pickle.load(f)
                self.image_from_camera = image_list[0]
                
    def set_targ_rect(self, targ_rect=None):
        self.targ_rect = targ_rect
        
# this main block is required to generate executable by pyinstaller
if __name__ == '__main__':
    #TODO Better way of resetting logging
    reload(logging)
    #https://stackoverflow.com/questions/13733552/logger-configuration-to-log-to-file-and-print-to-stdout
    #logger.root.hanlders = []
    test_class = TestClass()
    app, win = main(parent_class=test_class)
    app.exec_()
    print('Terminating')
    # TODO make this part of exit from labelme_app.py ** Leaves extraneous processes **
    win.listener.stop()
    win.pointer_proc.terminate()
    time.sleep(1)
    win.pointer_proc.close()
    del app
    del win
    print('Exiting')
    sys.exit()
