# -*- coding: utf-8 -*-

import functools
import os
import os.path as osp
import pathlib
import re
import webbrowser
import pandas as pd
import time
import shutil
import datetime
import glob
import sys
import numpy as np
import subprocess

import imgviz
from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy import QtGui
from qtpy import QtWidgets

from labelme import __appname__
from labelme import PY2
from labelme import QT5

from labelme import utils
from labelme import user_extns
from labelme.config import get_config
from labelme.label_file import LabelFile
from labelme.label_file import LabelFileError
#from labelme.logger import logger
import logging
from labelme.shape import Shape
from labelme.widgets import Canvas
from labelme.widgets import LabelDialog
from labelme.widgets import LabelListWidget
from labelme.widgets import LabelListWidgetItem
from labelme.widgets import ToolBar
from labelme.widgets import UniqueLabelQListWidget
from labelme.widgets import ZoomWidget

import PIL 
from PIL import Image
import traceback

import win32con
import win32api
from pynput import keyboard

import ezdxf

from eval_with_cv2_tracking import TechnicianUI
#TODO Instead of argparse and related in this module, get values from calling program
import argparse

#from modeless_confirm import ModelessConfirm

#import cProfile

#TODO:
#1.	*Refresh file list every 1 sec
#2.	Support oriented bounding boxes
#3.	Add "Verify" to File menu
#4.	Ability to create comments
#5.	Cleaner way of handing input/output dirs.  
#a.	Show output and source directories
#b.	If change one, change the other
#6.	Don't force mouse move to fully render context menu
#7.	If delete file, don't clear image
#8.	If polygons aren't selected and choose delete, don't prompt for delete confirmation
#9.	Add VerifyFile to config shortcuts
#10.	**Verify prompts for output dir every time
#11.	**Poor performance loading images
#12.	When in insert mode, highlight polygon if select an entry in the Polygon Labels list
#13.	*Rotate picture
#15.	 * Remember last input and output folders
#16.	Enter on keypad does not cause polygons to be selected/completed
#17.  Ability to view folders (input, output) in Explorer.  Use code in Tools/exportByLot
#18.  If shape has a flag, make a different color or different outline (dash instead of lines?)
#19.  By default, hide flags and unique label docks
#20.  Update Tutorial 
#21.  Propose merge with source fork.  Update GitHub doc.
#22.  When zoom with control keys, lose center of viewing area
#23.  Look at labelme v4.5 (we have v4.2)

# FIXME
# - [medium] Set max zoom value to something big enough for FitWidth/Window

# TODO(unknown):
# - [high] Add polygon movement with arrow keys
# - [high] Deselect shape when clicking and already selected(?)
# - [low,maybe] Open images with drag & drop.
# - [low,maybe] Preview images on file dialogs.
# - Zoom is too "steppy".

# TODO(Ground Truth Tool):
# - ** Remove hardcodes - get data from Ground Truth config file
#     > Default name:  "Annotation Session Groups.txt"
#     > Format:
#           SME:./SME*
#           Techs:./Tech*
#           Ground Truth:./Ground Truth 
#     > Option to load a different file name
# - Options to select folders:
#     SME
#     Tech
#     All Defined (from config file)
#     All folders
# - For non-GT shapes, use dotted lines?
# - Ability to copy a non-GT shape and paste.  Right now, pasted shape is locked.
# - Option to create GT from aggregate of selected shapes
# - Ability to edit current "Annotation Session Groups" file (in Notepad)
# - Test - if change output dir
# - Hide annotation dock by default
# - Turn off shapes in shape list by Annotator or defect.  Use text entry instead of annotation dock?
# - * Complete locking of non-GT shapes:
#     > Don't allow Delete key 
#     > Don't show context menus except copy
#     > Check double click behavior
# - If switch mode, apply filter to file list
# - *** Create installation procedure
# - Calc IoU.  Weight IoU based on size of Ground Truth (use shapely)
# - Add GT to ./testing scripts
# - If delete label file, in GT mode don't hide image - continue to show existing annotations



LABEL_COLORMAP = imgviz.label_colormap(value=200)

TISSUE_BOUNDARY_LABEL = 'Tissue Boundary'



class MainWindow(QtWidgets.QMainWindow):

    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = 0, 1, 2

    def __init__(
        self,
        config=None,
        filename=None,
        output=None,
        output_file=None,
        output_dir=None,
        parent_class=None,
    ):
        print('In custom version of app.py')
        #TODO Use labelme.logger but capture multiprocessing output
        self.logger = logging.getLogger()
        self.logger.info('Initializing')
        if output is not None:
            self.logger.warning(
                'argument output is deprecated, use output_file instead'
            )
            if output_file is None:
                output_file = output

        # see labelme/config/default_config.yaml for valid configuration
        if config is None:
            config = get_config()
        self._config = config
        
        # Get calibration data
        # TODO Determine final calibration architecture.  Store in config?
        self.calib_factors = {'w':81.6, 'h':81.8}  # Number of pixels in uom
        self.calib_uom = 'cm'  

        super(MainWindow, self).__init__()
        self.setWindowTitle(__appname__)

        # Whether we need to save or not.
        self.dirty = False

        self._noSelectionSlot = False
        
        self.parent_class = parent_class

        # Main widgets and related state.
        self.labelDialog = LabelDialog(
            parent=self,
            labels=self._config['labels'],
            sort_labels=self._config['sort_labels'],
            show_text_field=self._config['show_label_text_field'],
            completion=self._config['label_completion'],
            fit_to_content=self._config['fit_to_content'],
            flags=self._config['label_flags']
        )

        self.labelList = LabelListWidget()
        self.lastOpenDir = None

        # ====================================================
        # -------- Dock windows - begin definitions ----------
        # ----------------------------------------------------        
        
        #------------------------------
        # Dock window - Flags
        self.flag_dock = self.flag_widget = None
        self.flag_dock = QtWidgets.QDockWidget(self.tr('Flags'), self)
        self.flag_dock.setObjectName('Flags')
        self.flag_widget = QtWidgets.QListWidget()
        if config['flags']:
            self.loadFlags({k: False for k in config['flags']})
        self.flag_dock.setWidget(self.flag_widget)
        self.flag_widget.itemChanged.connect(self.setDirty)

        #------------------------------
        # Dock window - Label list
        self.labelList.itemSelectionChanged.connect(self.labelSelectionChanged)
        self.labelList.itemDoubleClicked.connect(self.editLabel)
        self.labelList.itemChanged.connect(self.labelItemChanged)
        self.labelList.itemDropped.connect(self.labelOrderChanged)
        self.shape_dock = QtWidgets.QDockWidget(
                        self.getLabelDockTitle(), self)
        self.shape_dock.setObjectName('Labels')
        self.shape_dock.setWidget(self.labelList)

        #------------------------------
        # Dock window - Unique labels
        self.uniqLabelList = UniqueLabelQListWidget()
        self.uniqLabelList.setToolTip(self.tr(
            "Select label to start annotating for it. "
            "Press 'Esc' to deselect."))
        if self._config['labels']:
            for label in self._config['labels']:
                item = self.uniqLabelList.createItemFromLabel(label)
                self.uniqLabelList.addItem(item)
                rgb = self._get_rgb_by_label(label)
                self.uniqLabelList.setItemLabel(item, label, rgb)
        self.label_dock = QtWidgets.QDockWidget(self.tr(u'Label List'), self)
        self.label_dock.setObjectName(u'Label List')
        self.label_dock.setWidget(self.uniqLabelList)

        #------------------------------
        # Dock window - file list
        self.fileSearch = QtWidgets.QLineEdit()
        self.fileSearch.setPlaceholderText(self.tr('Search Filename'))
        self.fileSearch.textChanged.connect(self.fileSearchChanged)
        self.fileListWidget = QtWidgets.QListWidget()
        self.fileListWidget.itemSelectionChanged.connect(
            self.fileSelectionChanged
        )
        fileListLayout = QtWidgets.QVBoxLayout()
        fileListLayout.setContentsMargins(0, 0, 0, 0)
        fileListLayout.setSpacing(0)
        fileListLayout.addWidget(self.fileSearch)
        fileListLayout.addWidget(self.fileListWidget)
        self.showLabeledCheckbox = QtWidgets.QCheckBox('Show Labeled')
        fileListLayout.addWidget(self.showLabeledCheckbox)
        self.showLabeledCheckbox.stateChanged.connect(self.refreshDirImages)        
        
        self.file_dock = QtWidgets.QDockWidget(self.tr(u'File List'), self)
        self.file_dock.setObjectName(u'Files')
        fileListWidget = QtWidgets.QWidget()
        fileListWidget.setLayout(fileListLayout)
        self.file_dock.setWidget(fileListWidget)
        
        #------------------------------
        # Dock window - Annotator list
        self.annotator_dock = QtWidgets.QDockWidget(self.tr(u'Annotator List'), self)
        self.annotator_dock.setObjectName(u'Annotators')
        annotatorListLayout = QtWidgets.QVBoxLayout()
        annotatorListLayout.setContentsMargins(0, 0, 0, 0)
        annotatorListLayout.setSpacing(0)
  
        self.selLabelsToShow = QtWidgets.QComboBox()
        
        #Set color of combobox to white
        #https://stackoverflow.com/questions/54160285/how-to-set-background-color-of-qcombobox-button
        cbstyle =  "QComboBox QAbstractItemView {"
        cbstyle += " border: 1px solid grey;"
        cbstyle += " background: white;"
        cbstyle += " selection-background-color: blue;"
        cbstyle += " }"
        cbstyle += " QComboBox {"
        cbstyle += " background: white;"
        cbstyle += "}"
        self.selLabelsToShow.setStyleSheet(cbstyle)
  
        for label in self._config['labels']:
            self.selLabelsToShow.addItem(label)
        annotatorListLayout.addWidget(self.selLabelsToShow)
  
        self.annotationSessionList = QtWidgets.QListWidget()
        #TODO Get from config file and glob
        valueList = ['SME A Sarvey, T1','SME A Sarvey, T2','SME A Sarvey, T3','SME R Jabbal, T1','SME R Jabbal, T2','SME R Jabbal, T3','SME T Canty, T1','SME T Canty, T2','SME T Canty, T3']
        self.annotationSessionList = QtWidgets.QListWidget()
        for value in valueList:
              listItem = QtWidgets.QListWidgetItem(value)
              listItem.setFlags(listItem.flags() | Qt.ItemIsUserCheckable)
              listItem.setCheckState(Qt.Checked)
              self.annotationSessionList.addItem(listItem)        
        annotatorListLayout.addWidget(self.annotationSessionList)
        
        #TODO Get from Config File
        self.groundTruthDirName = 'Ground Truth'
        self.groundTruthOpacity = 255
        self.groundTruthOpacityOther = 50
        #TODO Ensure groundTruthDirName exists
        
        annotatorListWidget = QtWidgets.QWidget()
        annotatorListWidget.setLayout(annotatorListLayout)
        self.annotator_dock.setWidget(annotatorListWidget)
		
        #------------------------------
        # Dock window - Process Status
        self.process_status_dock = QtWidgets.QDockWidget(self.tr(u'Process Status'), self)
        self.process_status_dock.setObjectName(u'Process Status')
        processStatusLayout = QtWidgets.QVBoxLayout()
        processStatusLayout.setContentsMargins(0, 0, 0, 0)
        processStatusLayout.setSpacing(0)
  
        self.selNestingApproach = QtWidgets.QComboBox()
        self.selNestingApproach.setStyleSheet(cbstyle)
  
        #TODO After selecting nesting software, remove this dropdown and all references to it
        nesting_approach_list = ['PowerNest','NestFab','NestLib']
        #nesting_approach_list = ['NestFab']
        for label in nesting_approach_list:
            self.selNestingApproach.addItem(label)
        processStatusLayout.addWidget(self.selNestingApproach)
  
        self.procStatusLog = QtWidgets.QPlainTextEdit()
        self.procStatusLog.setReadOnly(True)
        processStatusLayout.addWidget(self.procStatusLog)
    
        processStatusWidget = QtWidgets.QWidget()
        processStatusWidget.setLayout(processStatusLayout)
        self.process_status_dock.setWidget(processStatusWidget)

        # ----------------------------------------------------        
        # -------- Dock windows - end definition--------------
        # ====================================================

        self.zoomWidget = ZoomWidget()

        self.canvas = self.labelList.canvas = Canvas(
            epsilon=self._config['epsilon'],
            double_click=self._config['canvas']['double_click'],
        )
        self.canvas.zoomRequest.connect(self.zoomRequest)

        scrollArea = QtWidgets.QScrollArea()
        scrollArea.setWidget(self.canvas)
        scrollArea.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scrollArea.verticalScrollBar(),
            Qt.Horizontal: scrollArea.horizontalScrollBar(),
        }
        self.canvas.scrollRequest.connect(self.scrollRequest)

        self.canvas.newShape.connect(self.newShape)
        self.canvas.shapeMoved.connect(self.setDirty)
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)

        self.setCentralWidget(scrollArea)

        features = QtWidgets.QDockWidget.DockWidgetFeatures()
        for dock in ['flag_dock', 'label_dock', 'shape_dock', 'file_dock', 'process_status_dock']:
            if not dock in self._config:
                continue
            if self._config[dock]['closable']:
                features = features | QtWidgets.QDockWidget.DockWidgetClosable
            if self._config[dock]['floatable']:
                features = features | QtWidgets.QDockWidget.DockWidgetFloatable
            if self._config[dock]['movable']:
                features = features | QtWidgets.QDockWidget.DockWidgetMovable
            getattr(self, dock).setFeatures(features)
            if self._config[dock]['show'] is False:
                getattr(self, dock).setVisible(False)

        self.addDockWidget(Qt.RightDockWidgetArea, self.flag_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.label_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.shape_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.file_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.annotator_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.process_status_dock)

        # =====================
        # Action definition begin
        # ---------------------
        action = functools.partial(utils.newAction, self)
        shortcuts = self._config['shortcuts']
        quit = action(self.tr('&Quit'), self.close, shortcuts['quit'], 'quit',
                      self.tr('Quit application'))
        open_ = action(self.tr('&Open'),
                       self.openFile,
                       shortcuts['open'],
                       'open',
                       self.tr('Open image or label file'))
        opendir = action(self.tr('&Open Dir'), self.openDirDialog,
                         shortcuts['open_dir'], 'open', self.tr(u'Open Dir'))
        openNextImg = action(
            self.tr('&Next Image'),
            self.openNextImg,
            shortcuts['open_next'],
            'next',
            self.tr(u'Open next (hold Ctl+Shift to copy labels)'),
            enabled=False,
        )
        openPrevImg = action(
            self.tr('&Prev Image'),
            self.openPrevImg,
            shortcuts['open_prev'],
            'prev',
            self.tr(u'Open prev (hold Ctl+Shift to copy labels)'),
            enabled=False,
        )
        save = action(self.tr('&Save'),
                      self.saveFile, shortcuts['save'], 'save',
                      self.tr('Save labels to file'), enabled=False)
        saveAs = action(self.tr('&Save As'), self.saveFileAs,
                        shortcuts['save_as'],
                        'save-as', self.tr('Save labels to a different file'),
                        enabled=False)

        deleteFile = action(
            self.tr('&Delete File'),
            self.deleteFile,
            shortcuts['delete_file'],
            'delete',
            self.tr('Delete current label file'),
            enabled=False)
        
        verifyFile = action(
            self.tr('&Verify File'),
            self.verifyFile,
            None,
            'verify',
            self.tr('Create label file for current image, even if no defects'),
            enabled=False)

        changeOutputDir = action(
            self.tr('&Change Output Dir'),
            slot=self.changeOutputDirDialog,
            shortcut=shortcuts['save_to'],
            icon='open',
            tip=self.tr(u'Change where annotations are loaded/saved')
        )

        saveAuto = action(
            text=self.tr('Save &Automatically'),
            slot=lambda x: self.actions.saveAuto.setChecked(x),
            icon='save',
            tip=self.tr('Save automatically'),
            checkable=True,
            enabled=False,
        )
        saveAuto.setChecked(self._config['auto_save'])

        saveWithImageData = action(
            text='Save With Image Data',
            slot=self.enableSaveImageWithData,
            tip='Save image data in label file',
            checkable=True,
            checked=self._config['store_data'],
        )


        # ===============================================
        # User extensions - begin
        # -----------------------------------------------
        exportMasks = action(
            text='Export Masks',
            slot=self.exportMasks,
            tip='Exports a set of mask images for importing into ViDi',
            enabled=False
        )

        exportByLot = action(
            text='Export By Lot',
            slot=self.exportByLot,
            tip='Exports all images for a lot into a temporary folder',
            enabled=True
        )

        launchExternalViewer = action(
            text='View Image in External Viewer',
            slot=self.launchExternalViewer,
            tip='Loads current image in external image viewer',
            enabled=False
        )

        groundTruthBuilderMode = action(
            text='Ground Truth Mode',
            slot=self.groundTruthBuilderMode,
            tip='Browse images, and for each, compare multiple annotations for the image displayed.  Create Ground Truth annotation set.',
            enabled=True,
            checkable=True
        )
        groundTruthBuilderMode.setChecked(False)

        dispSettings = action(
            text='Display Settings',
            slot=self.dispSettings,
            tip='Displays settings such as the current input and output folders',
            enabled=True
        )

        takePicture = action(
            self.tr('Take Picture'),
            self.takePicture,
            None,
            'camera_icon',
            self.tr('Take picture for annotation'),
            checkable=False,
            enabled=True,
        )
        
        nest = action(
            self.tr('Nest'),
            self.nest,
            None,
            'nesting',
            self.tr('Create cut plan based on requirements and defects'),
            checkable=False,
            enabled=True,
        )
        
        # -----------------------------------------------
        # User extensions - End
        # ===============================================

        close = action('&Close', self.closeFile, shortcuts['close'], 'close',
                       'Close current file')

        toggle_keep_prev_mode = action(
            self.tr('Keep Previous Annotation'),
            self.toggleKeepPrevMode,
            shortcuts['toggle_keep_prev_mode'], None,
            self.tr('Toggle "keep pevious annotation" mode'),
            checkable=True)
        toggle_keep_prev_mode.setChecked(self._config['keep_prev'])

        createMode = action(
            self.tr('Create Polygons'),
            lambda: self.toggleDrawMode(False, createMode='polygon'),
            shortcuts['create_polygon'],
            'objects',
            self.tr('Start drawing polygons'),
            enabled=False,
        )
        createRectangleMode = action(
            self.tr('Create Rectangle'),
            lambda: self.toggleDrawMode(False, createMode='rectangle'),
            shortcuts['create_rectangle'],
            'objects',
            self.tr('Start drawing rectangles'),
            enabled=False,
        )
        createCircleMode = action(
            self.tr('Create Circle'),
            lambda: self.toggleDrawMode(False, createMode='circle'),
            shortcuts['create_circle'],
            'objects',
            self.tr('Start drawing circles'),
            enabled=False,
        )
        createLineMode = action(
            self.tr('Create Line'),
            lambda: self.toggleDrawMode(False, createMode='line'),
            shortcuts['create_line'],
            'objects',
            self.tr('Start drawing lines'),
            enabled=False,
        )
        createPointMode = action(
            self.tr('Create Point'),
            lambda: self.toggleDrawMode(False, createMode='point'),
            shortcuts['create_point'],
            'objects',
            self.tr('Start drawing points'),
            enabled=False,
        )
        createLineStripMode = action(
            self.tr('Create LineStrip'),
            lambda: self.toggleDrawMode(False, createMode='linestrip'),
            shortcuts['create_linestrip'],
            'objects',
            self.tr('Start drawing linestrip. Ctrl+LeftClick ends creation.'),
            enabled=False,
        )
        editMode = action(self.tr('Edit Polygons'), self.setEditMode,
                          shortcuts['edit_polygon'], 'edit',
                          self.tr('Move and edit the selected polygons'),
                          enabled=False)

        delete = action(self.tr('Delete Polygons'), self.deleteSelectedShape,
                        shortcuts['delete_polygon'], 'cancel',
                        self.tr('Delete the selected polygons'), enabled=False)
        copy = action(self.tr('Duplicate Polygons'), self.copySelectedShape,
                      shortcuts['duplicate_polygon'], 'copy',
                      self.tr('Create a duplicate of the selected polygons'),
                      enabled=False)
        undoLastPoint = action(self.tr('Undo last point'),
                               self.canvas.undoLastPoint,
                               shortcuts['undo_last_point'], 'undo',
                               self.tr('Undo last drawn point'), enabled=False)
        addPointToEdge = action(
            text=self.tr('Add Point to Edge'),
            slot=self.canvas.addPointToEdge,
            shortcut=shortcuts['add_point_to_edge'],
            icon='edit',
            tip=self.tr('Add point to the nearest edge'),
            enabled=False,
        )
        removePoint = action(
            text='Remove Selected Point',
            slot=self.canvas.removeSelectedPoint,
            icon='edit',
            tip='Remove selected point from polygon',
            enabled=False,
        )

        undo = action(self.tr('Undo'), self.undoShapeEdit,
                      shortcuts['undo'], 'undo',
                      self.tr('Undo last add and edit of shape'),
                      enabled=False)

        hideAll = action(self.tr('&Hide\nPolygons'),
                         functools.partial(self.togglePolygons, False),
                         icon='eye', tip=self.tr('Hide all polygons'),
                         enabled=False)
        showAll = action(self.tr('&Show\nPolygons'),
                         functools.partial(self.togglePolygons, True),
                         icon='eye', tip=self.tr('Show all polygons'),
                         enabled=False)

        help = action(self.tr('&Tutorial'), self.tutorial, icon='help',
                      tip=self.tr('Show tutorial page'))

        zoom = QtWidgets.QWidgetAction(self)
        zoom.setDefaultWidget(self.zoomWidget)
        self.zoomWidget.setWhatsThis(
            self.tr(
                'Zoom in or out of the image. Also accessible with '
                '{} and {} from the canvas.'
            ).format(
                utils.fmtShortcut(
                    '{},{}'.format(
                        shortcuts['zoom_in'], shortcuts['zoom_out']
                    )
                ),
                utils.fmtShortcut(self.tr("Ctrl+Wheel")),
            )
        )
        self.zoomWidget.setEnabled(False)

        zoomIn = action(self.tr('Zoom &In'),
                        functools.partial(self.addZoom, 1.1),
                        shortcuts['zoom_in'], 'zoom-in',
                        self.tr('Increase zoom level'), enabled=False)
        zoomOut = action(self.tr('&Zoom Out'),
                         functools.partial(self.addZoom, 0.9),
                         shortcuts['zoom_out'], 'zoom-out',
                         self.tr('Decrease zoom level'), enabled=False)
        zoomOrg = action(self.tr('&Original size'),
                         functools.partial(self.setZoom, 100),
                         shortcuts['zoom_to_original'], 'zoom',
                         self.tr('Zoom to original size'), enabled=False)
        fitWindow = action(self.tr('&Fit Window'), self.setFitWindow,
                           shortcuts['fit_window'], 'fit-window',
                           self.tr('Zoom follows window size'), checkable=True,
                           enabled=False)
        fitWidth = action(self.tr('Fit &Width'), self.setFitWidth,
                          shortcuts['fit_width'], 'fit-width',
                          self.tr('Zoom follows window width'),
                          checkable=True, enabled=False)
        # Group zoom controls into a list for easier toggling.
        zoomActions = (self.zoomWidget, zoomIn, zoomOut, zoomOrg,
                       fitWindow, fitWidth)
        self.zoomMode = self.FIT_WINDOW
        fitWindow.setChecked(Qt.Checked)
        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        edit = action(self.tr('&Edit Label'), self.editLabel,
                      shortcuts['edit_label'], 'edit',
                      self.tr('Modify the label of the selected polygon'),
                      enabled=False)

        fill_drawing = action(
            self.tr('Fill Drawing Polygon'),
            self.canvas.setFillDrawing,
            None,
            'color',
            self.tr('Fill polygon while drawing'),
            checkable=True,
            enabled=True,
        )
        fill_drawing.trigger()

        # ---------------------
        # Action definition end
        # =====================

        # Label list context menu.
        labelMenu = QtWidgets.QMenu()
        utils.addActions(labelMenu, (edit, delete))
        self.labelList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.labelList.customContextMenuRequested.connect(
            self.popLabelListMenu)

        # Store actions for further handling.
        self.actions = utils.struct(
            saveAuto=saveAuto,
            saveWithImageData=saveWithImageData,
            changeOutputDir=changeOutputDir,
            save=save, saveAs=saveAs, open=open_, close=close,
            deleteFile=deleteFile,
            verifyFile=verifyFile,
            toggleKeepPrevMode=toggle_keep_prev_mode,
            delete=delete, edit=edit, copy=copy,
            undoLastPoint=undoLastPoint, undo=undo,
            addPointToEdge=addPointToEdge, removePoint=removePoint,
            createMode=createMode, editMode=editMode,
            createRectangleMode=createRectangleMode,
            createCircleMode=createCircleMode,
            createLineMode=createLineMode,
            createPointMode=createPointMode,
            createLineStripMode=createLineStripMode,
            zoom=zoom, zoomIn=zoomIn, zoomOut=zoomOut, zoomOrg=zoomOrg,
            fitWindow=fitWindow, fitWidth=fitWidth,
            zoomActions=zoomActions,
            openNextImg=openNextImg, openPrevImg=openPrevImg,
            fileMenuActions=(open_, opendir, save, saveAs, close, verifyFile, quit),
            tool=(),
            # XXX: need to add some actions here to activate the shortcut
            editMenu=(
                edit,
                copy,
                delete,
                None,
                undo,
                undoLastPoint,
                None,
                addPointToEdge,
                None,
                toggle_keep_prev_mode,
            ),
            # menu shown at right click
            menu=(
                createMode,
                createRectangleMode,
                createCircleMode,
                createLineMode,
                createPointMode,
                createLineStripMode,
                editMode,
                edit,
                copy,
                delete,
                undo,
                undoLastPoint,
                addPointToEdge,
                removePoint,
                verifyFile,
            ),
            onLoadActive=(
                close,
                createMode,
                createRectangleMode,
                createCircleMode,
                createLineMode,
                createPointMode,
                createLineStripMode,
                editMode,
                verifyFile,
                exportMasks,
                launchExternalViewer,
            ),
            onShapesPresent=(saveAs, hideAll, showAll),
            exportMasks=exportMasks,
            launchExternalViewer=launchExternalViewer,
            exportByLot=exportByLot,
            groundTruthBuilderMode=groundTruthBuilderMode,
            takePicture=takePicture,
            nest=nest,
        )

        self.canvas.edgeSelected.connect(self.canvasShapeEdgeSelected)
        self.canvas.vertexSelected.connect(self.actions.removePoint.setEnabled)

        self.menus = utils.struct(
            file=self.menu(self.tr('&File')),
            edit=self.menu(self.tr('&Edit')),
            view=self.menu(self.tr('&View')),
            tools=self.menu(self.tr('&Tools')),
            help=self.menu(self.tr('&Help')),
            recentFiles=QtWidgets.QMenu(self.tr('Open &Recent')),
            labelList=labelMenu,
        )

        utils.addActions(
            self.menus.file,
            (
                open_,
                openNextImg,
                openPrevImg,
                opendir,
                self.menus.recentFiles,
                save,
                saveAs,
                saveAuto,
                changeOutputDir,
                saveWithImageData,
                verifyFile,
                close,
                deleteFile,
                None,
                quit,
            ),
        )
        utils.addActions(self.menus.help, (help,))
        utils.addActions(
            self.menus.view,
            (
                self.flag_dock.toggleViewAction(),
                self.label_dock.toggleViewAction(),
                self.shape_dock.toggleViewAction(),
                self.file_dock.toggleViewAction(),
                self.process_status_dock.toggleViewAction(),
                None,
                fill_drawing,
                None,
                hideAll,
                showAll,
                None,
                zoomIn,
                zoomOut,
                zoomOrg,
                None,
                fitWindow,
                fitWidth,
                None,
            ),
        )
        utils.addActions(
            self.menus.tools,
            (
                exportMasks,
                exportByLot,
                launchExternalViewer,
                groundTruthBuilderMode,
                dispSettings,
            ),
        )

        self.menus.file.aboutToShow.connect(self.updateFileMenu)

        # Custom context menu for the canvas widget:
        utils.addActions(self.canvas.menus[0], self.actions.menu)
        utils.addActions(
            self.canvas.menus[1],
            (
                action('&Copy here', self.copyShape),
                action('&Move here', self.moveShape),
            ),
        )
        # TODO "Copy Here" function does not seem to work.  Displays a new object, but doesn't actually create the object

        self.tools = self.toolbar('Tools')
        # Menu buttons on Left
        self.actions.tool = (
            takePicture,
            nest,
            save,
            None,
            createMode,
            editMode,
            copy,
            delete,
            undo,
            None,
            zoomIn,
            zoom,
            zoomOut,
            fitWindow,
            fitWidth,
        )

        self.statusBar().showMessage(self.tr('%s started.') % __appname__)
        self.statusBar().show()

        if (output_file is not None and
            (filename is None or output_file != filename) and
            self._config['auto_save']
            ):
            self.logger.warn(
                'If `auto_save` argument is True, `output_file` argument '
                'is ignored and output filename is automatically '
                'set as IMAGE_BASENAME.json.'
            )
        self.output_file = output_file
        self.output_dir = output_dir

        # Set up image acquisition
        self.acquired_image = osp.join(self.output_dir,'cur_image.jpg')
        self.del_cur_image()

        # Application state.
        self.image = QtGui.QImage()
        self.imagePath = None
        self.recentFiles = []
        self.maxRecent = 7
        self.otherData = None
        self.zoom_level = 100
        self.fit_window = False
        self.zoom_values = {}  # key=filename, value=(zoom_mode, zoom_value)
        self.scroll_values = {
            Qt.Horizontal: {},
            Qt.Vertical: {},
        }  # key=filename, value=scroll_value
        
        # Setup related to Ground Truth Mode
        self.setupGroundTruthBuilder(refreshImageList=False)
        dfAllImages = pd.DataFrame(columns=['Image Folder', 'File Name', 'Ground Truth Group', 'Is Ground Truth'])
        dfAllImages.index.name = 'Image Path'
        self.dfAllImages = dfAllImages
        gt_grp_transforms = []
        # TODO Get from config file
        gt_grp_transforms.append(lambda x:x[4:] if len(x) > 4 and x[3] == '-' and x[:2].isnumeric() else x)
        gt_grp_transforms.append(lambda x:x.lower())
        self.gt_grp_transforms = gt_grp_transforms

        if filename is not None and osp.isdir(filename):
            self.importDirImages(filename, load=False)
        else:
            self.filename = filename

        if config['file_search']:
            self.fileSearch.setText(config['file_search'])
            self.fileSearchChanged()

        # XXX: Could be completely declarative.
        # Restore application settings.
        self.settings = QtCore.QSettings('labelme', 'labelme')
        # FIXME: QSettings.value can return None on PyQt4
        self.recentFiles = self.settings.value('recentFiles', []) or []
        size = self.settings.value('window/size', QtCore.QSize(600, 500))
        position = self.settings.value('window/position', QtCore.QPoint(0, 0))
        self.resize(size)
        self.move(position)
        # or simply:
        # self.restoreGeometry(settings['window/geometry']
        self.restoreState(
            self.settings.value('window/state', QtCore.QByteArray()))

        # Populate the File menu dynamically.
        self.updateFileMenu()
        # Since loading the file may take some time,
        # make sure it runs in the background.
        if self.filename is not None:
            self.queueEvent(functools.partial(self.loadFile, self.filename))
            
        #===================================================================
        # Launch pointer processor in a separate process to prevent blocking
        #TODO clean up process - get args from calling program
        #TODO clean up process - set up multiprocessing pool here - don't create a process which then creates a pool
        if False:
            import multiprocessing as mp
    
            parser = argparse.ArgumentParser()
            parser.add_argument(
                  '-m',
                  '--model_file',
                  # default='/tmp/mobilenet_v1_1.0_224_quant.tflite',
                  default=r'C:\Users\mherzo\Documents\GitHub\tf-models\app\autoML\models\tflite-tissue_defect_ui_20200414053330\model.tflite',
                  help='.tflite model to be executed')
            #TODO make model labels soft:  default=r'C:\Users\mherzo\Documents\GitHub\tf-models\app\ui_label_map.pbtxt',
            parser.add_argument(
                  '--input_mean',
                  default=127.5, type=float,
                  help='input_mean')
            parser.add_argument(
                  '--input_std',
                  default=127.5, type=float,
                  help='input standard deviation')
            tracker_args = parser.parse_args()
            
            # main_seq()
            # TODO Load model in tu before passing to subprocesses
            self.tu = TechnicianUI('info')
        
            # ---------------------
            # Set up mapping data
            # ---------------------
            #scale_factor = 2.0
            self.tu.scale_factor = 1.0
            # Target area for mouse movements
            # Format:  [[y1, x1], [y2,x2]]
            # -- will be set in labelme -- 
            # self.tu.set_targ_rect([[46,12], [560, 610]])
            #self.tu.set_targ_rect([[0,0], [self.pos().y(),self.pos().x()]])
            self.tu.set_targ_rect()
        
            self.pointer_proc = mp.Process(target=self.tu.main_par, args=(tracker_args,))
            
            self.logger.info('Initializing pointer')
            self.pointer_proc.start()
            self.logger.info('Initializing pointer complete')
            #===================================================================
    
            # -----------------------------------------------
            # Set up hotkey to simulate mouse click
            #
            # TODO Turn off listener when exit from program.  Currently, it continues to run in a separate thread until the Python console is destroyed
            # shortcut = QtWidgets.QShortcut(QtGui.QKeySequence('F2'), self)
            # shortcut.setContext(QtCore.Qt.ApplicationShortcut)
            # shortcut.activated.connect(self.simulate_click)
            self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)            
            self.listener.start()
    
        # Callbacks:
        self.zoomWidget.valueChanged.connect(self.paintCanvas)

        self.populateModeActions()

    # https://pynput.readthedocs.io/en/latest/keyboard.html#pynput.keyboard.Key        
    def on_press(self, key):
        if key == keyboard.Key.f2:
            #print(f'{datetime.datetime.now():%H:%M:%S} F2')   
            self.simulate_click()

    def on_release(self, key):
        pass        

    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            utils.addActions(menu, actions)
        return menu

    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName('%sToolBar' % title)
        # toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        if actions:
            utils.addActions(toolbar, actions)
        self.addToolBar(Qt.LeftToolBarArea, toolbar)
        return toolbar

    # Support Functions

    def noShapes(self):
        return not len(self.labelList)

    def populateModeActions(self):
        tool, menu = self.actions.tool, self.actions.menu
        self.tools.clear()
        utils.addActions(self.tools, tool)
        self.canvas.menus[0].clear()
        utils.addActions(self.canvas.menus[0], menu)
        self.menus.edit.clear()
        actions = (
            self.actions.createMode,
            self.actions.createRectangleMode,
            self.actions.createCircleMode,
            self.actions.createLineMode,
            self.actions.createPointMode,
            self.actions.createLineStripMode,
            self.actions.editMode,
        )
        utils.addActions(self.menus.edit, actions + self.actions.editMenu)

    def setDirty(self):
        if self._config['auto_save'] or self.actions.saveAuto.isChecked():
            label_file = user_extns.imgFileToLabelFileName(self.imagePath, self.output_dir)
            self.saveLabels(label_file)
            return
        self.dirty = True
        self.actions.save.setEnabled(True)
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)
        title = __appname__
        if self.filename is not None:
            title = '{} - {}*'.format(title, self.filename)
        self.setWindowTitle(title)

    def setClean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)
        self.actions.createMode.setEnabled(True)
        self.actions.createRectangleMode.setEnabled(True)
        self.actions.createCircleMode.setEnabled(True)
        self.actions.createLineMode.setEnabled(True)
        self.actions.createPointMode.setEnabled(True)
        self.actions.createLineStripMode.setEnabled(True)
        title = __appname__
        if self.filename is not None:
            title = '{} - {}'.format(title, self.filename)
        self.setWindowTitle(title)

        if self.hasLabelFile():
            self.actions.deleteFile.setEnabled(True)
        else:
            self.actions.deleteFile.setEnabled(False)
            
        # TODO Set state to the mode when application begins, which is neither Edit mode or Create mode.  Or, go into Edit mode upon initialization
        self.setEditMode()

    def toggleActions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.actions.zoomActions:
            z.setEnabled(value)
        for action in self.actions.onLoadActive:
            action.setEnabled(value)

    def canvasShapeEdgeSelected(self, selected, shape):
        if self.canvas.shapeIsLocked(shape):
            return
        self.actions.addPointToEdge.setEnabled(
            selected and shape and shape.canAddPoint()
        )

    def queueEvent(self, function):
        QtCore.QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)

    def resetState(self):
        self.labelList.clear()
        self.filename = None
        self.imagePath = None
        self.imageData = None
        self.labelFile = None
        self.otherData = None
        self.canvas.resetState()

    def currentItem(self):
        items = self.labelList.selectedItems()
        if items:
            return items[0]
        return None

    def addRecentFile(self, filename):
        if filename in self.recentFiles:
            self.recentFiles.remove(filename)
        elif len(self.recentFiles) >= self.maxRecent:
            self.recentFiles.pop()
        self.recentFiles.insert(0, filename)
        
    def getLabelDockTitle(self):
        base_title = self.tr('Polygon Labels')
        if self.noShapes():
            return base_title
        return f'{base_title} - # Labels: {len(self.labelList)}'

    def setLabelDockTitle(self):
        self.shape_dock.setWindowTitle(self.getLabelDockTitle())


    # Callbacks

    def undoShapeEdit(self):
        self.canvas.restoreShape()
        self.labelList.clear()
        self.loadShapes(self.canvas.shapes)
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)

    def tutorial(self):
        url = 'https://github.com/wkentaro/labelme/tree/master/examples/tutorial'  # NOQA
        webbrowser.open(url)

    def toggleDrawingSensitive(self, drawing=True):
        """Toggle drawing sensitive.

        In the middle of drawing, toggling between modes should be disabled.
        """
        self.actions.editMode.setEnabled(not drawing)
        self.actions.undoLastPoint.setEnabled(drawing)
        self.actions.undo.setEnabled(not drawing)
        self.actions.delete.setEnabled(not drawing)

    def toggleDrawMode(self, edit=True, createMode='polygon'):
        # If, in create mode, the user switches to a different shape 
        # before committing the shape, the program will crash.
        if (not edit and createMode and 
             		    self.canvas.createMode and
            		    createMode != self.canvas.createMode and
                        self.canvas.current and
                        not self.canvas.current.isClosed()):
            user_extns.dispMsgBox(f'You are currently creating a {self.canvas.createMode}.  You cannot switch to creating a {createMode}.')
            return
        self.canvas.setEditing(edit)
        self.canvas.createMode = createMode
        if edit:
            self.actions.createMode.setEnabled(True)
            self.actions.createRectangleMode.setEnabled(True)
            self.actions.createCircleMode.setEnabled(True)
            self.actions.createLineMode.setEnabled(True)
            self.actions.createPointMode.setEnabled(True)
            self.actions.createLineStripMode.setEnabled(True)
        else:
            if createMode == 'polygon':
                self.actions.createMode.setEnabled(False)
                self.actions.createRectangleMode.setEnabled(True)
                self.actions.createCircleMode.setEnabled(True)
                self.actions.createLineMode.setEnabled(True)
                self.actions.createPointMode.setEnabled(True)
                self.actions.createLineStripMode.setEnabled(True)
            elif createMode == 'rectangle':
                self.actions.createMode.setEnabled(True)
                self.actions.createRectangleMode.setEnabled(False)
                self.actions.createCircleMode.setEnabled(True)
                self.actions.createLineMode.setEnabled(True)
                self.actions.createPointMode.setEnabled(True)
                self.actions.createLineStripMode.setEnabled(True)
            elif createMode == 'line':
                self.actions.createMode.setEnabled(True)
                self.actions.createRectangleMode.setEnabled(True)
                self.actions.createCircleMode.setEnabled(True)
                self.actions.createLineMode.setEnabled(False)
                self.actions.createPointMode.setEnabled(True)
                self.actions.createLineStripMode.setEnabled(True)
            elif createMode == 'point':
                self.actions.createMode.setEnabled(True)
                self.actions.createRectangleMode.setEnabled(True)
                self.actions.createCircleMode.setEnabled(True)
                self.actions.createLineMode.setEnabled(True)
                self.actions.createPointMode.setEnabled(False)
                self.actions.createLineStripMode.setEnabled(True)
            elif createMode == "circle":
                self.actions.createMode.setEnabled(True)
                self.actions.createRectangleMode.setEnabled(True)
                self.actions.createCircleMode.setEnabled(False)
                self.actions.createLineMode.setEnabled(True)
                self.actions.createPointMode.setEnabled(True)
                self.actions.createLineStripMode.setEnabled(True)
            elif createMode == "linestrip":
                self.actions.createMode.setEnabled(True)
                self.actions.createRectangleMode.setEnabled(True)
                self.actions.createCircleMode.setEnabled(True)
                self.actions.createLineMode.setEnabled(True)
                self.actions.createPointMode.setEnabled(True)
                self.actions.createLineStripMode.setEnabled(False)
            else:
                raise ValueError('Unsupported createMode: %s' % createMode)
        self.actions.editMode.setEnabled(not edit)

    def setEditMode(self):
        self.toggleDrawMode(True)

    def updateFileMenu(self):
        current = self.filename

        def exists(filename):
            return osp.exists(str(filename))

        menu = self.menus.recentFiles
        menu.clear()
        files = [f for f in self.recentFiles if f != current and exists(f)]
        for i, f in enumerate(files):
            icon = utils.newIcon('labels')
            action = QtWidgets.QAction(
                icon, '&%d %s' % (i + 1, QtCore.QFileInfo(f).fileName()), self)
            action.triggered.connect(functools.partial(self.loadRecent, f))
            menu.addAction(action)

    def popLabelListMenu(self, point):
        self.menus.labelList.exec_(self.labelList.mapToGlobal(point))

    def validateLabel(self, label):
        # no validation
        if self._config['validate_label'] is None:
            return True

        for i in range(self.uniqLabelList.count()):
            label_i = self.uniqLabelList.item(i).data(Qt.UserRole)
            if self._config['validate_label'] in ['exact']:
                if label_i == label:
                    return True
        return False

    def editLabel(self, item=None):
        if item and not isinstance(item, LabelListWidgetItem):
            raise TypeError('item must be LabelListWidgetItem type')

        if not self.canvas.editing():
            return
        if not item:
            item = self.currentItem()
        if item is None:
            return
        shape = item.shape()
        if shape is None:
            return
        text, flags, group_id = self.labelDialog.popUp(
            text=shape.label, flags=shape.flags, group_id=shape.group_id,
        )
        if text is None:
            return
        if not self.validateLabel(text):
            self.errorMessage(
                self.tr('Invalid label'),
                self.tr(
                    "Invalid label '{}' with validation type '{}'"
                ).format(text, self._config['validate_label'])
            )
            return
        shape.label = text
        shape.flags = flags
        shape.group_id = group_id
        item.setText(self.getShapeDisplayLabel(shape))
        self.setDirty()
        if not self.uniqLabelList.findItemsByLabel(shape.label):
            item = QtWidgets.QListWidgetItem()
            item.setData(Qt.UserRole, shape.label)
            self.uniqLabelList.addItem(item)

    def fileSearchChanged(self):
        self.importDirImages(
            self.lastOpenDir,
            pattern=self.fileSearch.text(),
            load=False,
        )

    def fileSelectionChanged(self):
        items = self.fileListWidget.selectedItems()
        if not items:
            return
        item = items[0]

        if not self.mayContinue():
            return

        currIndex = self.imageList.index(str(item.text()))
        if currIndex < len(self.imageList):
            filename = self.imageList[currIndex]
            if filename:
                self.loadFile(filename)
                #cProfile.runctx(fr'self.loadFile(r"{filename}")',globals(),locals(), filename=r'c:\tmp\profile.txt')

    # React to canvas signals.
    def shapeSelectionChanged(self, selected_shapes):
        self._noSelectionSlot = True
        for shape in self.canvas.selectedShapes:
            shape.selected = False
        self.labelList.clearSelection()
        self.canvas.selectedShapes = selected_shapes
        for shape in self.canvas.selectedShapes:
            shape.selected = True
            item = self.labelList.findItemByShape(shape)
            self.labelList.selectItem(item)
            self.labelList.scrollToItem(item)
        self._noSelectionSlot = False
        n_selected = len(selected_shapes)
        self.actions.delete.setEnabled(n_selected)
        self.actions.copy.setEnabled(n_selected)
        self.actions.edit.setEnabled(n_selected == 1)

    def getShapeDisplayLabel(self, shape):
        source = None
        group_id = None
        # TODO When import shape info from .json, create a "shape" object, instead of just a "dict".
        #      - This will be a lot of work, but will avoid having to translate between dict and object formats
        if type(shape).__name__ == 'dict':
            label = shape['label']
            if 'group_id' in shape:
                group_id = shape['group_id']
            if 'source' in shape:
                source = shape['source']
        else:  #object
            label = shape.label
            group_id = shape.group_id
            if hasattr(shape,'source'):
                source = shape.source
        text = f'{label}'
        if group_id:
            text += f' ({group_id})'
        if source:
            text += f' ({source})'
        return text

    def addLabel(self, shape):
        text = self.getShapeDisplayLabel(shape)
        label_list_item = LabelListWidgetItem(text, shape)
        self.labelList.addItem(label_list_item)
        if not self.uniqLabelList.findItemsByLabel(shape.label):
            item = self.uniqLabelList.createItemFromLabel(shape.label)
            self.uniqLabelList.addItem(item)
            rgb = self._get_rgb_by_label(shape.label)
            self.uniqLabelList.setItemLabel(item, shape.label, rgb)
        self.labelDialog.addLabelHistory(shape.label)
        for action in self.actions.onShapesPresent:
            action.setEnabled(True)

        rgb = self._get_rgb_by_label(shape.label)
        if rgb is None:
            return

        r, g, b = rgb
        label_list_item.setText(
            '{} <font color="#{:02x}{:02x}{:02x}">●</font>'
            .format(text, r, g, b)
        )
        if hasattr(shape,'opacity'):
            opacity=shape.opacity
        else:
            opacity=255
        shape.line_color = QtGui.QColor(r, g, b, opacity)
        shape.vertex_fill_color = QtGui.QColor(r, g, b, opacity)
        shape.hvertex_fill_color = QtGui.QColor(255, 255, 255, opacity)
        shape.fill_color = QtGui.QColor(r, g, b, 128)
        shape.select_line_color = QtGui.QColor(255, 255, 255, opacity)
        shape.select_fill_color = QtGui.QColor(r, g, b, 155)
        self.setLabelDockTitle()

    def _get_rgb_by_label(self, label):
        if self._config['shape_color'] == 'auto':
            item = self.uniqLabelList.findItemsByLabel(label)[0]
            label_id = self.uniqLabelList.indexFromItem(item).row() + 1
            label_id += self._config['shift_auto_shape_color']
            return LABEL_COLORMAP[label_id % len(LABEL_COLORMAP)]
        elif (self._config['shape_color'] == 'manual' and
              self._config['label_colors'] and
              label in self._config['label_colors']):
            return self._config['label_colors'][label]
        elif self._config['default_shape_color']:
            return self._config['default_shape_color']

    def remLabels(self, shapes):
        for shape in shapes:
            item = self.labelList.findItemByShape(shape)
            self.labelList.removeItem(item)
        self.setLabelDockTitle()

    def loadShapes(self, shapes, replace=True):
        self._noSelectionSlot = True
        for shape in shapes:
            self.addLabel(shape)
        self.labelList.clearSelection()
        self._noSelectionSlot = False
        self.canvas.loadShapes(shapes, replace=replace)
        self.setLabelDockTitle()

    def loadLabels(self, shapes):
        s = []
        for shape in shapes:
            label = shape['label']
            # Temporary attributes.  If 'source' is defined, assume others are as well
            if self.isGroundTruthBuilderMode and 'source' in shape:
                source = shape['source']
                opacity = shape['opacity']
                locked = shape['locked']
                disp_label = shape['disp_label']
            else:
                source = None
                opacity = None
                locked = False
                disp_label = self.getShapeDisplayLabel(shape)
            points = shape['points']
            shape_type = shape['shape_type']
            flags = shape['flags']
            group_id = shape['group_id']
            other_data = shape['other_data']

            shape = Shape(
                label=label,
                shape_type=shape_type,
                group_id=group_id,
                locked=locked,
            )
            for x, y in points:
                shape.addPoint(QtCore.QPointF(x, y))
            shape.close()

            default_flags = {}
            if self._config['label_flags']:
                for pattern, keys in self._config['label_flags'].items():
                    if re.match(pattern, label):
                        for key in keys:
                            default_flags[key] = False
            shape.flags = default_flags
            shape.flags.update(flags)
            shape.other_data = other_data
            if source:
                shape.source = source
                shape.opacity = opacity
                shape.locked = locked
                shape.disp_label = disp_label

            s.append(shape)
        self.loadShapes(s)

    def loadFlags(self, flags):
        self.flag_widget.clear()
        for key, flag in flags.items():
            item = QtWidgets.QListWidgetItem(key)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if flag else Qt.Unchecked)
            self.flag_widget.addItem(item)

    def saveLabels(self, filename):
        lf = LabelFile()

        def format_shape(s):
            data = s.other_data.copy()
            data.update(dict(
                label=s.label.encode('utf-8') if PY2 else s.label,
                points=[(p.x(), p.y()) for p in s.points],
                group_id=s.group_id,
                shape_type=s.shape_type,
                flags=s.flags,
            ))
            return data

        shapes = [format_shape(item.shape())
                          for item in self.labelList
                          if not (self.isGroundTruthBuilderMode and 
                                  hasattr(item.shape(), 'source') and 
                                  item.shape().source != 'Ground Truth')
                 ]
        flags = {}
        for i in range(self.flag_widget.count()):
            item = self.flag_widget.item(i)
            key = item.text()
            flag = item.checkState() == Qt.Checked
            flags[key] = flag
        try:
            imagePath = osp.relpath(
                self.imagePath, osp.dirname(filename))
            imageData = self.imageData if self._config['store_data'] else None
            if osp.dirname(filename) and not osp.exists(osp.dirname(filename)):
                os.makedirs(osp.dirname(filename))
            lf.save(
                filename=filename,
                shapes=shapes,
                imagePath=imagePath,
                imageData=imageData,
                imageHeight=self.image.height(),
                imageWidth=self.image.width(),
                otherData=self.otherData,
                flags=flags,
            )
            self.labelFile = lf
            items = self.fileListWidget.findItems(
                self.imagePath, Qt.MatchExactly
            )
            if len(items) > 0:
                if len(items) != 1:
                    raise RuntimeError('There are duplicate files.')
                items[0].setCheckState(Qt.Checked)
            # disable allows next and previous image to proceed
            # self.filename = filename
            return True
        except LabelFileError as e:
            self.errorMessage(
                self.tr('Error saving label data'),
                self.tr('<b>%s</b>') % e
            )
            return False

    def copySelectedShape(self):
        added_shapes = self.canvas.copySelectedShapes()
        self.labelList.clearSelection()
        for shape in added_shapes:
            #TODO See note in getShapeDisplayLabel regarding shape object vs dict
            if self.canvas.shapeIsLocked(shape):
                shape.source = self.groundTruthDirName
                shape.opacity = self.groundTruthOpacity
                shape.locked = False
                shape.disp_label = self.getShapeDisplayLabel(shape)
                self.canvas.repaint()
            self.addLabel(shape)
        self.setDirty()

    def labelSelectionChanged(self):
        if self._noSelectionSlot:
            return
        if self.canvas.editing():
            selected_shapes = []
            for item in self.labelList.selectedItems():
                selected_shapes.append(item.shape())
            if selected_shapes:
                self.canvas.selectShapes(selected_shapes)
            else:
                self.canvas.deSelectShape()

    def labelItemChanged(self, item):
        shape = item.shape()
        self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)

    def labelOrderChanged(self):
        self.setDirty()
        self.canvas.loadShapes([item.shape() for item in self.labelList])

    # Callback functions:

    def newShape(self):
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """
        items = self.uniqLabelList.selectedItems()
        text = None
        if items:
            text = items[0].data(Qt.UserRole)
        flags = {}
        group_id = None
        if self._config['display_label_popup'] or not text:
            previous_text = self.labelDialog.edit.text()
            text, flags, group_id = self.labelDialog.popUp(text)
            if not text:
                self.labelDialog.edit.setText(previous_text)

        if text and not self.validateLabel(text):
            self.errorMessage(
                self.tr('Invalid label'),
                self.tr(
                    "Invalid label '{}' with validation type '{}'"
                ).format(text, self._config['validate_label'])
            )
            text = ''
        if text:
            self.labelList.clearSelection()
            shape = self.canvas.setLastLabel(text, flags)
            shape.group_id = group_id
            self.addLabel(shape)
            self.actions.editMode.setEnabled(True)
            self.actions.undoLastPoint.setEnabled(False)
            self.actions.undo.setEnabled(True)
            self.setDirty()
        else:
            self.canvas.undoLastLine()
            self.canvas.shapesBackups.pop()

    def scrollRequest(self, delta, orientation):
        units = - delta * 0.1  # natural scroll
        bar = self.scrollBars[orientation]
        value = bar.value() + bar.singleStep() * units
        self.setScroll(orientation, value)

    def setScroll(self, orientation, value):
        self.scrollBars[orientation].setValue(value)
        self.scroll_values[orientation][self.filename] = value

    def setZoom(self, value):
        self.actions.fitWidth.setChecked(False)
        self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(value)
        self.zoom_values[self.filename] = (self.zoomMode, value)

    def addZoom(self, increment=1.1):
        self.setZoom(self.zoomWidget.value() * increment)

    def zoomRequest(self, delta, pos):
        canvas_width_old = self.canvas.width()
        units = 1.1
        if delta < 0:
            units = 0.9
        self.addZoom(units)

        canvas_width_new = self.canvas.width()
        if canvas_width_old != canvas_width_new:
            canvas_scale_factor = canvas_width_new / canvas_width_old

            x_shift = round(pos.x() * canvas_scale_factor) - pos.x()
            y_shift = round(pos.y() * canvas_scale_factor) - pos.y()

            self.setScroll(
                Qt.Horizontal,
                self.scrollBars[Qt.Horizontal].value() + x_shift,
            )
            self.setScroll(
                Qt.Vertical,
                self.scrollBars[Qt.Vertical].value() + y_shift,
            )

    def setFitWindow(self, value=True):
        if value:
            self.actions.fitWidth.setChecked(False)
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
        if value:
            self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

    def togglePolygons(self, value):
        for item in self.labelList:
            item.setCheckState(Qt.Checked if value else Qt.Unchecked)

    def loadFile(self, filename=None):
        """Load the specified file, or the last opened file if None."""
        # changing fileListWidget loads file
        if (filename in self.imageList and
                self.fileListWidget.currentRow() !=
                self.imageList.index(filename)):
            self.fileListWidget.setCurrentRow(self.imageList.index(filename))
            self.fileListWidget.repaint()
            return

        self.resetState()
        self.canvas.setEnabled(False)
        if filename is None:
            filename = self.settings.value('filename', '')
        filename = str(filename)
        if not QtCore.QFile.exists(filename):
            self.errorMessage(
                self.tr('Error opening file'),
                self.tr('No such file: <b>%s</b>') % filename
            )
            return False
        # assumes same name, but json extension
        self.status(self.tr("Loading %s...") % osp.basename(str(filename)))
        label_file = user_extns.imgFileToLabelFileName(filename, self.output_dir)
        if QtCore.QFile.exists(label_file) and \
                LabelFile.is_label_file(label_file):
            try:
                self.labelFile = LabelFile(label_file)
            except LabelFileError as e:
                self.errorMessage(
                    self.tr('Error opening file'),
                    self.tr(
                        "<p><b>%s</b></p>"
                        "<p>Make sure <i>%s</i> is a valid label file."
                    ) % (e, label_file)
                )
                self.status(self.tr("Error reading %s") % label_file)
                return False
            self.imageData = self.labelFile.imageData
            self.imagePath = osp.join(
                osp.dirname(label_file),
                self.labelFile.imagePath,
            )
            self.otherData = self.labelFile.otherData
        else:
            self.imageData = LabelFile.load_image_file(filename)
            if self.imageData:
                self.imagePath = filename
            self.labelFile = None
        image = QtGui.QImage.fromData(self.imageData)

        if image.isNull():
            formats = ['*.{}'.format(fmt.data().decode())
                       for fmt in QtGui.QImageReader.supportedImageFormats()]
            self.errorMessage(
                self.tr('Error opening file'),
                self.tr(
                    '<p>Make sure <i>{0}</i> is a valid image file.<br/>'
                    'Supported image formats: {1}</p>'
                ).format(filename, ','.join(formats))
            )
            self.status(self.tr("Error reading %s") % filename)
            return False
        self.image = image
        self.filename = filename
        if self._config['keep_prev']:
            prev_shapes = self.canvas.shapes
        self.canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
        flags = {k: False for k in self._config['flags'] or []}
        shapes = []
        if self.labelFile:
            for s in self.labelFile.shapes:
                shapes.append(s)
            if self.labelFile.flags is not None:
                flags.update(self.labelFile.flags)
        # Display additional labels if needed.  Assume image is the same for all label files selected
        if self.isGroundTruthBuilderMode:
            #self.logger.debug('Loading ground truth data')
            for s in shapes:
                s['source'] = self.groundTruthDirName
                s['opacity'] = self.groundTruthOpacity
                s['locked'] = False
                s['disp_label'] = self.getShapeDisplayLabel(s)
            gt_grp = self.dfAllImages['Ground Truth Group'].loc[filename]
            addl_files = self.dfAllImages[(self.dfAllImages['Ground Truth Group'] == gt_grp)
                                           & (~self.dfAllImages['Is Ground Truth'])].index
            for filename_addl in addl_files:
                label_file_addl = user_extns.imgFileToLabelFileName(filename_addl, self.output_dir)
                labelFile_addl = None
                if QtCore.QFile.exists(label_file_addl) and \
                        LabelFile.is_label_file(label_file_addl):
                    try:
                        labelFile_addl = LabelFile(label_file_addl, loadImage=False)
                    except LabelFileError as e:
                        self.errorMessage(
                            self.tr('Error opening file'),
                            self.tr(
                                "<p><b>%s</b></p>"
                                "<p>Make sure <i>%s</i> is a valid label file."
                            ) % (e, label_file_addl)
                        )
                        self.status(self.tr("Error reading %s") % label_file_addl)
                        return False
                if labelFile_addl:
                    for s in labelFile_addl.shapes:
                        s['source'] = self.dfAllImages['Image Folder'].loc[filename_addl]
                        s['opacity'] = self.groundTruthOpacityOther
                        s['locked'] = True
                        s['disp_label'] = self.getShapeDisplayLabel(s)
                        shapes.append(s)
                    if labelFile_addl.flags is not None:
                        #TODO Test flags from multiple annotations
                        flags.update(labelFile_addl.flags)
        self.loadLabels(shapes)
        self.loadFlags(flags)
        if self._config['keep_prev'] and self.noShapes():
            self.loadShapes(prev_shapes, replace=False)
            self.setDirty()
        else:
            self.setClean()
        self.canvas.setEnabled(True)
        # set zoom values
        is_initial_load = not self.zoom_values
        if self.filename in self.zoom_values:
            self.zoomMode = self.zoom_values[self.filename][0]
            self.setZoom(self.zoom_values[self.filename][1])
        elif is_initial_load or not self._config['keep_prev_scale']:
            self.adjustScale(initial=True)
        # set scroll values
        for orientation in self.scroll_values:
            if self.filename in self.scroll_values[orientation]:
                self.setScroll(
                    orientation, self.scroll_values[orientation][self.filename]
                )
        self.paintCanvas()
        self.addRecentFile(self.filename)
        self.toggleActions(True)
        self.status(self.tr("Loaded %s") % osp.basename(str(filename)))
        return True

    def resizeEvent(self, event):
        if self.canvas and not self.image.isNull()\
           and self.zoomMode != self.MANUAL_ZOOM:
            self.adjustScale()
        super(MainWindow, self).resizeEvent(event)

    def paintCanvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()

    def adjustScale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        value = int(100 * value)
        self.zoomWidget.setValue(value)
        self.zoom_values[self.filename] = (self.zoomMode, value)

    def scaleFitWindow(self):
        """Figure out the size of the pixmap to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.centralWidget().width() - e
        h1 = self.centralWidget().height() - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() - 2.0
        return w / self.canvas.pixmap.width()

    def enableSaveImageWithData(self, enabled):
        self._config['store_data'] = enabled
        self.actions.saveWithImageData.setChecked(enabled)

    def closeEvent(self, event):
        if not self.mayContinue():
            event.ignore()
        self.settings.setValue(
            'filename', self.filename if self.filename else '')
        self.settings.setValue('window/size', self.size())
        self.settings.setValue('window/position', self.pos())
        self.settings.setValue('window/state', self.saveState())
        self.settings.setValue('recentFiles', self.recentFiles)
        # ask the use for where to save the labels
        # self.settings.setValue('window/geometry', self.saveGeometry())

    # User Dialogs #

    def loadRecent(self, filename):
        if self.mayContinue():
            self.loadFile(filename)

    def openPrevImg(self, _value=False):
        keep_prev = self._config['keep_prev']
        if Qt.KeyboardModifiers() == (Qt.ControlModifier | Qt.ShiftModifier):
            self._config['keep_prev'] = True

        if not self.mayContinue():
            return

        if len(self.imageList) <= 0:
            return

        if self.filename is None:
            return

        currIndex = self.imageList.index(self.filename)
        if currIndex - 1 >= 0:
            filename = self.imageList[currIndex - 1]
            if filename:
                self.loadFile(filename)

        self._config['keep_prev'] = keep_prev

    def openNextImg(self, _value=False, load=True):
        keep_prev = self._config['keep_prev']
        if Qt.KeyboardModifiers() == (Qt.ControlModifier | Qt.ShiftModifier):
            self._config['keep_prev'] = True

        if not self.mayContinue():
            return

        if len(self.imageList) <= 0:
            return

        filename = None
        if self.filename is None:
            filename = self.imageList[0]
        else:
            currIndex = self.imageList.index(self.filename)
            if currIndex + 1 < len(self.imageList):
                filename = self.imageList[currIndex + 1]
            else:
                filename = self.imageList[-1]
        self.filename = filename

        if self.filename and load:
            self.loadFile(self.filename)

        self._config['keep_prev'] = keep_prev

    def openFile(self, _value=False):
        if not self.mayContinue():
            return
        path = osp.dirname(str(self.filename)) if self.filename else '.'
        formats = ['*.{}'.format(fmt.data().decode())
                   for fmt in QtGui.QImageReader.supportedImageFormats()]
        filters = self.tr("Image & Label files (%s)") % ' '.join(
            formats + ['*%s' % LabelFile.suffix])
        filename = QtWidgets.QFileDialog.getOpenFileName(
            self, self.tr('%s - Choose Image or Label file') % __appname__,
            path, filters)
        if QT5:
            filename, _ = filename
        filename = str(filename)
        if filename:
            self.loadFile(filename)

    def changeOutputDirDialog(self, _value=False):
        default_output_dir = self.output_dir
        if default_output_dir is None and self.filename:
            default_output_dir = osp.dirname(self.filename)
        if default_output_dir is None:
            default_output_dir = self.currentPath()

        output_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr('%s - Save/Load Annotations in Directory') % __appname__,
            default_output_dir,
            QtWidgets.QFileDialog.ShowDirsOnly |
            QtWidgets.QFileDialog.DontResolveSymlinks,
        )
        output_dir = str(output_dir)

        if not output_dir:
            return

        self.output_dir = output_dir

        self.statusBar().showMessage(
            self.tr('%s . Annotations will be saved/loaded in %s') %
            ('Change Annotations Dir', self.output_dir))
        self.statusBar().show()

        current_filename = self.filename
        self.importDirImages(self.lastOpenDir, load=False)

        if current_filename in self.imageList:
            # retain currently selected file
            self.fileListWidget.setCurrentRow(
                self.imageList.index(current_filename))
            self.fileListWidget.repaint()

    def saveFile(self, _value=False, verify=False):
        if not verify:
            assert not self.image.isNull(), "cannot save empty image"
        if verify or self._config['flags'] or self.hasLabels():
            if self.labelFile:
                # DL20180323 - overwrite when in directory
                self._saveFile(self.labelFile.filename)
            elif self.output_file:
                self._saveFile(self.output_file)
                self.close()
            else:
                self._saveFile(self.saveFileDialog())

    def saveFileAs(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        if self.hasLabels():
            self._saveFile(self.saveFileDialog())

    def saveFileDialog(self):
        caption = self.tr('%s - Choose File') % __appname__
        filters = self.tr('Label files (*%s)') % LabelFile.suffix
        if self.output_dir:
            dlg = QtWidgets.QFileDialog(
                self, caption, self.output_dir, filters
            )
        else:
            dlg = QtWidgets.QFileDialog(
                self, caption, self.currentPath(), filters
            )
        dlg.setDefaultSuffix(LabelFile.suffix[1:])
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.setOption(QtWidgets.QFileDialog.DontConfirmOverwrite, False)
        dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, False)
        basename = osp.basename(osp.splitext(self.filename)[0])
        if self.output_dir:
            default_labelfile_name = osp.join(
                self.output_dir, basename + LabelFile.suffix
            )
        else:
            default_labelfile_name = osp.join(
                self.currentPath(), basename + LabelFile.suffix
            )
        filename = dlg.getSaveFileName(
            self, self.tr('Choose File'), default_labelfile_name,
            self.tr('Label files (*%s)') % LabelFile.suffix)
        if isinstance(filename, tuple):
            filename, _ = filename
        return filename

    def _saveFile(self, filename):
        if filename and self.saveLabels(filename):
            self.addRecentFile(filename)
            self.setClean()

    def closeFile(self, _value=False):
        if not self.mayContinue():
            return
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)
        self.actions.saveAs.setEnabled(False)

    def getLabelFile(self):
        if self.filename.lower().endswith('.json'):
            label_file = self.filename
        else:
            label_file = osp.splitext(self.filename)[0] + '.json'

        return label_file

    def verifyFile(self):
        self.saveFile(verify=True)

    def deleteFile(self):
        mb = QtWidgets.QMessageBox
        msg = self.tr('You are about to permanently delete this label file, '
                      'proceed anyway?')
        answer = mb.warning(self, self.tr('Attention'), msg, mb.Yes | mb.No)
        if answer != mb.Yes:
            return

        label_file = self.getLabelFile()
        if osp.exists(label_file):
            os.remove(label_file)
            self.logger.info('Label file is removed: {}'.format(label_file))

            item = self.fileListWidget.currentItem()
            item.setCheckState(Qt.Unchecked)

            self.resetState()
            
    # Message Dialogs. #
    def hasLabels(self):
        if self.noShapes():
            self.errorMessage(
                'No objects labeled',
                'You must label at least one object to save the file.')
            return False
        return True

    def hasLabelFile(self):
        if self.filename is None:
            return False

        label_file = self.getLabelFile()
        return osp.exists(label_file)

    def mayContinue(self):
        if not self.dirty:
            return True
        mb = QtWidgets.QMessageBox
        msg = self.tr('Save annotations to "{}" before closing?').format(
            self.filename)
        answer = mb.question(self,
                             self.tr('Save annotations?'),
                             msg,
                             mb.Save | mb.Discard | mb.Cancel,
                             mb.Save)
        if answer == mb.Discard:
            return True
        elif answer == mb.Save:
            self.saveFile()
            return True
        else:  # answer == mb.Cancel
            return False

    def errorMessage(self, title, message):
        return QtWidgets.QMessageBox.critical(
            self, title, '<p><b>%s</b></p>%s' % (title, message))

    def currentPath(self):
        return osp.dirname(str(self.filename)) if self.filename else '.'

    def toggleKeepPrevMode(self):
        self._config['keep_prev'] = not self._config['keep_prev']

    def deleteSelectedShape(self):
        yes, no = QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No
        msg = self.tr(
            'You are about to permanently delete {} polygons, '
            'proceed anyway?'
        ).format(len(self.canvas.selectedShapes))
        if yes == QtWidgets.QMessageBox.warning(
                self, self.tr('Attention'), msg,
                yes | no):
            self.remLabels(self.canvas.deleteSelected())
            self.setDirty()
            if self.noShapes():
                for action in self.actions.onShapesPresent:
                    action.setEnabled(False)

    def copyShape(self):
        self.canvas.endMove(copy=True)
        self.labelList.clearSelection()
        for shape in self.canvas.selectedShapes:
            self.addLabel(shape)
        self.setDirty()

    def moveShape(self):
        self.canvas.endMove(copy=False)
        self.setDirty()

    def openDirDialog(self, _value=False, dirpath=None):
        if not self.mayContinue():
            return

        defaultOpenDirPath = dirpath if dirpath else '.'
        if self.lastOpenDir and osp.exists(self.lastOpenDir):
            defaultOpenDirPath = self.lastOpenDir
        else:
            defaultOpenDirPath = osp.dirname(self.filename) \
                if self.filename else '.'

        targetDirPath = str(QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr('%s - Open Directory') % __appname__,
            defaultOpenDirPath,
            QtWidgets.QFileDialog.ShowDirsOnly |
            QtWidgets.QFileDialog.DontResolveSymlinks))
        self.importDirImages(targetDirPath)

    def refreshDirImages(self):
        #print(f'dirpath={dirpath}.  showLabeledCheckbox.isChecked()={self.showLabeledCheckbox.isChecked()}')
        self.importDirImages(self.lastOpenDir, pattern=self.fileSearch.text(), load=False)

    @property
    def imageList(self):
        lst = []
        for i in range(self.fileListWidget.count()):
            item = self.fileListWidget.item(i)
            lst.append(item.text())
        return lst

    # TODO If self.scanAllImages takes too long, only perform this if needed.  E.g. if filtering, or switching GroundTruthMode, not needed.
    def importDirImages(self, dirpath, pattern=None, load=True):
        
        def parse_img_path(file_path):
            parts = pathlib.Path(file_path).parts
            file_path = parts[-1]
            gt_grp = file_path
            for fn in self.gt_grp_transforms:
                gt_grp = fn(gt_grp)
            return [parts[-2], file_path, gt_grp] 

        self.actions.openNextImg.setEnabled(True)
        self.actions.openPrevImg.setEnabled(True)

        if not self.mayContinue() or not dirpath:
            return

        self.lastOpenDir = dirpath
        self.filename = None
        self.fileListWidget.clear()
        all_images = self.scanAllImages(dirpath)
        if self.isGroundTruthBuilderMode:
            self.dfAllImages.drop(self.dfAllImages.index, inplace=True)
            df = self.dfAllImages
            #TODO Find more elegant way of setting index before adding data
            df['Image Folder'] = [None for i in range(len(all_images))]
            df.index = all_images
            # row.name in the apply function below is the index of the row being processed
            df[['Image Folder','File Name', 'Ground Truth Group']] = df.apply(lambda row:parse_img_path(row.name), 
                                                        axis = 1, result_type = 'expand')
            df['Is Ground Truth'] = df['Image Folder'].str.upper() == self.groundTruthDirName.upper()
            self.dfAllImages = df
        for filename in all_images:
            if pattern and pattern not in filename:
                continue
            # TODO:  Support XML and other label file formats
            label_file = user_extns.imgFileToLabelFileName(filename, self.output_dir)
            good_label_file = QtCore.QFile.exists(label_file) and \
                    LabelFile.is_label_file(label_file)
            if not self.showLabeledCheckbox.isChecked() and good_label_file:
                continue
            if self.isGroundTruthBuilderMode:
                if not self.dfAllImages['Is Ground Truth'].loc[filename]:
                    continue
            item = QtWidgets.QListWidgetItem(filename)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            if good_label_file:
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
            self.fileListWidget.addItem(item)
        self.openNextImg(load=load)

    def scanAllImages(self, folderPath):
        extensions = ['.%s' % fmt.data().decode("ascii").lower()
                      for fmt in QtGui.QImageReader.supportedImageFormats()]
        images = []

        for root, dirs, files in os.walk(folderPath):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    relativePath = osp.join(root, file)
                    images.append(relativePath)
        images.sort(key=lambda x: x.lower())
        return images

    # This routine is related to user_extns.exportAnnotationsForImage, but it is not the same:
    # The user could have unsaved annotations when they choose export
    # TODO Verify that this processes unsaved annotations.  May need to read from self.labelList instead of self.labelFile
    def exportMasks(self):
        targ_dir = r'c:\tmp\work3'
        targ_dir_and_prefix = targ_dir
        targ_dir_and_prefix = user_extns.inputdialog(msg=r'Target folder ',default_value=targ_dir_and_prefix).value
        if not targ_dir_and_prefix:
            msg = 'No images exported'
            self.status(msg)
            print(msg)
            return
        labels_to_export = set([shape['label'] for shape in self.labelFile.shapes])
        print(f'labels_to_export={labels_to_export}')
        for label in labels_to_export:
            shapes_to_export = [s for s in self.labelFile.shapes if s['label'] == label]
            print(f'shapes_to_export={shapes_to_export}')
            #self.loadLabels(shapes_to_export)
            # Manually draw shapes - can't seem to get a bitmap from Qt
			
            overlay_over_image = False #Useful for debugging -- ensure annotations are properly placed on picture
            if overlay_over_image:
                pixmap = self.canvas.pixmap.copy() #QtGui.QPixmap()
            else:
                cur_pixmap_size = self.canvas.pixmap.size()
                pixmap = QtGui.QPixmap(cur_pixmap_size.width(), cur_pixmap_size.height())
            p = QtGui.QPainter(pixmap)
            for s in shapes_to_export:
                self.paint_object(p, s)
            p.end()
            img_to_export = pixmap.toImage()
            img_to_export.convertToFormat(QtGui.QImage.Format_Indexed8)
            basename = osp.basename(self.labelFile.filename)
            basename = osp.splitext(basename)[0]
            targ_file = osp.join(targ_dir_and_prefix,basename + f'_{label.replace("/","")}.png')
            #TODO Move to PIL without saving to disk in order 
            img_to_export.save(targ_file)
                    
            img_tmp = PIL.Image.open(targ_file)
            img_tmp = img_tmp.convert("L")
            img_tmp.save(targ_file)    
        msg = 'Image export complete'
        self.status(msg)
        print(msg)
            
    # Paint a Shape object using a painter
    #
    # p = Painter
    # s_dict = Shape dict
    # s_obj = Shape obj
    #
    def paint_object(self, p, s_dict = None, s_obj = None, color = None):
        color_obj = None
        if color:
            color_obj = QtGui.QColor(color)
        if s_dict:
            s_obj_new = Shape(label=s_dict['label'], 
                          shape_type=s_dict['shape_type'],
                          flags=s_dict['flags'], 
                          group_id=s_dict['group_id'],
                          line_color=color_obj)
            points = s_dict['points']
            for pt in points:
                s_obj_new.addPoint(QtCore.QPointF(pt[0],pt[1]))
        elif s_obj:
            s_obj_new = Shape(label=s_obj.label, 
                          shape_type=s_obj.shape_type,
                          flags=s_obj.flags, 
                          group_id=s_obj.group_id,
                          line_color=color_obj)
            points = s_obj.points
            for pt in points:
                s_obj_new.addPoint(QtCore.QPointF(pt.x(),pt.y()))
        else:
            return
        if color_obj:
            s_obj_new.fill_color = color_obj
        s_obj_new.selected = False
        s_obj_new.close = True
        s_obj_new.fill = True
        s_obj_new.point_size = 0
        s_obj_new.paint(p)


    def exportByLot(self):
        user_extns.exportByLot()
        
    def launchExternalViewer(self):
        user_extns.launchExternalViewer(self.filename)

    def groundTruthBuilderMode(self):
        self.setupGroundTruthBuilder()
        
    def setupGroundTruthBuilder(self, refreshImageList=True):
        self.isGroundTruthBuilderMode = self.actions.groundTruthBuilderMode.isChecked()
        self.annotator_dock.setVisible(self.isGroundTruthBuilderMode)
        if refreshImageList:
            self.refreshDirImages()
            
    def dispSettings(self):
        msg =  f'Output dir={self.output_dir}\n'
        msg += f'Input dir={self.filename}\n'
        msg += f'Last opened dir={self.lastOpenDir}\n'
        user_extns.dispMsgBox(msg)
    
    def takePicture(self):
        if self.parent_class is None:
            print('Unable to acquire image - not linked to image acquisition module')
            return
        try:
            # mc = ModelessConfirm('Remove the pointer from the tissue.  To take a picture, click the foot pedal or mouse.', title='Taking Picture', window_pos=QtCore.QPoint(100,200))
            # mc.show()
            # mc.raise_()
            # result = mc.exec_()
            msg = 'Remove the pointer from the tissue.  To take a picture, click the foot pedal or mouse.'
            msgBox = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Information,'Acquiring picture', msg)
            msgBox.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
            result = msgBox.exec()
            if result != QtWidgets.QMessageBox.Ok:
                return
            
            # The dialog box blocks processing of the model, so after closing the box,
            # give the process time to acquire another image

            demo = True
            if not demo:
                print(f'Image # before sleep={self.parent_class.image_num}')
                time.sleep(1)
                print(f'Image # after sleep={self.parent_class.image_num}')
                image_from_camera = self.parent_class.image_from_camera
                if image_from_camera is None:
                    self.errorMessage("Error acquiring image", "Image from acquisition moudle is not available.  Please contact your administrator.")                
                    return
                img = Image.fromarray(image_from_camera)
                outfile = self.acquired_image
                # Write file
                img.save(outfile,'JPEG')
            else:
                outfile = self.acquired_image
                shutil.copyfile(r'c:\tmp\work1\20200211-143020-Img.bmp', outfile)
            self.loadFile(outfile)
            self.toggleDrawMode(False, createMode='polygon')
            #self.createMode()
        except Exception as e:
            print(traceback.print_exc())
            print(f'Error acquiring image.  Error={e}.')


    def del_cur_image(self):
        if osp.exists(self.acquired_image):
            os.remove(self.acquired_image)
        
        
    def disp_to_log(self,msg, disp_time=True):
        cursor = QtGui.QTextCursor(self.procStatusLog.document())
        cursor.setPosition(0)
        self.procStatusLog.setTextCursor(cursor)
        if disp_time:
            disp_msg = f'{datetime.datetime.now():%H:%M:%S}: {msg}'
        else:
            disp_msg = msg
        disp_msg += '\n'
        self.procStatusLog.insertPlainText(disp_msg)
        app = QtWidgets.QApplication.instance()
        app.processEvents()
        
    

    def nest(self):
        #msgBox = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Information, 'Nesting', 'Nesting not yet implemented')
        #result = msgBox.exec()
        #msgBox.exec()
        
        # TODO More precise way to clean out files.  Do by lot, not at each nesting step.
        # Clean out old files
        for file in glob.glob(osp.join(self.output_dir,'*')):
            if file == self.acquired_image or (not self.labelFile is None and file == self.labelFile.filename):
                continue
            os.remove(file)
            
        err_msg = None
        #TODO Find a better way of exiting a block without executing all of it.  Currently using a 'while' loop with a break at the end.
        while not err_msg:
            self.disp_to_log('** Nesting begins **')
            self.disp_to_log('Getting tissue boundary', False)
            num_found = 0
            if not self.labelList is None:
                for item in self.labelList:
                    if item.shape() is None:
                        continue
                    if item.shape().label == TISSUE_BOUNDARY_LABEL:
                        num_found += 1
            if num_found == 0:
                err_msg = 'Automatic identification of tissue boundary not yet implemented.  Please define manually (label={TISSUE_BOUNDARY_LABEL}).'
            elif num_found > 1:
                err_msg = f'Only one tissue boundary can be defined (label={TISSUE_BOUNDARY_LABEL}).  Found {num_found}.'
            if err_msg:
                break
            
            # TODO Implemenet real cutting requirements
            self.cutting_req_file = r'c:\tmp\work1\cutting_requirements.xlsx'
            self.disp_to_log(f'Getting cutting requirements from {self.cutting_req_file}', False)
            if not osp.exists(self.cutting_req_file):
                err_msg = f'Unable to find cutting requirements in {self.cutting_req_file}'
                break
            # TODO Handle import errors, especially missing values, and errors in general
            self.cutting_req_df = pd.read_excel(self.cutting_req_file)
            self.cutting_req_df.astype({'Quantity':int,'Priority':int}, copy=False)
            
            self.boundary_to_export = [item.shape() for item in self.labelList if not item.shape() is None and item.shape().label == TISSUE_BOUNDARY_LABEL]
            self.shapes_to_export = [item.shape() for item in self.labelList if not item.shape() is None and item.shape().label != TISSUE_BOUNDARY_LABEL]

            approach_str = str(self.selNestingApproach.currentText())
            if approach_str == 'NestFab':
                self.nest_nestFab()
            elif approach_str == 'PowerNest':
                self.nest_powerNest()
            else:
                err_msg = f'Approach {approach_str} not implemented'
                break
            
            # If we get to here, we've successfully executed all conditions, so just leave the loop
            break
            
        if err_msg:
            #TODO More elegant method of displaying error messages
            msgBox = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Information, 'Nesting', err_msg)
            result = msgBox.exec()
            self.disp_to_log(f'Error:  {err_msg}', False)
        self.disp_to_log('** Nesting ends **')

    def nest_nestFab(self):
        
        # TODO ** Calibrate DXF sheet
        # TODO * Support circles

        self.disp_to_log('Processing tissue portion of image', False)
        # --------------------------------------------------------------------
        # Create sheet - color = defect/not tissue.  White = good tissue
        
        # Make an empty image, same size as original.  All pixels black.
        overlay_over_image = False # Set to True for debugging
        cur_pixmap_size = self.canvas.pixmap.size()
        if overlay_over_image:
            pixmap = self.canvas.pixmap.copy() #QtGui.QPixmap()
        else:
            pixmap = QtGui.QPixmap(cur_pixmap_size.width(), cur_pixmap_size.height())
        
        p = QtGui.QPainter(pixmap)
        
        # Set all pixels inside tissue boundary to white
        self.paint_object(p, s_obj=self.boundary_to_export[0], color='white')
        img = pixmap.toImage()
        #img = img.convertToFormat(img.Format_Mono)
        #img.invertPixels()
        input_tissue_img = osp.join(self.output_dir,'input_tissue.png')
        img.save(input_tissue_img)
        #pixmap.convertFromImage(img)
        
        # Set all pixels within each defect to black
        self.shapes_to_export = [item.shape() for item in self.labelList if not item.shape() is None and item.shape().label != TISSUE_BOUNDARY_LABEL]
        self.disp_to_log(f'Excluding {len(self.shapes_to_export)} defects', False)
        for s in self.shapes_to_export:
            self.paint_object(p, s_obj=s, color='black')
        img_to_export = pixmap.toImage()
        img_to_export = img_to_export.convertToFormat(img.Format_Mono)
        input_sheet_img = osp.join(self.output_dir,'input_sheet.png')
        img_to_export.save(input_sheet_img)
        p.end()

        # Create DXF
        input_sheet_dxf = osp.join(self.output_dir,'input_sheet.dxf')
        doc = ezdxf.new('R2000')  # image requires the DXF R2000 format or later
        msp = doc.modelspace()  # adding entities to the model space
        
        # Not sure if it matters whether we draw the tissue boundary first, but do it just to make sure
        # TODO Support shapes other than polygons -- e.g. circles
        for s in self.boundary_to_export + self.shapes_to_export:
            points_uom = [self.point_pixels_to_uom(pt) for pt in s.points]
            points = []
            # From labelme/shape.py
            if s.shape_type == 'rectangle':
                assert len(points_uom) in [1, 2]
                if len(points_uom) == 2:
                    #rectangle = s.getRectFromLine(*s.points)
                    pass
            elif s.shape_type == "circle":
                assert len(points_uom) in [1, 2]
                if len(points_uom) == 2:
                    #rectangle = s.getCircleRectFromLine(points_uom)
                    #line_path.addEllipse(rectangle)
                    pass
            elif s.shape_type == "linestrip":
                #for i, p in enumerate(points_uom):
                #    line_path.lineTo(p)
                #    s.drawVertex(vrtx_path, i)
                pass
            # Polygon?
            else:
                #self.logger.debug(f'points_uom = {points_uom}')
                #self.logger.info(f'points_uom = {points_uom}')                
                for i, pt in enumerate(points_uom):
                    points += [(pt[0],pt[1])]
                # Close the polygon
                points += [points[0]]
                msp.add_lwpolyline(points)

        doc.saveas(input_sheet_dxf)
        
        self.disp_to_log(f'Exported sheet to {input_sheet_dxf}', False)

        # Read cutting requirements
        # TODO Handle contours and non-rectangular shapes.  Remove filter on Type = File
        # TODO Ensure self.cutting_req_df is set
        self.disp_to_log(f'Preparing cutting requirements for nesting', False)
        df_targ = self.cutting_req_df[self.cutting_req_df['Type'] != 'File'].copy()
        
        df_targ['Rotation'] =  -1
        df_targ['Tilt'] = 0
        df_targ['Mirror'] = 0
        df_targ['IsSheet'] = 0
        
        # Convert from cm to mm
        df_targ['Value1'] = df_targ['Value1'] * 10
        df_targ['Value2'] = df_targ['Value2'] * 10

        # Enforce data types
        df_targ = df_targ.astype({'Quantity':int,
                                  'Priority':int,
                                  'Rotation':int,
                                  'Tilt':int,
                                  'Mirror':int,
                                  'IsSheet':int})

        # Append sheet (IsSheet = 1)
        # TODO:  Set lot and piece #, or change Name
        new_key = len(df_targ)
        # Using df_targ[new_key,'Field name'] = value messes up the data frame data types, 
        # as when you add the row, the integer columns become NA, which make the values float/object
        df_targ.loc[new_key] = {'Type':'File',
                                'Name':'Lot NHxxx Piece x',
                                'Quantity':1,
                                'Priority':1,
                                'Rotation':-1,
                                'Tilt':0,
                                'Mirror':0,
                                'Value1':input_sheet_dxf,
                                'Value2':'',
                                'IsSheet':1}
        input_csv = osp.join(self.output_dir,'cutting_inputs_nestfab.csv')
        df_targ.to_csv(input_csv, index=False)
        self.disp_to_log(f'Performing nesting computations', False)

        # Call nesting


    def nest_powerNest(self):
        
        disp_data = True
        def log_data(msg):
            if disp_data:
                print(msg)
        
        # https://stackoverflow.com/questions/15109548/set-pythonpath-before-import-statements/15109660
        sys.path.append(r"C:\Users\mherzo\Box Sync\Herzog_Michael - Personal Folder\2020\Machine Vision Misc\Nesting\PowerNest\PowerNestCloud\python")
        
        import powernest as pn
        from powernest import Point, Orientation
        
        session = pn.CreateSession()
        orientation = pn.CreateFreeOrientation(0)
        
        #pn.time_to_run = 10

        # part_data = np.array([['Rectangle', '16x20', 100, 1, 20.0, 16.0],
        #                       ['Rectangle', '8x16', 100, 2, 16.0, 8.0],
        #                       ['Rectangle', '4x7', 100, 3, 7.0, 4.0],
        #                       ['Rectangle', '1x2', 100, 4, 2.0, 1.0],
        #                       ['File', 'ContourLarge', 0, 99, 'File name', None]])
        #part_data = np.array([['Rectangle', '3x5', 100, 3, 5, 1]], dtype=object)
        #df_parts = pd.DataFrame(part_data, 
        #                        columns=['Type', 'Name', 'Quantity', 'Priority', 'Value1', 'Value2'])
        #df_parts.astype({'Quantity':int,'Priority':int})
        
        #shapeList = [(2,2), (4,7), (4,8), (8,16), (16,20)]
        #shapes = [AddRectangle(*pt) for pt in shapeList]
        partList = []
        log_data(f"Part,Type,Length,Width,Qty,Priority")
        for df_idx in self.cutting_req_df.index:
            row = self.cutting_req_df.loc[df_idx]
            if row['Type'] != 'Rectangle':
                continue
            shape = pn.AddRectangle(session, float(row['Value1']),float(row['Value2'])) 
        
            # TODO Use AddSeveralParts
            # TODO Limit parts to a max cm2
            for part_instance in range(min(row['Quantity'],10)):
                part = pn.AddPart(session, shape,[orientation])
                log_data(f"Part,{row['Type']},{float(row['Value1'])},{float(row['Value2'])},{row['Quantity']},{row['Priority']}")
                # TODO Check error for adding part
                error_code = pn.PartSetPriority(session, part, row['Priority'])
                if error_code:
                    self.disp_to_log(f'Error setting part priority.  Name {row["Name"]}, instance {part_instance}, priority {row["Priority"]}.', False)
                    #TODO Need more robust way to handle error
                    continue
                partList += [part]

        # Build sheet
        log_data(f"Sheet,Type,Point List")
        pointList = []
        for s in self.boundary_to_export:
            points_uom = [self.point_pixels_to_uom(pt) for pt in s.points]
            # From labelme/shape.py
            if s.shape_type == 'rectangle':
                assert len(points_uom) in [1, 2]
                if len(points_uom) == 2:
                    #rectangle = s.getRectFromLine(*s.points)
                    pass
            elif s.shape_type == "circle":
                assert len(points_uom) in [1, 2]
                if len(points_uom) == 2:
                    #rectangle = s.getCircleRectFromLine(points_uom)
                    #line_path.addEllipse(rectangle)
                    pass
            elif s.shape_type == "linestrip":
                #for i, p in enumerate(points_uom):
                #    line_path.lineTo(p)
                #    s.drawVertex(vrtx_path, i)
                pass
            # Polygon?
            else:
                #self.logger.debug(f'points_uom = {points_uom}')
                #self.logger.info(f'points_uom = {points_uom}')                
                for i, pt in enumerate(points_uom):
                    pointList += [Point(pt[0],pt[1])]

        curSheet = pn.AddPolygonalSheet(session, pointList)
        log_data(f"Sheet,Boundary,'{[(pt.x, pt.y) for pt in pointList]}'")
        
        # Add defects to sheet
        # TODO Combine this code with the one to build a sheet
        for s in self.shapes_to_export:
            points_uom = [self.point_pixels_to_uom(pt) for pt in s.points]
            defectPoints = []
            # From labelme/shape.py
            if s.shape_type == 'rectangle':
                assert len(points_uom) in [1, 2]
                if len(points_uom) == 2:
                    #rectangle = s.getRectFromLine(*s.points)
                    pass
            elif s.shape_type == "circle":
                assert len(points_uom) in [1, 2]
                if len(points_uom) == 2:
                    #rectangle = s.getCircleRectFromLine(points_uom)
                    #line_path.addEllipse(rectangle)
                    pass
            elif s.shape_type == "linestrip":
                #for i, p in enumerate(points_uom):
                #    line_path.lineTo(p)
                #    s.drawVertex(vrtx_path, i)
                pass
            # Polygon?
            else:
                #self.logger.debug(f'points_uom = {points_uom}')
                #self.logger.info(f'points_uom = {points_uom}')                
                for i, pt in enumerate(points_uom):
                    defectPoints += [Point(pt[0],pt[1])]
                pn.SheetAddDefect(session, curSheet, defectPoints);
                log_data(f"Sheet,Defect,'{[(pt.x, pt.y) for pt in defectPoints]}'")
        
        sheetList = [curSheet]
        sheetQuantityList = [1]
        
        self.disp_to_log(f'Starting nesting with requested time {pn.time_to_run}s.  # parts={len(partList)}.')
        try:
            multi_result = None
            errorMessage = None
            outputId = None
            #TODO Run request to nesting software asynchronously to check on status
            #TODO Gracefully handle condition if no part can fit in sheet
            # multiResult, errorMessage, outputId = pn.HttpMultiNest(session, 
            #                             sheetList, 
            #                             sheetQuantityList, 
            #                             partList, 
            #                             pn.time_to_run,
            #                             pn.powernest_server, pn.powernest_port, 
            #                             pn.powernest_login)
            multiResult, errorMessage, outputId = pn.MultiNest(session, 
                                        sheetList, 
                                        sheetQuantityList, 
                                        partList, 
                                        pn.time_to_run)
        except Exception as e:
            self.disp_to_log(f'Error getting nesting result.  Traceback:  {traceback.print_exc()}')
        if not (errorMessage is None) and (len(errorMessage) > 0):
            errorNesting = True
            self.disp_to_log(f'multiResult Error {errorMessage}, id={outputId}')
        else:
            errorNesting = False
            self.disp_to_log(f'Received nesting results, id={outputId}')

        svg_path = os.path.join(os.path.dirname(__file__), 'NestingResult.svg')
        if os.path.exists(svg_path):
            os.remove(svg_path)
        if not errorNesting:
            error_code = pn.DrawMultiSvg(session, multiResult, svg_path)
            if not error_code:
                self.disp_to_log(f'Nesting result image written to {svg_path}', False)
            else:
                self.disp_to_log(f'Error {pn.ErrorMessage(error_code)}', False)
                
            # Launch viewer
            # TODO render nested products in main LabelMe screen
            self.disp_to_log(f'Viewer launched.  File={svg_path}',False)
            output = subprocess.Popen(args=svg_path,
                                      shell=True)
            # error_code = os.system(svg_path)
            # if not error_code:
            #     self.disp_to_log(f'Viewer launched.  File={svg_path}',False)
            # else:
            #     #TODO Get error description
            #     self.disp_to_log(f'Error launching viewer.  Code={error_code}.', False)
        
        #TODO Use Try-Except to clean up crashes
        bool_status = pn.DeleteSession(session)
        if bool_status <= 0:
            self.disp_to_log(f'Error:  processing complete, but unable to delete session.  Error status={bool_status}.')
        else:
            self.disp_to_log(f'Session ended.  Return code={bool_status}')        
                

    def simulate_click(self):
        #https://stackoverflow.com/questions/33319485/how-to-simulate-a-mouse-click-without-interfering-with-actual-mouse-in-python
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0)        


    def pixels_to_uom(self, pixels, dimension):
        if dimension not in self.calib_factors:
            self.logger.error(f'In pixels_to_uom.  Dimension = {dimension} not in {self.calib_factors}')
            return
        return pixels / self.calib_factors[dimension]
    
    def point_pixels_to_uom(self, point_pixels):
        point_uom = []
        dimensions = self.calib_factors.keys()
        if hasattr(point_pixels, 'x') and hasattr(point_pixels,'y'):
            in_vals = [point_pixels.x(), point_pixels.y()]
        else:
            in_vals = point_pixels
        for value, dimension in zip(in_vals, dimensions):
            point_uom += [self.pixels_to_uom(value, dimension)]
        return point_uom
