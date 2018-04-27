import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')
import random
from multiprocessing import Process, Queue
import copy
import math
from time import time as timer
from time import sleep
import time
import cv2
import subprocess
from mymodules import scenario




#======= Select mode ==============
#RESULT_FILENAME = "FREE"
#RESULT_FILENAME = "RESULT"
#RESULT_FILENAME = "RESULT2"
RESULT_FILENAME = "RESULT_static_opt"
#RESULT_FILENAME = "RESULT_static_slow"
#RESULT_FILENAME = "RESULT_static_fast"
#=========================

#======global variables =====================
stress =0
currentOption = 5# meaning start mode
useScenario = True


TIMECONSTRAINT = 0.6 if RESULT_FILENAME != "RESULT2" else 0.3
if RESULT_FILENAME == "RESULT" or RESULT_FILENAME ==  "RESULT2" : currentOption = 0
elif RESULT_FILENAME == "RESULT_static_opt" : currentOption = 4
elif RESULT_FILENAME == "RESULT_static_slow" : currentOption = 3
elif RESULT_FILENAME == "RESULT_static_fast" : currentOption = 5
elif RESULT_FILENAME == "FREE" : stress = 0 ; currentOption = 0 ; useScenario = False


FRAMERATE = 0 #it is just constant
STARTTIME = time.time() #program start time
#===========================================


#========help codes================================
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
#================================================


#==============make models : load graph and sessions==================
def loadGraphAndSession(MODEL_NUMBER,MODEL_NAME, SPEED, mAP) :
    PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    detection_graph.as_default()
    # detection_graph_rfcn_resnet101_coco_11_06_2017.as_default()
    sess = tf.Session(graph=detection_graph)

    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')


    #pre running
    image_np = np.zeros((300,300,3),np.uint8)
    image_np_expanded = np.expand_dims(image_np, axis=0)
    sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    dict = {'detection_graph': detection_graph, 'sess' : sess, 'image_tensor' : image_tensor ,'detection_boxes' : detection_boxes , 'detection_scores' : detection_scores , 'detection_classes' : detection_classes ,       'num_detections' : num_detections, 'modelname' : MODEL_NAME , 'modelnumber':MODEL_NUMBER,  'speed' : SPEED , 'mAP' : mAP}
    return dict
#=====================================================


#=============define help classes=====================
class FrameAndStamps :
    def __init__(self,frame_ , framenumber_,timestamp_):
        self.framenumber = framenumber_
        self.frame = frame_
        self.timestamp = timestamp_

    def getTimestamp(self):
        return self.timestamp

    def getFrame(self):
        return self.frame

    def getFramenumber(self):
        return self.framenumber
#==========================================================




'''*********************************************************************************************************************************************************************************
 * def processFunc_framePutter(frameQueue, controlQueue):
*********************************************************************************************************************************************************************************'''
def processFunc_framePutter(frameQ, processControlQ,framerateControlQ,useScenario):

    #=========================================================================================================
    STARTTIME = time.time()
    framerate = FRAMERATE
    camera_framePutter = cv2.VideoCapture(0)
    _, frame = camera_framePutter.read()
    inputrate = 0
    lastAcceptedTimeChecker = 0
    isScenario = useScenario
    myscenario = scenario.Scenario(STARTTIME)
    cv2.namedWindow('FramePutter', 0)
    #cv2.resizeWindow('FramePutter', 500, 600)
    #=========================================================================================================



    # ===========opencv trackbar controllers==================================================================
    def framerateController(x):
        nonlocal framerate
        nonlocal framerateControlQ
        framerate = x
        framerateControlQ.put(x)
        pass
    #----------------------------------------------------------------------
    cv2.createTrackbar('frame rate', 'FramePutter', 0, 50, framerateController)
    #=========================================================================================================




    # ========================================================================================================
    while processControlQ.qsize() == 0 and (isScenario is False or myscenario.isEnd() is False):

        # ===== expieriment scenario ======================================
        if(isScenario is True):
            myscenario.playScenario()
            if myscenario.getCurrentFramerate() != framerate :
                framerate = myscenario.getCurrentFramerate()

        # =================================================================

        #----------------------------------------------------------------------
        period = 1 if framerate == 0 else 1.0/framerate #framerate 0 means waiting 1sec
        if time.time() - lastAcceptedTimeChecker >= period :

            # take a photo. then stamp useful infomations.
            # ----------------------------------------------------------------------
            _, frame = camera_framePutter.read()

            timestamp = time.time()
            frame_copied = copy.deepcopy(frame)
            framenumber = 0
            frameAndStamps = FrameAndStamps(frame_copied, framenumber, timestamp)

            # ----------------------------------------------------------------------
            # coping frame. then put into waitQ
            # ----------------------------------------------------------------------
            frameQ.put(frameAndStamps)
            inputrate = 1.0 / (time.time() - lastAcceptedTimeChecker)
            framerateControlQ.put(inputrate)
            # stamp additional informations on currently input frame.
            # ----------------------------------------------------------------------
            #cv2.putText(frame, "Frame number : {} ".format(framenumber),   (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            cv2.putText(frame, "total elapsed time : {} sec".format(round(time.time() - STARTTIME), 4),
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            cv2.putText(frame, "Current Input Rate: {} Q len: {}".format(round(1.0/(time.time() - lastAcceptedTimeChecker), 4), frameQ.qsize()),
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            cv2.imshow('FramePutter', frame)
            # ----------------------------------------------------------------------

            lastAcceptedTimeChecker = time.time()
            pass

        else :
            sleep(0.00001)
            pass
        # ----------------------------------------------------------------------


        # quit the program on the press of key 'w'
        #----------------------------------------------------------------------
        if cv2.waitKey(1) & 0xFF == ord('w'):
            break
        #----------------------------------------------------------------------

    #===========  end loop  =================================================================================




    # == clean up process ================================================================================
    camera_framePutter.release()
    cv2.destroyAllWindows()
    # ==================================================================================================

'''*********************************************************************************************************************************************************************************
 * end function
*********************************************************************************************************************************************************************************'''





#=================load graph and lables==========================================
CWD_PATH = os.getcwd()
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')

# create models
MODEL_ssd_mobilenet_v1_coco_11_06_2017 = loadGraphAndSession('Model5','ssd_mobilenet_v1_coco_11_06_2017', 40,21)    #processtime: 30 COCO mAP: 21
MODEL_ssd_inception_v2_coco_11_06_2017=  loadGraphAndSession('Model4','ssd_inception_v2_coco_11_06_2017', 60,24)    #processtime: 42 COCO mAP: 24
MODEL_faster_rcnn_inception_v2_coco = loadGraphAndSession('Model3','faster_rcnn_inception_v2_coco', 126,28)          #processtime: 58 COCO mAP: 28
MODEL_faster_rcnn_resnet50_coco = loadGraphAndSession('Model2','faster_rcnn_resnet50_coco', 215,30)                  #processtime: 89 COCO mAP: 30
MODEL_faster_rcnn_resnet101_coco = loadGraphAndSession('Model1','faster_rcnn_resnet101_coco', 235,32)               #processtime: 106COCO mAP: 32

#decrement order by mAP
MODEL_LIST = [MODEL_faster_rcnn_resnet101_coco , MODEL_faster_rcnn_resnet50_coco , MODEL_faster_rcnn_inception_v2_coco, MODEL_ssd_inception_v2_coco_11_06_2017, MODEL_ssd_mobilenet_v1_coco_11_06_2017]


# Loading label map
NUM_CLASSES = 90
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,use_display_name=True)
category_index = label_map_util.create_category_index(categories)
#==========================================================================


#=========configurations==========================
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)
#================================================





'''*********************************************************************************************************************************************************************************
 * MAIN PROCEDURE
*********************************************************************************************************************************************************************************'''
# ==============main===================================
if __name__ == "__main__" :

    # ==========ready for main loop===================

    frameQ = Queue()
    processControlQ = Queue()
    framerateControlQ = Queue()
    process_framePutter = Process(target=processFunc_framePutter, args=(frameQ, processControlQ, framerateControlQ,useScenario ))
    process_framePutter.start()

    processingTime = 0.1
    ffelapsedTime = 0.05 #feed forwarding time
    MODEL = MODEL_LIST[0] #current model
    lastChanged = 0

    timelimit = TIMECONSTRAINT #sec
    QMax = 20

    framerate = 15
    V = 0.01
    expOutputRate =0
    expInputRate = 0

    #result
    fps = 0
    cumulatedTimeaveragePerformance = 0
    timestamp = 0
    delay = 0

    f = open(RESULT_FILENAME, 'w')
    f.close()
    f = open(RESULT_FILENAME, 'a')

    STARTTIME = time.time()
    cv2.namedWindow('object detection', 0)
    # ======================================================


    # ===========opencv trackbar controllers=================
    def stressController(x):
        global stress
        stress = x
        pass


    def optionController(x):
        global currentOption
        currentOption = x
        pass
    # ======================================================
    cv2.createTrackbar('stress','object detection', 0, 255, stressController)
    cv2.createTrackbar('currentOptions', 'object detection', 0, len(MODEL_LIST)+1, optionController) #0 for auto, last for nonprocess
    # ================================================




    # ==cal drift func ==================
    def calDrift(M,M_CUR,QSIZE) :
        drift = 0

        # ========cal variables related with model i ==================
        # model i's and model last's expected ouput rate
        nonfftime = processingTime - ffelapsedTime

        expPTime_M = ( nonfftime + ffelapsedTime * (M['speed'] / M_CUR['speed']))  # expected processing time when using model M
        expOutputRate_M = 1.0 / expPTime_M  # expected outputRate when using model i
        expExhautionRate_M = (1 - (framerate - expOutputRate_M) / expOutputRate_M)
        # ===============================================================


        #=======cal Qmax=============
        QMax = int(timelimit/(1/expOutputRate_M))
        if(QSIZE == -1) : QSIZE = QMax
        #============================


        # ========cal drift =========================
        rewardTerm = M['mAP'] * min(framerate if not framerate == 0 else 0.00001 ,expOutputRate_M ) if expOutputRate_M> (1/timelimit) else 0
        backlogTerm = QSIZE * expOutputRate_M

        drift = V * rewardTerm  + backlogTerm
        drift_dict = {'drift' : drift, 'rewardTerm':rewardTerm, 'backlogTerm':backlogTerm}
        #===================================

        return drift_dict
    #====================================


    #=================main loop=========================================
    while True:
      processingTimeChecker = time.time()  # -- timer

      #============================load frame from buffer=====================================
      if(framerateControlQ.qsize()>0) :
          while (framerateControlQ.qsize() > 1):
              if cv2.waitKey(1) & 0xFF == ord('q') or (process_framePutter.is_alive() == False and frameQ.qsize() <= 0):
                  processControlQ.put(1)
                  break
              framerateControlQ.get()

          framerate = framerateControlQ.get()

      frameAndStamps = copy.deepcopy(frameQ.get())
      image_np = frameAndStamps.getFrame()
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      #============================load frame from buffer end ================================




      # ============================== model selection  =======================================================================
      # =====================auto adapting mode ================================================
      if(currentOption == 0) :
          if(time.time() - lastChanged > 0.5): # timelimit/2) : #adaptation period

              #=========================== cal V ===========================================
              VTEMP = 9999
              temp = 0

              MODEL_last = MODEL_LIST[len(MODEL_LIST)-1]
              MODEL_last_last = MODEL_LIST[len(MODEL_LIST)-2]

              #========cal min VTEMP ====
              calDrift_I = calDrift(MODEL_last_last,MODEL,-1)
              calDrift_Last = calDrift(MODEL_last,MODEL,-1)

              temp = (calDrift_Last['backlogTerm'] - calDrift_I['backlogTerm']) / (calDrift_I['rewardTerm'] - calDrift_Last['rewardTerm']) *0.7
              if temp < VTEMP and temp >=0:
                  VTEMP = temp
              #=========================================================================
              pass

              V = VTEMP if VTEMP >0 and VTEMP<9999 else V

              #============================ cal V end  ====================================




              #============================ select optimal model ==============================
              MODELTEMP = MODEL_LIST[len(MODEL_LIST)-1]              #init model with fastest but having lowest performance to handle too fast sampling rate or too heavy stress
              maxTemp = -9999
              temp = 0
              for i in MODEL_LIST:

                  #========cal variables related with model i       ==================
                  #model i's expected ouput rate
                  nonfftime = processingTime - ffelapsedTime
                  expPTime_i =  (nonfftime + ffelapsedTime * (i['speed'] / MODEL['speed']))    #expected processing time when using model i
                  expOutputRate_i = 1.0 / expPTime_i                                       #expected outputRate when using model i
                  # ========cal variables related with model i end  ==================


                  #===============cal max drift ====================================
                  #temp = V * i's mAP  * ( 1 - (framerate- outputrate_i) / outputrate_i  ) ^ (qsize/qMax)  + qsize * outpurate_i
                  #temp = V * i['mAP'] * pow((1 - (framerate - expOutputRate_i) / expOutputRate_i), (frameQ.qsize() / QMax)) + frameQ.qsize()* expOutputRate_i
                  temp = calDrift(i,MODEL,frameQ.qsize())['drift']
                  if temp > maxTemp:
                      MODELTEMP = i
                      expOutputRate = expOutputRate_i
                      expInputRate = framerate
                      maxTemp = temp
                  #===========================================================
                  pass
              #============================ select optimal model end ==============================




              #============================ adapt model ==============================
              MODEL = MODELTEMP
              lastChange = time.time()
              #============================ adapt model end ==========================
          pass
      # =====================auto adapting mode ================================================




      # ===================== static mode ================================================
      elif currentOption < len(MODEL_LIST)+1 :
          MODEL = MODEL_LIST[currentOption-1]
          expOutputRate = 1/processingTime
          expInputRate = framerate

      else : #currentOption == len(MODEL_LIST)+1   #do not predict. this means a dummy model whose speed is max and accurate is 0.
          pass
      # ===================== static mode end  ===========================================
      # ============================== model selection end =============================================================






      # ============================== object detecting  =============================================================
      if(currentOption != len(MODEL_LIST)+1) :

          frameAndStamps = copy.deepcopy(frameQ.get())
          image_np = frameAndStamps.getFrame()
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(image_np, axis=0)

          processingTimeChecker = time.time()  # -- timer
          ffelapsedTimeChecker = time.time()

          (boxes, scores, classes, num) = MODEL['sess'].run(
              [MODEL['detection_boxes'], MODEL['detection_scores'], MODEL['detection_classes'], MODEL['num_detections']],
              feed_dict={MODEL['image_tensor']: image_np_expanded})


          # Visualization of the results of a detection.
          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=8)
      # ============================== object detecting  end ========================================================
      ffelapsedTimeTemp = time.time() - ffelapsedTimeChecker
      ffelapsedTime = ffelapsedTimeTemp
      #ffelapsedTime = 0.5 * ffelapsedTimeTemp + 0.5 * ffelapsedTime



      # ==== wait as stress ========================================================================================
      # simulate stress. Now, it is just arbitrary wait function
      mywait = random.uniform(stress - stress * 0, stress + stress * 0)
      mywait = mywait if (mywait > 0) else 0
      sleep(mywait / 122)
      # ==== wait as stress end ====================================================================================

      processingTimeTemp = time.time() - processingTimeChecker
      processingTime = processingTimeTemp
      #processingTime = 0.5*processingTimeTemp + 0.5* processingTime






      # ================== post process     ==========================================================================
      #================ cal result ========================
      #fps
      fps = 1.0 / processingTime

      #current model
      currentmodel = str("NONE") if currentOption == len(MODEL_LIST)+1 else str(MODEL['modelname'])
      currentmodelNumber = str("NONE") if currentOption == len(MODEL_LIST)+1 else str(MODEL['modelnumber'])

      #delay
      delay = time.time() - frameAndStamps.getTimestamp()

      #time stamp
      timestamp = time.time() - STARTTIME

      #time average performance
      cumulatedTimeaveragePerformance += MODEL['mAP']
      timeaveragePerformance = cumulatedTimeaveragePerformance / (time.time() - STARTTIME)

      #print log
      print(round(timestamp,3),"\t", currentmodel ,"\t",round(fps,3),"\t",round(expOutputRate,3),"\t",round(delay,3),"\t",round(timeaveragePerformance,3))
      data = \
          str(round(timestamp,3))+ "\t" + \
          currentmodelNumber + "\t"+ \
          str(round(fps,3))+ "\t" +\
          str(round(expOutputRate,3))+ "\t" + \
          str(round(delay,3))+ "\t" + \
          str(round(timeaveragePerformance,3)) + "\t"+ \
          str(round(timelimit,3)) + "\t" +\
          str(framerate) + "\n"


      f.write(data)
      #================ cal result end  ====================



      #===    post useful describtion  ======================
      cv2.putText(image_np, "Current Model:" + currentmodel                                                     , (20,  50) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
      cv2.putText(image_np, "Current Output Rate: {} /sec ".format(round(fps, 4))                               , (20, 100) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
      cv2.putText(image_np, "Expected Input Rate: {} /sec ".format(round(expInputRate, 4))                      , (20, 150) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
      cv2.putText(image_np, "Expected Output Rate: {} /sec ".format(round(expOutputRate, 4))                    , (20, 200) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

      cv2.putText(image_np, "Q len: {}".format(frameQ.qsize())                                                  , (20, 250) , cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 125, 0), 3)
      cv2.putText(image_np, "Delay: {}".format(round(delay,4))                                                  , (20, 300) , cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 125, 0), 3)
      cv2.putText(image_np, "ffelapsedTime: {}".format(round(ffelapsedTime,4))                                  , (20, 350) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

      cv2.putText(image_np, "time average performance: {}".format(round(timeaveragePerformance,4))              , (20, 400) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

      cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))
      # =================================================
      #================== post process end   ==========================================================================




      #===cv exit handler========
      if cv2.waitKey(1) & 0xFF == ord('q') or (process_framePutter.is_alive() == False and frameQ.qsize() <= 0):
        processControlQ.put(1)
        break
      #===========================
    # =================   main loop end   =========================================


    #=============clean up ===========================================
    f.close()
    cv2.destroyAllWindows()
    process_framePutter.join()
    #=======================================================================
#===================================================================================

'''*********************************************************************************************************************************************************************************
 * END MAIN
*********************************************************************************************************************************************************************************'''