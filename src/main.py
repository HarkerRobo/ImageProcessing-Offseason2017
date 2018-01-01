import logging
import time
import threading

import cv2
import gstreamer as gs
from processing import VideoCapture, process
from processing.process import DEBUG
import networking
import networking.messages as m
Gst = gs.Gst

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
logger.addHandler(sh)

gs.delete_socket()

print(
# gs.RaspiCam(vflip=True, hFlip=True, expmode=6, framerate=30, ec=10,
    #             awb=False, ar=1, ab=2.5, width=640, height=480) +
    gs.Webcam(device='/dev/video2', width=320, height=240, framerate=30) +
    gs.PipelinePart('videoconvert') +
    # gs.Tee('t',
    #        gs.Valve('valve') + gs.H264Video() + gs.H264Stream(),

    # gs.PipelinePart('videoscale ! video/x-raw, width=320, height=240') +
    gs.SHMSink()
    # )
)

pipeline = gs.pipeline(
    # gs.RaspiCam(vflip=True, hFlip=True, expmode=6, framerate=30, ec=10,
    #             awb=False, ar=1, ab=2.5, width=640, height=480) +
    gs.Webcam(device='/dev/video2', width=320, height=240, framerate=30) +
    gs.PipelinePart('videoconvert') +
    # gs.Tee('t',
    #        gs.Valve('valve') + gs.H264Video() + gs.H264Stream(),
    # gs.PipelinePart('videoscale ! video/x-raw, width=320, height=240') +
    gs.SHMSink()
    # )
)


# Start debugging the gstreamer pipeline
debuggingThread = gs.MessagePrinter(pipeline)
debuggingThread.start()

pipeline.set_state(Gst.State.PLAYING)

logger.debug(pipeline.get_state(Gst.CLOCK_TIME_NONE))

caps = gs.get_sink_caps(pipeline.get_by_name(gs.SINK_NAME))
cap_string = gs.make_command_line_parsable(caps)

vc = VideoCapture(gs.SHMSrc(cap_string))
vc.setDaemon(True)
vc.start()

# Now that the capture filters have been (hopefully) successfully
# captured, GStreamer doesn't need to be debugged anymore and the thread
# can be stopped.
debuggingThread.stop()

sock, clis = networking.server.create_socket_and_client_list(port=6000)
gsthandler = networking.create_gst_handler(pipeline, gs.SRC_NAME, None, #'valve',
                                           None) #gs.UDP_NAME)
cntset, cnthandler = networking.create_cnt_handler(gsthandler)

acceptThread = threading.Thread(target=networking.server.AcceptClients,
                                args=[sock, clis, cnthandler])
acceptThread.daemon = True # Makes the thread quit with the current thread
acceptThread.start()

frames = 0
start = time.time()

while True:
    stat, img = vc.read()
    # print(pipeline.get_by_name('queue0').get_property('current-level-time') /1e9)
    # print(pipeline.get_by_name('queue1').get_property('current-level-time') /1e9)
    if stat:
        valid, bb = process(img)
        smessage = m.create_message(m.TYPE_RESULTS, {m.FIELD_CORNERS: bb})
        networking.server.broadcast(sock, clis, smessage, lambda c: c not in cntset)

        lmessage = m.create_message(m.TYPE_RESULTS, {m.FIELD_CORNERS: bb, 'valid': [x.array.tolist() for x in valid]})
        networking.server.broadcast(sock, clis, lmessage, lambda c: c in cntset)

        if DEBUG:
            cv2.imshow('frame', img)
        frames += 1
        print('FPS: {}'.format(frames / (time.time()-start)))
    else:
        print('No image')
    if DEBUG and cv2.waitKey(1) == ord('q'):
        break
