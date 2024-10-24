import numpy as np
from ALP4 import *
import time
from PyQt5.QtCore import QThread, pyqtSignal
import cv2
from scipy.ndimage import rotate

from loguru import logger
import torch
import sys


def run_dmd_trigger(DMD):
    bitDepth = 1
    imgBlack = np.zeros([DMD.nSizeY, DMD.nSizeX])
    imgWhite = np.ones([DMD.nSizeY, DMD.nSizeX]) * (2**8 - 1)
    imgSeq = [imgBlack.ravel(), imgWhite.ravel()] * 800

    DMD.ProjControl(controlType=ALP_PROJ_MODE, value=ALP_SLAVE)
    DMD.DevControl(controlType=ALP_TRIGGER_EDGE, value=ALP_EDGE_FALLING)
    # DMD.ProjControl(controlType=ALP_PROJ_STEP, value=ALP_LEVEL_HIGH)
    DMD.SeqAlloc(nbImg=1, bitDepth=bitDepth)
    DMD.SetTiming(pictureTime=2000)

    # Load and run the sequence
    for i in range(1600):
        DMD.SeqPutEx(imgData=imgSeq[i], LineOffset=0, LineLoad=0)
        DMD.Run(loop=False)
        DMD.Wait()
        logger.info(f"Sequence {i} displayed")


def initialize_dmd():
    DMD = ALP4(version="4.3")
    DMD.Initialize()
    return DMD


def run_loop(DMD):
    bitDepth = 8
    imgBlack = np.zeros([DMD.nSizeY, DMD.nSizeX])

    imgWhite = np.ones([DMD.nSizeY, DMD.nSizeX]) * (2**8 - 1)
    imgSeq = np.concatenate([imgBlack.ravel(), imgWhite.ravel()])

    DMD.SeqAlloc(nbImg=2, bitDepth=bitDepth)
    DMD.SeqPut(imgData=imgSeq)
    DMD.SetTiming(pictureTime=2000000, synchPulseWidth=1500000)
    DMD.Run()
    time.sleep(1000)


def run_a_image(image):
    logger.info(f"Initializing DMD")
    DMD = ALP4(version="4.3")
    DMD.Initialize()
    bitDepth = 1
    if isinstance(image, torch.Tensor):
        image = ((image * 255).cpu().detach().numpy()).astype(np.uint8)
        # print('[DMD]',np.unique(image))
    else:
        image = (image > 100).astype(np.uint8)
        image = (image * 255).astype(np.uint8)
        # print(np.unique(image))

    # pad the image to[DMD.nSizeY,DMD.nSizeX]
    pad = int((DMD.nSizeY - image.shape[0]) / 2)
    pad1 = int((DMD.nSizeX - image.shape[0]) / 2)

    image = np.pad(
        image, ((pad, pad), (pad1, pad1)), "constant", constant_values=(0, 0)
    )

    rotated_image = rotate(image, -45, reshape=False)

    cv2.imwrite("img/img.png", rotated_image)
    imgSeq = rotated_image.ravel()

    DMD.SeqAlloc(nbImg=1, bitDepth=bitDepth)
    DMD.SeqPut(imgData=imgSeq)
    DMD.SetTiming(pictureTime=200000)

    # Run the sequence in an infinite loop
    DMD.Run()

    return DMD


def run_a_image_event(image, e):
    logger.info(f"Initializing DMD")
    DMD = ALP4(version="4.3")
    DMD.Initialize()
    bitDepth = 1

    image = (image * 255).astype(np.uint8)
    # print('[DMD]',np.unique(image))
    # print(np.unique(image))

    # pad the image to[DMD.nSizeY,DMD.nSizeX]
    pad = int((DMD.nSizeY - image.shape[0]) / 2)
    pad1 = int((DMD.nSizeX - image.shape[0]) / 2)

    print(image.shape)
    image = np.pad(
        image, ((pad, pad), (pad1, pad1)), "constant", constant_values=(0, 0)
    )

    rotated_image = rotate(image, -45, reshape=False)

    cv2.imwrite("img/error/dmd_img.png", rotated_image)
    img = rotated_image.ravel()

    # create a dark image
    imgBlack = np.zeros([DMD.nSizeY, DMD.nSizeX])
    # concatenate the image with the dark image
    imgSeq = np.concatenate([imgBlack.ravel(), img])

    DMD.SeqAlloc(nbImg=2, bitDepth=bitDepth)
    e.wait()

    DMD.SeqPut(imgData=imgSeq)
    DMD.SetTiming(pictureTime=500000)

    DMD.Run(loop=True)
    time.sleep(0.8)

    DMD.Halt()
    DMD.FreeSeq()
    DMD.Free()


def run_a_image_event_batch(imglist, batch_Size, e):
    logger.info(f"Initializing DMD")
    DMD = ALP4(version="4.3")
    DMD.Initialize()
    bitDepth = 1

    DMD.SeqAlloc(nbImg=batch_Size * 2, bitDepth=bitDepth)
    e.wait()

    DMD.SeqPut(imgData=imglist)
    DMD.SetTiming(pictureTime=200)

    # Run the sequence in an infinite loop
    time_sleep = batch_Size * 0.02 + 0.05
    DMD.Run(loop=False)
    time.sleep(time_sleep)
    logger.info(f"Image sequence displayed")
    e.clear()

    DMD.Halt()
    DMD.FreeSeq()
    DMD.Free()


class DMD_Trigger_Thread(QThread):
    errorOccurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.DMD = None

    def run(self):
        try:
            self.DMD = initialize_dmd()
            run_dmd_trigger(self.DMD)
        except Exception as e:
            print(f"Error occurred: {e}")
            self.errorOccurred.emit(str(e))

    def stop(self):
        logger.info("DMD thread stopped")
        if self.DMD is not None:
            self.DMD.Halt()
            self.DMD.FreeSeq()
            self.DMD.Free()
            self.terminate()


class DMD_thread(QThread):
    errorOccurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.DMD = None

    def run(self):
        try:
            self.DMD = initialize_dmd()
            run_loop(self.DMD)
        except Exception as e:
            logger.error(f"Error occurred: {e}")
            self.errorOccurred.emit(str(e))

    def stop(self):
        print("DMD thread stopped")
        if self.DMD is not None:
            self.DMD.Halt()
            self.DMD.FreeSeq()
            self.DMD.Free()
            self.terminate()


if __name__ == "__main__":
    DMD = initialize_dmd()
    # run_loop(self.DMD)
    run_dmd_trigger(DMD)
    DMD.Halt()
    DMD.FreeSeq()
    DMD.Free()
