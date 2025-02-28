from numpy.lib.function_base import re
from pyvisa.constants import *
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from time import sleep
import scipy.io as scio
from ALP4 import *
import cv2
import ctypes
from PIL import Image
import numpy as np

import torch
import time
import numpy as np
from metavision_core.event_io.raw_reader import initiate_device
from metavision_core.event_io import EventsIterator, RawReader
from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator
from metavision_sdk_core import (
    PeriodicFrameGenerationAlgorithm,
    ColorPalette,
    OnDemandFrameGenerationAlgorithm,
    BaseFrameGenerationAlgorithm,
)
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent

from system.utils import get_biases_from_file, get_roi_from_file, contrast_exponential
import metavision_hal
from multiprocessing import shared_memory
from PIL import Image
from scipy.ndimage import rotate
from loguru import logger
import queue
from DCAM.dcamapi4 import *
from DCAM.dcam import Dcamapi, Dcam
import threading
from multiprocessing import Process, SimpleQueue
import keyboard
from system.utils import events_to_diff_image_positive_only


class DiMD:

    def __init__(self, nbImg, ill_time=2000, pic_time=4000):
        self.ill_time = ill_time
        self.pic_time = pic_time
        self.DMD = ALP4(version="4.3")
        self.DMD.Initialize()
        self.nbImg = nbImg
        self.SeqId = self.DMD.SeqAlloc(nbImg=self.nbImg, bitDepth=1)

    def put_imgs(self, imgs, transform=False, pic_time=400, ill_time=200):
        # self.SeqId = self.DMD.SeqAlloc(nbImg=self.nbImg, bitDepth=1)
        try:
            if transform:
                imgSeq = []
                black_image = np.zeros_like(imgs[0]).ravel()
                for i in range(len(imgs)):
                    imgSeq.append(black_image)
                    imgSeq.append(self.tonumpyarray(imgs[i]))
                imgSeq = np.concatenate(imgSeq)
            else:
                imgSeq = np.concatenate([_.ravel() for _ in imgs])

            self.DMD.SeqPut(imgData=imgSeq, SequenceId=self.SeqId, PicLoad=self.nbImg)
            # print(self.DMD.SeqInquire(ALP_TRIGGER_IN_DELAY, self.SeqId))

            self.DMD.SetTiming(self.SeqId, pictureTime=pic_time, illuminationTime=ill_time)
            # self.DMD.SeqControl(ALP_BITNUM, 2, self.SeqId)
            # self.DMD.ProjControl(controlType=ALP_PROJ_MODE, value=ALP_SLAVE)
            # self.DMD.ProjControl(controlType=ALP_PROJ_STEP, value=ALP_EDGE_FALLING)

            self.DMD.Run(loop=False, SequenceId=self.SeqId)
        except Exception as e:
            print(e)

    def put_img(self, img, transform=True):
        self.reset()
        self.SeqId = self.DMD.SeqAlloc(1, bitDepth=1)
        if transform:
            imgSeq = self.tonumpyarray(img)
        else:
            imgSeq = img.ravel()
        self.DMD.SeqPut(imgData=imgSeq, SequenceId=self.SeqId)
        self.DMD.SetTiming(pictureTime=self.pic_time, illuminationTime=self.ill_time)
        # self.DMD.ProjControl(controlType=ALP_PROJ_MODE, value=ALP_MASTER)
        # self.DMD.ProjControl(controlType=ALP_PROJ_STEP, value=ALP_EDGE_FALLING)
        self.DMD.Run(loop=False, SequenceId=self.SeqId)

    def tonumpyarray(self, img):
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        img = img.astype(np.uint8)

        return img.ravel()

    def set_position(self, rx=None, ry=None):
        if rx is not None and ry is not None:
            h, w = 1600, 2560  # Adjusted to match the DMD size
            mh, mw = h // 2 + ry, w // 2 + rx
            hw1 = 541 // 2
            self.border_tblr = [
                mh - hw1,
                h - (mh + hw1) - 1,
                mw - hw1,
                w - (mw + hw1) - 1,
            ]
        print(self.border_tblr)
        tmp = np.ones((541, 541)).astype(np.uint8) * 255
        tmp = cv2.copyMakeBorder(tmp, *self.border_tblr, borderType=cv2.BORDER_CONSTANT, value=0)
        # cv2.imwrite('tmp.png', tmp)
        return tmp

    def put_circle(self, center, radius):
        self.reset()
        img = np.zeros((1600, 2560), dtype=np.uint8)  # DMD size adjusted here
        cv2.circle(img, center[0], int(radius[0]), (255, 255), thickness=-1)
        cv2.circle(img, center[1], int(radius[1]), (255, 255), thickness=-1)
        self.put_img(img, transform=False)

    def put_white(self):
        img = np.ones((1600, 2560), dtype=np.uint8) * 255  # DMD size adjusted here
        self.put_img(img, transform=False)

    def reset(self):
        self.DMD.Halt()
        if self.SeqId is not None:
            self.DMD.FreeSeq(self.SeqId)
            self.SeqId = None

    def close(self):
        self.reset()
        self.DMD.Free()


class SLM:

    def __init__(
        self,
        dll_path="lib\\hpkSLMdaLV.dll",
    ):
        # Load the DLL library
        self.lib = ctypes.windll.LoadLibrary(dll_path)
        self.bIDList = (ctypes.c_uint8 * 10)()

        # Define function prototypes
        self._define_functions()
        self.open_device()

    def _define_functions(self):
        """Define ctypes function prototypes for the SLM DLL."""
        self.Open_Dev = self.lib.Open_Dev
        self.Open_Dev.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_int32]
        self.Open_Dev.restype = ctypes.c_int32

        self.Close_Dev = self.lib.Close_Dev
        self.Close_Dev.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_int32]
        self.Close_Dev.restype = ctypes.c_int32

        self.Write_FMemArray = self.lib.Write_FMemArray
        self.Write_FMemArray.argtypes = [
            ctypes.c_uint8,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        self.Write_FMemArray.restype = ctypes.c_int32

        self.Check_HeadSerial = self.lib.Check_HeadSerial
        self.Check_HeadSerial.argtypes = [
            ctypes.c_uint8,
            ctypes.c_char_p,
            ctypes.c_int32,
        ]
        self.Check_HeadSerial.restype = ctypes.c_int32

        self.Change_DispSlot = self.lib.Change_DispSlot
        self.Change_DispSlot.argtypes = [ctypes.c_uint8, ctypes.c_uint32]
        self.Change_DispSlot.restype = ctypes.c_int32

    def open_device(self):
        """Open the SLM device."""
        result = self.Open_Dev(self.bIDList, ctypes.c_int32(1))
        if result != 1:
            raise RuntimeError(f"Failed to open SLM device. Error code: {result}")
        return result

    def close_device(self):
        """Close the SLM device."""
        result = self.Close_Dev(self.bIDList, ctypes.c_int32(1))
        if result != 1:
            raise RuntimeError(f"Failed to close SLM device. Error code: {result}")
        return result

    def write_fmem_array(self, bID, phase, x_pixel, y_pixel, slot_no):
        """Write data to frame memory array."""
        phase = np.array(phase, dtype=np.uint8).flatten()
        array_in = (ctypes.c_uint8 * len(phase))(*phase)
        array_size = ctypes.c_int32(x_pixel * y_pixel)

        result = self.Write_FMemArray(
            ctypes.c_uint8(bID),
            array_in,
            array_size,
            ctypes.c_uint32(x_pixel),
            ctypes.c_uint32(y_pixel),
            ctypes.c_uint32(slot_no),
        )

        if result != 1:
            raise RuntimeError(f"Failed to write to frame memory array. Error code: {result}")

        return result

    def change_disp_slot(self, bID, slot_no):
        """Change the display slot of the SLM."""
        result = self.Change_DispSlot(ctypes.c_uint8(bID), ctypes.c_uint32(slot_no))
        if result != 1:
            raise RuntimeError(f"Failed to change display slot. Error code: {result}")
        return result

    def write_image(self, image, bID=5, slot_no=1):
        """Process and write an image to the SLM."""
        phase = Image.open(image)
        # Write to SLM
        self.write_fmem_array(bID, phase, 1272, 1024, slot_no)
        self.change_disp_slot(bID, slot_no)
        print("Write success")
        # self.close_device()

    def write_phase(self, phase, bID=5, slot_no=1):
        """Process and write an image to the SLM."""
        if isinstance(phase, torch.Tensor):
            phase = (((phase) / (2 * np.pi) * 255).cpu().detach().numpy().astype(np.uint8))
        else:
            phase = phase.astype(np.uint8)
        pad1 = int((1272 - phase.shape[0]) / 2)
        pad2 = int((1024 - phase.shape[1]) / 2)
        phase = np.pad(phase, ((pad1, pad1), (pad2, pad2)), "constant", constant_values=0)
        phase = np.flipud(phase)
        cropped_image1 = self.crop_center(Image.fromarray(phase))
        cropped_image2 = self.crop_center(Image.open("BlazedGrating_Period2.bmp"))
        superimposed_image = self.superimpose_images(cropped_image1, cropped_image2)
        superimposed_image_pil = Image.fromarray(superimposed_image)
        phase = self.pad_image(superimposed_image_pil)

        # Write to SLM
        self.write_fmem_array(bID, phase, 1272, 1024, slot_no)
        self.change_disp_slot(bID, slot_no)
        print("Write success")

    @staticmethod
    def crop_center(img, size=400):
        width, height = img.size
        left = (width - size) / 2
        top = (height - size) / 2
        right = (width + size) / 2
        bottom = (height + size) / 2
        return img.crop((left, top, right, bottom))

    @staticmethod
    def superimpose_images(img1, img2):
        """Superimpose two images."""
        img1 = np.array(img1, dtype=np.float32)
        img2 = np.array(img2, dtype=np.float32)
        return np.mod(img1 + img2, 255).astype(np.uint8)

    @staticmethod
    def pad_image(img, new_size=(1272, 1024)):
        padded_image = Image.new("L", (new_size[0], new_size[1]), color="black")
        offset = ((new_size[0] - img.width) // 2, (new_size[1] - img.height) // 2)
        padded_image.paste(img, offset)
        return padded_image


class EventCamera:

    def __init__(self, device_path=""):
        self.device_path = device_path
        self.device = None
        # self.device = self._init_device()
        self.height, self.width = None, None
        self.biases = {}
        self.roi = {"x": 430, "y": 185, "width": 360, "height": 360}
        self.captured_frames = SimpleQueue()
        self._capture_thread = None
        self._continuous_capture_started = False

    def _init_device(self):
        """Initialize the event camera device."""
        device = initiate_device(path=self.device_path)
        return device

    def set_camera_params(self, device, bias_file="hpf.bias"):
        """Configure the camera parameters, including biases and ROI."""
        # Trigger In configuration
        i_trigger_in = device.get_i_trigger_in()
        i_trigger_in.enable(metavision_hal.I_TriggerIn.Channel.MAIN)

        # Bias settings
        i_ll_biases = device.get_i_ll_biases()
        if i_ll_biases is not None and bias_file:
            self.biases = get_biases_from_file(bias_file)
            for bias_name, bias_value in self.biases.items():
                i_ll_biases.set(bias_name, bias_value)
            self.biases = i_ll_biases.get_all_biases()

        print(f"Biases: {str(self.biases)}")

        i_roi = device.get_i_roi()
        if i_roi is not None:
            dev_roi = i_roi.Window(self.roi["x"], self.roi["y"], self.roi["width"], self.roi["height"])
            i_roi.set_window(dev_roi)
            i_roi.enable(True)
        print(f"ROI: {str(self.roi)}")

    def start_raw_data_logging(self):
        """Start logging raw data from the event camera."""
        raw_stream = RawReader.from_device(device=self.device, max_events=int(1e9))
        self.device.get_i_events_stream().start()
        j = 0
        while j < 400:
            raw_stream.load_delta_t(33333)
            j = j + 1
            print(j)
        self.device.get_i_events_stream().stop()

    def _events_to_diff_image(self, events, sensor_size):
        """Convert events to a differential image."""
        xs = events["x"]
        ys = events["y"]

        mask = (xs < sensor_size[1]) * (ys < sensor_size[0]) * (xs >= 0) * (ys >= 0)
        coords = np.stack((ys * mask, xs * mask))

        try:
            abs_coords = np.ravel_multi_index(coords, sensor_size)
        except ValueError as e:
            logger.error(f"Error in events_to_diff_image: {e}")
            raise

        img = np.bincount(abs_coords, minlength=sensor_size[0] * sensor_size[1])
        img = img.reshape(sensor_size)
        return img

    def close(self):
        """Close the event camera device."""
        self.device.stop()


class EventCapture(EventCamera):

    def __init__(self, device_path=""):
        super().__init__(device_path)

    def get_with_timeout(self, timeout):
        end_time = time.time() + timeout
        while time.time() < end_time:
            if not self.captured_frames.empty():
                return self.captured_frames.get()
            time.sleep(0.001)
        return None

    def start_continuous_capture(self, batch_size=100, pic_time=2000, ill_time=1000):
        if self._continuous_capture_started:
            print("already started")
            return

        self._capture_thread = Process(target=self._continuous_capture_thread_function,
                                       args=(batch_size, pic_time, ill_time))
        self._capture_thread.daemon = True
        self._capture_thread.start()
        self._continuous_capture_started = True
        print("started")

    def _continuous_capture_thread_function(self, batch_size=100, pic_time=2000, ill_time=1000):
        print("Continuous capture thread started")
        device = self._init_device()
        self.set_camera_params(device)

        my_iterator = EventsIterator.from_device(device, delta_t=pic_time * 2)
        raw_stream = RawReader.from_device(device=device, max_events=int(1e9))
        global_event_count = 0
        triggered_thresholds = []
        threshold_step = batch_size * 2
        next_threshold = -1
        try:
            last_trigger_time = None
            for evs in my_iterator:
                if evs.size != 0:
                    triggers = my_iterator.reader.get_ext_trigger_events().copy()
                    current_trigger_count = len(triggers)
                    # print(f"Current trigger count: {current_trigger_count}")
                    trigger_condition_met = False

                    if len(
                            triggers
                    ) > 0 and current_trigger_count > next_threshold and next_threshold not in triggered_thresholds:
                        trigger_condition_met = True
                        triggered_thresholds.append(next_threshold)
                        print(f"Trigger condition met for  {len(triggers)} triggers")
                        next_threshold += threshold_step

                    if trigger_condition_met:
                        latest_trigger_time = triggers[-1]['t']
                        if last_trigger_time is None or latest_trigger_time > last_trigger_time:
                            last_trigger_time = latest_trigger_time
                            print(f"Producer process: seeking time")
                            if global_event_count == 0:
                                raw_stream.seek_time(triggers[0]['t'] - ill_time)
                            else:
                                # print('seeking time')
                                raw_stream.seek_time(triggers[threshold_step * (len(triggered_thresholds) - 1) -
                                                              1]['t'])
                            i = 0
                            start = time.time()
                            while i < batch_size:
                                eventdata = raw_stream.load_delta_t(pic_time)
                                # eventdata = eventdata[eventdata['p'] > 0]
                                img_bgr = np.zeros((720, 1080), dtype=np.uint8)
                                img = self._events_to_diff_image(eventdata, (720, 1080))
                                img_bgr[img > 0] = 255
                                img = img_bgr[185:(185 + 360), 430:(430 + 360)]

                                self.captured_frames.put({'frame': img, 'triggers': triggers, 'index': i})
                                i += 1
                                global_event_count += 1

        except KeyboardInterrupt:
            print("Keyboard interrupt, stopping")

    def get_batched_frames(self, batch_size, timeout_per_frame_sec=10):
        frame_batch = []
        frames_received = 0
        while frames_received < batch_size:
            try:
                frame = self.get_with_timeout(timeout_per_frame_sec)['frame']
                if frame is None:
                    print(f"Timeout ({timeout_per_frame_sec}s) while getting frame.")
                    break
                frame_batch.append(frame)
                frames_received += 1
                cv2.imwrite(f"frame_{frames_received}.png", frame)
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                break
        return frame_batch


class DcamCamera:
    """
    A class to manage Hamamatsu DCAM camera operations,
    including setup for external triggering and image capture.
    """

    def __init__(self, device_index=0):
        """
        Initializes the DcamCamera object.

        Args:
            device_index (int): The index of the DCAM device to use (default: 0).
        Raises:
            RuntimeError: If DCAM API initialization fails, no devices are found,
                          or device opening fails.
        """
        if not Dcamapi.init():
            raise RuntimeError("Error initializing DCAM API.")

        device_count = Dcamapi.get_devicecount()
        if device_count < 1:
            Dcamapi.uninit()
            raise RuntimeError("No DCAM devices found.")

        self.dcam = Dcam(device_index)
        if not self.dcam.dev_open():
            Dcamapi.uninit()
            raise RuntimeError("Error opening DCAM device.")
        self._is_initialized = True

        self.captured_frames = queue.Queue()
        self._capture_thread = None
        self._continuous_capture_started = False

    def set_subarray_roi(self, hpos, hsize, vpos, vsize):
        dcam = self.dcam
        dcam.prop_setvalue(DCAM_IDPROP.SUBARRAYHPOS, hpos)
        dcam.prop_setvalue(DCAM_IDPROP.SUBARRAYHSIZE, hsize)
        dcam.prop_setvalue(DCAM_IDPROP.SUBARRAYVPOS, vpos)
        dcam.prop_setvalue(DCAM_IDPROP.SUBARRAYVSIZE, vsize)

        dcam.prop_setvalue(DCAM_IDPROP.SUBARRAYMODE, DCAMPROP.MODE.ON)
        print(f"Subarray ROI set to: HPOS={hpos}, HSIZE={hsize}, VPOS={vpos}, VSIZE={vsize}")

    def setup_external_trigger(self):
        """
        Sets up the camera for external triggering.

        Raises:
            RuntimeError: If setting trigger properties fails.
        """
        dcam = self.dcam

        if not dcam.prop_setvalue(DCAM_IDPROP.TRIGGERSOURCE, DCAMPROP.TRIGGERSOURCE.EXTERNAL):
            raise RuntimeError("Error setting trigger source to external.")

        if not dcam.prop_setvalue(DCAM_IDPROP.TRIGGER_MODE, DCAMPROP.TRIGGER_MODE.NORMAL):
            raise RuntimeError("Error setting trigger mode.")

        if not dcam.prop_setvalue(DCAM_IDPROP.TRIGGERACTIVE, DCAMPROP.TRIGGERACTIVE.EDGE):
            raise RuntimeError("Error setting trigger active edge.")

        if not dcam.prop_setvalue(DCAM_IDPROP.TRIGGERPOLARITY, DCAMPROP.TRIGGERPOLARITY.POSITIVE):
            raise RuntimeError("Error setting trigger polarity.")
        if not dcam.prop_setvalue(DCAM_IDPROP.READOUTSPEED, DCAMPROP.READOUTSPEED.FASTEST):
            raise RuntimeError("Error setting readout speed.")
        if not dcam.prop_setvalue(DCAM_IDPROP.EXPOSURETIME, 0.02):
            raise RuntimeError("Error setting exposure time.")

        # print(dcam.prop_getattr(DCAM_IDPROP.IMAGE_PIXELTYPE))
        # if not dcam.prop_setvalue(DCAM_IDPROP.IMAGE_PIXELTYPE,
        #                           DCAM_PIXELTYPE.MONO16):
        #     raise RuntimeError("Error setting pixel type to MONO8.")

        print("Camera is set up for external triggering.")

    def start_continuous_capture(self):
        if self._continuous_capture_started:
            print("already started")
            return

        self.setup_external_trigger()
        self.set_subarray_roi(596, 480, 856, 480)
        dcam = self.dcam

        if not dcam.buf_alloc(1):
            raise RuntimeError("Error allocating buffer.")

        if not dcam.cap_start():
            raise RuntimeError("Error starting capture.")

        print("Starting continuous capture in the background...")
        self._capture_thread = threading.Thread(target=self._continuous_capture_thread_function)
        self._capture_thread.daemon = True
        self._capture_thread.start()
        self._continuous_capture_started = True
        print("started")

    def _continuous_capture_thread_function(self):
        dcam = self.dcam
        capture_count = 0

        try:
            while True:
                if not dcam.wait_capevent_frameready(20000):
                    print("wait_capevent_frameready timeout.")

                    if keyboard.is_pressed('q'):
                        print("Keyboard interrupt detected. Exiting.")
                        break

                    continue

                frame_data = dcam.buf_getlastframedata()
                if frame_data.dtype == np.uint16:
                    # frame_data = np.clip(frame_data, 0, 65535 // 2)
                    imax = np.amax(frame_data)
                    imul = 255 / imax
                    frame_data = frame_data * imul
                    frame_data = np.clip(frame_data, 0, 255).astype(np.uint8)
                    # frame_data = contrast_exponential(frame_data, 2)

                if frame_data is False:
                    raise RuntimeError("Error getting frame data.")

                self.captured_frames.put(frame_data)
                capture_count += 1

        except KeyboardInterrupt:
            print("keyboard interrupt")
        finally:
            cv2.destroyAllWindows()
            dcam.cap_stop()
            dcam.buf_release()
            print("end")
            self._continuous_capture_started = False

    def get_batched_frames(self, batch_size, timeout_per_frame_ms=5000):
        frame_batch = []
        frames_received = 0
        while frames_received < batch_size:
            try:
                frame = self.captured_frames.get(timeout_per_frame_ms)
                frame_batch.append(frame)
                frames_received += 1
                cv2.imwrite(f"frame_{frames_received}.png", frame)
            except queue.Empty:
                print(f"timeout_per_frame_ms={timeout_per_frame_ms} ms")
                break
        return frame_batch

    def stop_continuous_capture(self):
        if self._continuous_capture_started:
            print("stopping")

            self._continuous_capture_started = False
            if self._capture_thread and self._capture_thread.is_alive():
                self._capture_thread.join(timeout=10)
                if self._capture_thread.is_alive():
                    print("waiting for thread to stop")
                else:
                    print("stopped")
            else:
                print("thread not running")
        else:
            print("not started")

    def get_frame(self):
        try:
            img = self.captured_frames.get(timeout=5)
            return img
        except queue.Empty:
            print("No frame received.")
            return None

    def cleanup(self):
        """
        Releases resources, closes the device and uninitializes the DCAM API.
        """
        if hasattr(self, 'dcam') and self.dcam.dev_isopen():
            self.dcam.dev_close()
        if hasattr(self, '_is_initialized') and self._is_initialized:
            Dcamapi.uninit()
        print("DCAM resources cleaned up.")


if __name__ == "__main__":
    dcamcamra = DcamCamera()
    if not dcamcamra._continuous_capture_started:
        dcamcamra.start_continuous_capture()
    dcamcamra.get_batched_frames(100)

    # EventCapture1 = EventCapture()
    # EventCapture1.start_continuous_capture()
    # time.sleep(10)
    pass
