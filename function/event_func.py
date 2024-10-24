import sys
import argparse
import cv2
import os
import time
import numpy as np
from skvideo.io import FFmpegWriter

from metavision_core.event_io.raw_reader import initiate_device
from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator
from metavision_sdk_core import (
    PeriodicFrameGenerationAlgorithm,
    ColorPalette,
    OnDemandFrameGenerationAlgorithm,
    BaseFrameGenerationAlgorithm,
)
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent
from .utils import get_biases_from_file
from .utils import get_roi_from_file
import metavision_hal

from multiprocessing import shared_memory
from PIL import Image
from scipy.ndimage import rotate
from loguru import logger


def initialize_camera():
    device = initiate_device(path="")

    # Configuration of Trigger In
    # https://docs.prophesee.ai/stable/metavision_sdk/modules/metavision_hal/python_api/bindings.html?highlight=i_triggerin#metavision_hal.I_TriggerIn
    i_trigger_in = device.get_i_trigger_in()
    i_trigger_in.enable(
        metavision_hal.I_TriggerIn.Channel.MAIN
    )  # For our GEN3 VGA EVK channel is 0, for other cameras, contact vendor

    ## bias setting
    i_hw_identification = device.get_i_hw_identification()
    sensor_info = i_hw_identification.get_sensor_info()
    base_path = "camera" + i_hw_identification.get_serial()

    # Retrieve biases within a bias_file
    biases = {}
    bias_file = None

    i_ll_biases = device.get_i_ll_biases()
    if i_ll_biases is not None:
        if bias_file:
            biases = get_biases_from_file(bias_file)
            for bias_name, bias_value in biases.items():
                i_ll_biases.set(bias_name, bias_value)
        biases = i_ll_biases.get_all_biases()

    if device.get_i_ll_biases():
        log_path = base_path + "_" + time.strftime("%y%m%d_%H%M%S", time.localtime())

    print(f"biases:  {str(i_ll_biases.get_all_biases())}")

    # ROI setting
    roi = {"x": 250, "y": 20, "width": 700, "height": 700}

    i_roi = device.get_i_roi()
    if i_roi is not None:
        dev_roi = i_roi.Window(roi["x"], roi["y"], roi["width"], roi["height"])
        i_roi.set_window(dev_roi)
        i_roi.enable(True)
    print(f"ROI:  {str(roi)}")

    return device


def play_and_record(device, output_raw: str, on_new_frame_callback, thread):
    # Activate RAW logging
    # https://docs.prophesee.ai/stable/metavision_sdk/modules/metavision_hal/python_api/bindings.html?highlight=i_eventsstream#metavision_hal.I_EventsStream
    i_events_stream = device.get_i_events_stream()
    i_events_stream.log_raw_data(output_raw)

    # Now that initializations are done, we can proceed with application
    events_iterator = EventsIterator.from_device(device)
    height, width = events_iterator.get_size()
    frame_gen = PeriodicFrameGenerationAlgorithm(
        sensor_height=height, sensor_width=width, accumulation_time_us=4900, fps=60
    )

    frame_gen.set_output_callback(on_new_frame_callback)

    for evs in events_iterator:
        frame_gen.process_events(evs)
        if not thread.running:
            break

        if evs.size != 0 and thread.running:
            triggers = events_iterator.reader.get_ext_trigger_events()
            if len(triggers) > 0:
                print("there are " + str(len(triggers)) + " external trigger events!)")
                for trigger in triggers:
                    print(trigger)
                events_iterator.reader.clear_ext_trigger_events()
    return


def init_device(path=""):
    """初始化设备"""
    device = initiate_device(path=path)
    # 其他初始化代码 ...
    return device


def set_camera_params(device):
    # Configuration of Trigger In
    # https://docs.prophesee.ai/stable/metavision_sdk/modules/metavision_hal/python_api/bindings.html?highlight=i_triggerin#metavision_hal.I_TriggerIn
    i_trigger_in = device.get_i_trigger_in()
    i_trigger_in.enable(
        metavision_hal.I_TriggerIn.Channel.MAIN
    )  # For our GEN3 VGA EVK channel is 0, for other cameras, contact vendor

    ## bias setting
    i_hw_identification = device.get_i_hw_identification()
    sensor_info = i_hw_identification.get_sensor_info()
    base_path = "camera" + i_hw_identification.get_serial()

    # Retrieve biases within a bias_file
    biases = {}
    bias_file = "D:\project\control\hpf.bias"

    i_ll_biases = device.get_i_ll_biases()
    if i_ll_biases is not None:
        if bias_file:
            biases = get_biases_from_file(bias_file)
            for bias_name, bias_value in biases.items():
                i_ll_biases.set(bias_name, bias_value)
        biases = i_ll_biases.get_all_biases()

    if device.get_i_ll_biases():
        log_path = base_path + "_" + time.strftime("%y%m%d_%H%M%S", time.localtime())

    print(f"biases:  {str(i_ll_biases.get_all_biases())}")

    # ROI setting
    # roi = {"x": 385, "y": 95, "width": 530, "height": 530}
    roi = {"x": 395, "y": 85, "width": 530, "height": 530}

    i_roi = device.get_i_roi()
    if i_roi is not None:
        dev_roi = i_roi.Window(roi["x"], roi["y"], roi["width"], roi["height"])
        i_roi.set_window(dev_roi)
        i_roi.enable(True)
    print(f"ROI:  {str(roi)}")


def start_raw_data_logging(device):
    i_events_stream = device.get_i_events_stream()
    i_events_stream.log_raw_data("out.raw")


def events_to_diff_image(events, sensor_size, strict_coord=True):
    """
    Place events into an image using numpy
    """
    xs = events["x"]
    ys = events["y"]

    mask = (xs < sensor_size[1]) * (ys < sensor_size[0]) * (xs >= 0) * (ys >= 0)
    if strict_coord:
        assert (mask == 1).all()
    coords = np.stack((ys * mask, xs * mask))

    try:
        abs_coords = np.ravel_multi_index(coords, sensor_size)
    except ValueError:
        raise ValueError(
            "Issue with input arrays! coords={}, min_x={}, min_y={}, max_x={}, max_y={}, coords.shape={}, sum(coords)={}, sensor_size={}".format(
                coords,
                min(xs),
                min(ys),
                max(xs),
                max(ys),
                coords.shape,
                np.sum(coords),
                sensor_size,
            )
        )

    # img = np.bincount(abs_coords, weights=ps, minlength=sensor_size[0]*sensor_size[1])
    img = np.bincount(abs_coords, minlength=sensor_size[0] * sensor_size[1])
    img = img.reshape(sensor_size)
    return img


def process_and_display_events(shared_memory_name, e, shared_int, device, args):
    events_iterator = EventsIterator.from_device(device, delta_t=1000000)
    height, width = events_iterator.get_size()
    frame_gen = OnDemandFrameGenerationAlgorithm(
        height=height,
        width=width,
        accumulation_time_us=490000,
    )

    img = np.zeros((height, width), dtype=np.uint8)
    DMD_load_img = np.zeros((1600, 2560, 10), np.uint8)
    existing_shm = shared_memory.SharedMemory(name=shared_memory_name)
    shared_array = np.ndarray(
        DMD_load_img.shape, dtype=DMD_load_img.dtype, buffer=existing_shm.buf
    )

    e.set()

    for evs in events_iterator:
        frame_gen.process_events(evs)
        img_bgr = np.zeros((height, width), dtype=np.uint8)

        # print(selected_evs['t'].max(), selected_evs['t'].min())
        img = events_to_diff_image(evs, sensor_size=(height, width))
        img_bgr[img < 0] = 255
        img_bgr[img > 0] = 0

        pad = int((1600 - img_bgr.shape[0]) / 2)
        pad1 = int((2560 - img_bgr.shape[1]) / 2)

        image = np.pad(
            img_bgr, ((pad, pad), (pad1, pad1)), "constant", constant_values=(0, 0)
        )

        rotated_image = rotate(image, -45, reshape=False)
        DMD_load_img[:, :, 0] = rotated_image[:, :]

        cv2.imwrite("img.png", img_bgr)
        # time.sleep(60)
        image_path = "D:\project\control\log\\01-31\green\\20-53-39\\Output1.1_load.png"
        img = Image.open(image_path)
        img_array = np.array(img)
        DMD_load_img[:, :, 0] = img_array[:, :]
        np.copyto(shared_array, DMD_load_img)
        cv2.imwrite("DMD_load_img.png", DMD_load_img[:, :, 0])
        e.set()
        logger.info("image process")
        events_iterator.reader.clear_ext_trigger_events()
    return DMD_load_img[:, :, 0]


def event_recorder(shm_name, e, shared_int):
    device = init_device()
    set_camera_params(device)
    start_raw_data_logging(device)
    process_and_display_events(shm_name, e, shared_int, device)
