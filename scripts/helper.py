import argparse
import time
import numpy as np
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from metavision_core.event_io.raw_reader import initiate_device
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import OnDemandFrameGenerationAlgorithm
from metavision_hal import I_TriggerIn
from loguru import logger
from numba import njit


# ---------------- Helper Functions ----------------


def get_biases_from_file(path: str):
    """
    Helper function to read bias from a file.
    """
    biases = {}
    try:
        with open(path, "r") as biases_file:
            for line in biases_file:
                if line.startswith("%"):
                    continue
                split = line.split("%")
                biases[split[1].strip()] = int(split[0])
    except IOError:
        logger.error(f"Cannot open bias file: {path}")
    return biases


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Play & Record Events from Camera")
    parser.add_argument(
        "-s", "--serial-number", default="", help="Serial number of the camera to open."
    )
    parser.add_argument(
        "-o", "--output-raw", default="out1.raw", help="Path to output raw file."
    )
    parser.add_argument(
        "-b",
        "--bias-file",
        dest="bias_file",
        default="hpf.bias",
        help="Path to bias file.",
    )
    return parser.parse_args()


# ---------------- Device Initialization ----------------


def init_device(serial_number):
    """Initialize the camera device."""
    try:
        device = initiate_device(path=serial_number)
        return device
    except Exception as e:
        logger.error(f"Failed to initialize device: {e}")
        return None


def set_camera_params(device, bias_file_path):
    """Configure camera parameters."""
    try:
        # Set up Trigger In
        i_trigger_in = device.get_i_trigger_in()
        i_trigger_in.enable(I_TriggerIn.Channel.MAIN)

        # Set biases
        i_ll_biases = device.get_i_ll_biases()
        if i_ll_biases:
            biases = get_biases_from_file(bias_file_path)
            for bias_name, bias_value in biases.items():
                i_ll_biases.set(bias_name, bias_value)

        # Set ROI
        roi = {"x": 250, "y": 20, "width": 700, "height": 300}
        i_roi = device.get_i_roi()
        if i_roi:
            dev_roi = i_roi.Window(roi["x"], roi["y"], roi["width"], roi["height"])
            i_roi.set_window(dev_roi)
            i_roi.enable(True)
    except Exception as e:
        logger.error(f"Failed to set camera parameters: {e}")


# ---------------- Event Processing ----------------


def events_to_diff_image(events, sensor_size=(720, 1280)):
    """
    Convert events to an image.
    """
    xs = events["x"]
    ys = events["y"]
    ps = events["p"] * 2 - 1

    coords = np.stack((ys, xs))
    abs_coords = np.ravel_multi_index(coords, sensor_size)
    img = np.bincount(abs_coords, weights=ps, minlength=sensor_size[0] * sensor_size[1])
    img = img.reshape(sensor_size)

    return img


def producer(q, events_iterator, num_consumers, e):
    """Producer thread to fetch events."""
    for i, evs in enumerate(events_iterator):
        e.wait()  # Wait for event to start
        q.put(evs)
        if i == 2000:  # Limit the number of events for testing purposes
            break
    for _ in range(num_consumers):
        q.put(None)  # Signal consumers to stop


def consumer(q, consumer_id, results, lock):
    """Consumer thread to process events."""
    while True:
        item = q.get()
        if item is None:
            q.put(None)  # Ensure other consumers get stop signal
            break
        img = events_to_diff_image(item)
        with lock:
            results.append(img)


def start_raw_data_logging(device, output_raw):
    """Start logging raw data."""
    i_events_stream = device.get_i_events_stream()
    i_events_stream.log_raw_data(output_raw)


def start_event_processing(device, e):
    """
    Start event processing using a thread pool.
    """
    e.wait()  # Wait for the event to start
    events_iterator = EventsIterator.from_device(device, delta_t=225)
    q = queue.Queue()
    num_consumers = 8
    results = []
    lock = threading.Lock()

    # Start producer thread
    producer_thread = threading.Thread(
        target=producer, args=(q, events_iterator, num_consumers, e)
    )
    producer_thread.start()

    # Start consumer threads
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=num_consumers) as executor:
        for i in range(num_consumers):
            executor.submit(consumer, q, i, results, lock)
    end_time = time.time()

    logger.info(f"Processing time: {end_time - start_time}s")
    producer_thread.join()

    logger.info(f"Total results collected: {len(results)}")


# ---------------- Main Execution ----------------


def event_recorder(shm_name, e, shared_int):
    """Main function to record events from camera."""
    args = parse_args()
    device = init_device(args.serial_number)
    if device:
        set_camera_params(device, args.bias_file)
        start_raw_data_logging(device, args.output_raw)
        start_event_processing(device, e)
