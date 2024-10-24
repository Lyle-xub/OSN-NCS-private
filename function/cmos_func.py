from DCAM.dcamapi4 import *
from DCAM.dcam import Dcamapi, Dcam
import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys
import time
from loguru import logger


def setup_camera_for_external_trigger(dcam: Dcam):
    """
    Configures the camera to use an external trigger.

    Args:
        dcam (Dcam): The camera object.

    Raises:
        RuntimeError: If there is an error setting the trigger source, trigger mode, or trigger active edge.

    Returns:
        None
    """
    # Set the trigger source to external
    if not dcam.prop_setvalue(
        DCAM_IDPROP.TRIGGERSOURCE, DCAMPROP.TRIGGERSOURCE.EXTERNAL
    ):
        raise RuntimeError("Error setting trigger source to external.")

    # Set the trigger mode - adjust as needed based on your camera
    if not dcam.prop_setvalue(DCAM_IDPROP.TRIGGER_MODE, DCAMPROP.TRIGGER_MODE.NORMAL):
        raise RuntimeError("Error setting trigger mode.")

    # Set the trigger active edge - adjust as needed based on your trigger hardware
    if not dcam.prop_setvalue(DCAM_IDPROP.TRIGGERACTIVE, DCAMPROP.TRIGGERACTIVE.EDGE):
        raise RuntimeError("Error setting trigger active edge.")

    logger.info("Camera is set up for external triggering.")


def capture_image_on_trigger(dcam: Dcam, callback, thread):
    """
    Waits for an external trigger and captures an image.

    Args:
        dcam (Dcam): Dcam object for image capture and processing.
        callback (function): The callback function to be called when an image is captured.
        thread (Thread): The thread object controlling the image capture.

    Raises:
        RuntimeError: If buffer allocation fails or capture start fails.

    Notes:
        - This function waits for an external trigger signal and captures an image upon trigger.
        - Uses the running attribute of the thread object to control the loop.
        - After capturing the image, the callback function is called to process the image data.

    """
    # Allocate memory to store the image
    if not dcam.buf_alloc(1):
        raise RuntimeError("Error allocating buffer.")

    # Start capture (waiting for trigger)
    if not dcam.cap_start():
        raise RuntimeError("Error starting capture.")

    logger.info("Waiting for trigger...")
    thread.dcam = dcam

    try:
        while thread.running:  # Use thread.running to control the loop
            if not dcam.wait_capevent_frameready(10000):  # 10-second timeout
                logger.warning(
                    "Timeout waiting for frame ready. Waiting for next trigger..."
                )
                continue

            frame_data = dcam.buf_getlastframedata()
            if frame_data is False:
                raise RuntimeError("Error getting frame data.")

            callback(frame_data)
            logger.info("Image captured on trigger.")
    except KeyboardInterrupt:
        logger.warning("Terminating capture due to interrupt.")
    finally:
        dcam.cap_stop()
        dcam.buf_release()


def init_camera():
    """
    Initializes the DCAM API and opens the first device.

    Returns:
        Dcam: The opened DCAM device.

    Raises:
        RuntimeError: If there is an error initializing the DCAM API,
            no DCAM devices are found, or there is an error opening
            the DCAM device.
    """
    # Initialize the DCAM API
    if not Dcamapi.init():
        raise RuntimeError("Error initializing DCAM API.")

    # Get the device count
    device_count = Dcamapi.get_devicecount()
    if device_count < 1:
        raise RuntimeError("No DCAM devices found.")

    # Open the first device
    dcam = Dcam(0)
    if not dcam.dev_open():
        raise RuntimeError("Error opening DCAM device.")

    return dcam


def capture(dcam):
    """
    Captures an image using the specified camera.

    Args:
        dcam: The camera object.

    Returns:
        None
    """
    dcam = init_camera()
    try:
        setup_camera_for_external_trigger(dcam)
        capture_image_on_trigger(dcam)
    finally:
        dcam.dev_close()
        Dcamapi.uninit()


def capture_one_image(expose_time=0.04):
    """
    Captures a single image using Dcam and returns the processed and original image data.

    Returns:
        data (numpy.ndarray): The processed image data.
        orig_data (numpy.ndarray): The original image data.
    """
    data, orig_data = None, None

    if Dcamapi.init() is not False:
        dcam = Dcam(0)
        if dcam.dev_open() is not False:
            dcam.prop_setgetvalue(DCAM_IDPROP.EXPOSURETIME, expose_time)
            if dcam.buf_alloc(3) is not False:
                dcam.cap_start()
                if dcam.wait_capevent_frameready(10000) is not False:
                    data = dcam.buf_getlastframedata()
                    orig_data = data.copy()
                    # data = np.flipud(data)
                    # data = np.fliplr(data)
                    data = cv2.convertScaleAbs(data, alpha=(255.0 / 65535.0))
                    logger.info("Image captured and processed successfully.")
                else:
                    dcamerr = dcam.lasterr()
                    if dcamerr.is_timeout():
                        logger.warning("Timeout occurred during image capture.")
                    else:
                        logger.error(f"Dcam.wait_event() failed with error: {dcamerr}")
                dcam.cap_stop()
                dcam.buf_release()
            else:
                logger.error(f"Dcam.buf_alloc(3) failed with error: {dcam.lasterr()}")
            dcam.dev_close()
        else:
            logger.error(f"Dcam.dev_open() failed with error: {dcam.lasterr()}")
    else:
        logger.error(f"Dcamapi.init() failed with error: {Dcamapi.lasterr()}")

    Dcamapi.uninit()
    return data, orig_data


if __name__ == "__main__":
    capture_one_image()
