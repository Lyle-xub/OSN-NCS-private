import os
import torch
import numpy
from ctypes import *
from time import sleep
import imageio.v2 as imageio
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout
import interface
import numpy as np
import cv2
from loguru import logger
from interface.imageInterface import *
from .utils import *
import time


def initialize_SLM(trigger=1):
    """
    Initialize the Spatial Light Modulator (SLM) and return relevant parameters.
    """
    # Load the DLLs
    cdll.LoadLibrary(
        "C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\SDK\\Blink_C_wrapper"
    )
    slm_lib = CDLL("Blink_C_wrapper")
    cdll.LoadLibrary(
        "C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\SDK\\ImageGen"
    )
    image_lib = CDLL("ImageGen")

    # Basic parameters for calling Create_SDK
    bit_depth = c_uint(8)
    num_boards_found = c_uint(0)
    constructed_okay = c_uint(-1)
    is_nematic_type = c_bool(1)
    RAM_write_enable = c_bool(1)
    use_GPU = c_bool(1)
    max_transients = c_uint(20)
    board_number = c_uint(1)
    wait_For_Trigger = c_uint(trigger)
    flip_immediate = c_uint(0)  # only supported on the 1024
    timeout_ms = c_uint(5000)
    center_x = c_float(256)
    center_y = c_float(256)
    VortexCharge = c_uint(3)
    fork = c_uint(0)
    RGB = c_uint(0)
    OutputPulseImageFlip = c_uint(1)
    OutputPulseImageRefresh = c_uint(0)  # only supported on 1920x1152, FW rev 1.8.

    # Call the Create_SDK constructor
    try:
        slm_lib.Create_SDK(
            bit_depth,
            byref(num_boards_found),
            byref(constructed_okay),
            is_nematic_type,
            RAM_write_enable,
            use_GPU,
            max_transients,
            0,
        )
    except:
        raise Exception("Failed to construct Blink SDK")

    if constructed_okay.value == 0:
        raise Exception("Blink SDK did not construct successfully")

    if num_boards_found.value != 1:
        raise Exception(
            "Expected 1 SLM controller, found {}".format(num_boards_found.value)
        )

    logger.info("Blink SDK was successfully constructed")

    height = c_uint(slm_lib.Get_image_height(board_number))
    width = c_uint(slm_lib.Get_image_width(board_number))
    depth = c_uint(slm_lib.Get_image_depth(board_number))  # Bits per pixel
    Bytes = c_uint(depth.value // 8)

    return (
        slm_lib,
        image_lib,
        width,
        height,
        depth,
        Bytes,
        board_number,
        wait_For_Trigger,
        flip_immediate,
        OutputPulseImageFlip,
        OutputPulseImageRefresh,
        timeout_ms,
    )


def load_LUT(slm_lib, board_number, width, depth):
    """
    Load the appropriate LUT file based on the SLM specifications.
    """
    lut_path = "C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\"
    if width.value == 512:
        if depth.value == 8:
            slm_lib.Load_LUT_file(
                board_number, (lut_path + "512x512_linearVoltage.LUT").encode("utf-8")
            )
        elif depth.value == 16:
            slm_lib.Load_LUT_file(
                board_number,
                (lut_path + "512x512_16bit_linearVoltage.LUT").encode("utf-8"),
            )
    elif width.value == 1920:
        slm_lib.Load_LUT_file(
            board_number, (lut_path + "1920x1152_linearVoltage.LUT").encode("utf-8")
        )
    elif width.value == 1024:
        slm_lib.Load_LUT_file(
            # board_number, (lut_path + "1024x1024_linearVoltage.LUT").encode("utf-8")
            board_number,
            (lut_path + "slm6628_at532.LUT").encode("utf-8"),
        )
    else:
        raise Exception("Unsupported width for LUT loading")


def load_images(image_paths):
    """
    Load images from the specified paths.
    """
    images = []
    for path in image_paths:
        image = imageio.imread(path)
        image = (image).astype(numpy.uint8)
        image = image.ravel()
        images.append(image)

    return images


def write_image(
    slm_lib,
    board_number,
    image,
    height,
    width,
    Bytes,
    wait_For_Trigger,
    flip_immediate,
    OutputPulseImageFlip,
    OutputPulseImageRefresh,
    timeout_ms,
):
    """
    Write a single image to the SLM.
    """
    retVal = slm_lib.Write_image(
        board_number,
        image.ctypes.data_as(POINTER(c_ubyte)),
        height.value * width.value * Bytes.value,
        wait_For_Trigger,
        flip_immediate,
        OutputPulseImageFlip,
        OutputPulseImageRefresh,
        timeout_ms,
    )
    if retVal == -1:
        raise Exception("Failed to write image to SLM")


def cycle_images(
    slm_lib,
    board_number,
    images,
    num_cycles,
    height,
    width,
    Bytes,
    wait_For_Trigger,
    flip_immediate,
    OutputPulseImageFlip,
    OutputPulseImageRefresh,
    timeout_ms,
):
    """
    Cycle through a set of images for a given number of iterations.
    """
    for i in range(num_cycles):
        for image in images:
            write_image(
                slm_lib,
                board_number,
                image,
                height,
                width,
                Bytes,
                wait_For_Trigger,
                flip_immediate,
                OutputPulseImageFlip,
                OutputPulseImageRefresh,
                timeout_ms,
            )
            retVal = slm_lib.ImageWriteComplete(board_number, timeout_ms)
            if retVal == -1:
                logger.warning("ImageWriteComplete failed, trigger never received?")
                return
            logger.info("success_image")
            sleep(0)  # Adjust as needed


def cleanup_SLM(slm_lib):
    """
    Properly closes the SDK and frees up resources.
    """
    slm_lib.Delete_SDK()
    logger.info("Blink SDK was successfully deleted")


def superimpose_images_slm(image1, image2):
    img1 = np.array(image1)
    img2 = np.array(image2)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    superimpose = np.mod(img1 + img2, 255).astype(np.uint8)
    return superimpose


def write_one_image_superimpose(phase):
    """
    Write_one_image to the SLM.
    """
    phase = ((phase) / (2 * np.pi) * 255).cpu().detach().numpy().astype(np.uint8)

    pad = int((1024 - phase.shape[0]) / 2)
    phase = np.pad(phase, ((pad, pad), (pad, pad)), "constant", constant_values=(0, 0))
    # Load the second image
    image2_path = "C:/Program Files/Meadowlark Optics/Blink OverDrive Plus/WFC Files/slm6628_at532_WFC.bmp"
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # Superimpose images
    phase = superimpose_images_slm(phase, image2)

    cv2.imwrite("img/phase_load.bmp", phase)

    (
        slm_lib,
        image_lib,
        width,
        height,
        depth,
        Bytes,
        board_number,
        wait_For_Trigger,
        flip_immediate,
        OutputPulseImageFlip,
        OutputPulseImageRefresh,
        timeout_ms,
    ) = initialize_SLM(trigger=0)
    load_LUT(slm_lib, board_number, width, depth)
    write_image(
        slm_lib,
        board_number,
        phase.ravel(),
        height,
        width,
        Bytes,
        wait_For_Trigger,
        flip_immediate,
        OutputPulseImageFlip,
        OutputPulseImageRefresh,
        timeout_ms,
    )
    retVal = slm_lib.ImageWriteComplete(board_number, timeout_ms)

    return slm_lib


def write_one_image(phase):
    """
    Write_one_image to the SLM.
    """
    if isinstance(phase, torch.Tensor):
        phase = ((phase) / (2 * np.pi) * 255).cpu().detach().numpy().astype(np.uint8)
    else:
        phase = phase[0]
        phase = phase.astype(np.uint8)

    pad = int((1024 - phase.shape[0]) / 2)
    print(phase.shape)
    phase = np.pad(phase, ((pad, pad), (pad, pad)), "constant", constant_values=(0, 0))
    cv2.imwrite("img/phase_load.bmp", phase)

    cropped_image1 = crop_center(Image.fromarray(phase))
    cropped_image2 = crop_center(
        Image.open("D:\project\control\img/reconstruction/BlazedGrating_Period2.bmp")
    )
    superimposed_image = superimpose_images(cropped_image1, cropped_image2)
    superimposed_image_pil = Image.fromarray(superimposed_image)
    phase = pad_image(superimposed_image_pil)
    phase = np.array(phase)

    phase = np.flipud(phase)

    (
        slm_lib,
        image_lib,
        width,
        height,
        depth,
        Bytes,
        board_number,
        wait_For_Trigger,
        flip_immediate,
        OutputPulseImageFlip,
        OutputPulseImageRefresh,
        timeout_ms,
    ) = initialize_SLM(trigger=0)
    load_LUT(slm_lib, board_number, width, depth)
    write_image(
        slm_lib,
        board_number,
        phase.ravel(),
        height,
        width,
        Bytes,
        wait_For_Trigger,
        flip_immediate,
        OutputPulseImageFlip,
        OutputPulseImageRefresh,
        timeout_ms,
    )
    retVal = slm_lib.ImageWriteComplete(board_number, timeout_ms)

    return slm_lib


def write_one_image_event(phase, e):
    """
    Write_one_image to the SLM.
    """
    if isinstance(phase, torch.Tensor):
        phase = ((phase) / (2 * np.pi) * 255).cpu().detach().numpy().astype(np.uint8)
    else:
        phase = phase[0]
        phase = phase.astype(np.uint8)

    pad1 = int((1272 - phase.shape[0]) / 2)
    pad2 = int((1024 - phase.shape[0]) / 2)
    print(phase.shape)
    phase = np.pad(
        phase, ((pad1, pad1), (pad2, pad2)), "constant", constant_values=(0, 0)
    )
    cv2.imwrite("img/phase_load.bmp", phase)

    cropped_image1 = crop_center(Image.fromarray(phase))
    cropped_image2 = crop_center(
        Image.open("D:\project\control\img/reconstruction/BlazedGrating_Period2.bmp")
    )
    superimposed_image = superimpose_images(cropped_image1, cropped_image2)
    superimposed_image_pil = Image.fromarray(superimposed_image)
    phase = pad_image(superimposed_image_pil)
    phase = np.array(phase)

    phase = np.flipud(phase)

    (
        slm_lib,
        image_lib,
        width,
        height,
        depth,
        Bytes,
        board_number,
        wait_For_Trigger,
        flip_immediate,
        OutputPulseImageFlip,
        OutputPulseImageRefresh,
        timeout_ms,
    ) = initialize_SLM(trigger=0)
    load_LUT(slm_lib, board_number, width, depth)
    e.wait()

    begin = time.time()
    write_image(
        slm_lib,
        board_number,
        phase.ravel(),
        height,
        width,
        Bytes,
        wait_For_Trigger,
        flip_immediate,
        OutputPulseImageFlip,
        OutputPulseImageRefresh,
        timeout_ms,
    )
    print("Time to write image:", time.time() - begin)
    retVal = slm_lib.ImageWriteComplete(board_number, timeout_ms)
    time.sleep(2)

    cleanup_SLM(slm_lib)


def write_a_image(image="img/holo/big_shift_22.bmp"):
    import cv2
    import numpy as np

    image = cv2.imread(image)
    image = image.astype(np.uint8)
    (
        slm_lib,
        image_lib,
        width,
        height,
        depth,
        Bytes,
        board_number,
        wait_For_Trigger,
        flip_immediate,
        OutputPulseImageFlip,
        OutputPulseImageRefresh,
        timeout_ms,
    ) = initialize_SLM(trigger=0)
    load_LUT(slm_lib, board_number, width, depth)
    write_image(
        slm_lib,
        board_number,
        image.ravel(),
        height,
        width,
        Bytes,
        wait_For_Trigger,
        flip_immediate,
        OutputPulseImageFlip,
        OutputPulseImageRefresh,
        timeout_ms,
    )
    retVal = slm_lib.ImageWriteComplete(board_number, timeout_ms)

    return slm_lib


def ctr_main():
    (
        slm_lib,
        image_lib,
        width,
        height,
        depth,
        Bytes,
        board_number,
        wait_For_Trigger,
        flip_immediate,
        OutputPulseImageFlip,
        OutputPulseImageRefresh,
        timeout_ms,
    ) = initialize_SLM()

    load_LUT(slm_lib, board_number, width, depth)

    image_paths = ["img/big_shift_22.bmp", "img/BlazedGrating_Period5.bmp"]
    images = load_images(image_paths)

    try:
        cycle_images(
            slm_lib,
            board_number,
            images,
            100000,
            height,
            width,
            Bytes,
            wait_For_Trigger,
            flip_immediate,
            OutputPulseImageFlip,
            OutputPulseImageRefresh,
            timeout_ms,
        )
    finally:
        cleanup_SLM(slm_lib)


class SLM_trigger_thread(QThread):
    errorOccurred = pyqtSignal(str)

    def __init__(
        self, image_paths=["img/big_shift_22.bmp", "img/BlazedGrating_Period5.bmp"]
    ):
        QThread.__init__(self)
        self.image_paths = image_paths
        self.slm_lib = None
        self.running = True

    def run(self):
        try:
            (
                self.slm_lib,
                image_lib,
                width,
                height,
                depth,
                Bytes,
                board_number,
                wait_For_Trigger,
                flip_immediate,
                OutputPulseImageFlip,
                OutputPulseImageRefresh,
                timeout_ms,
            ) = initialize_SLM()
            load_LUT(self.slm_lib, board_number, width, depth)

            images = load_images(self.image_paths)

            cycle_images(
                self.slm_lib,
                board_number,
                images,
                100000,
                height,
                width,
                Bytes,
                wait_For_Trigger,
                flip_immediate,
                OutputPulseImageFlip,
                OutputPulseImageRefresh,
                timeout_ms,
            )
        except Exception as e:
            print(f"Error occurred: {e}")
            self.errorOccurred.emit(str(e))
            if self.slm_lib is not None:
                cleanup_SLM(self.slm_lib)
                logger.info("SLM thread resources cleaned up and stopped")

    def stop(self):
        logger.info("Stopping SLM thread...")
        self.running = False


class SLM_thread(QThread):
    """
    A thread class for controlling the SLM (Spatial Light Modulator).

    Attributes:
        errorOccurred (pyqtSignal): A signal emitted when an error occurs.
        image_paths (list): A list of image paths to be displayed on the SLM.
        slm_lib: A reference to the SLM library.
        running (bool): A flag indicating whether the thread is running.

    Methods:
        run(): The main execution method of the thread.
        stop(): Stops the thread and cleans up resources.
    """

    errorOccurred = pyqtSignal(str)

    def __init__(
        self, image_paths=["img/big_shift_22.bmp", "img/BlazedGrating_Period5.bmp"]
    ):
        """
        Initializes the SLM_thread object.

        Args:
            image_paths (list, optional): A list of image paths to be displayed on the SLM.
        """
        QThread.__init__(self)
        self.image_paths = image_paths
        self.slm_lib = None
        self.running = True

    def run(self):
        """
        The main execution method of the thread.
        """
        self.running = True
        try:
            self.slm_lib = write_a_image()
        except Exception as e:
            print(f"Error occurred: {e}")
            self.errorOccurred.emit(str(e))
            if self.slm_lib is not None:
                cleanup_SLM(self.slm_lib)
                logger.info("SLM thread resources cleaned up and stopped")

    def stop(self):
        """
        Stops the thread and cleans up resources.
        """
        logger.info("SLM thread stopped")
        if self.slm_lib is not None:
            cleanup_SLM(self.slm_lib)
        self.running = False


if __name__ == "__main__":
    ctr_main()
