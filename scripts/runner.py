import numpy as np
from ALP4 import *
import time
import multiprocessing
from multiprocessing import Process, shared_memory, Value, Event
from PIL import Image
from loguru import logger
from scripts.helper import event_recorder


# ------------------- DMD Functions -------------------


def initialize_dmd(version="4.3", bit_depth=1, picture_time=1000000, trigger=False):
    """
    Initialize the DMD and configure its settings.
    """
    try:
        dmd = ALP4(version=version)
        dmd.Initialize()

        if trigger:
            dmd.ProjControl(controlType=ALP_PROJ_MODE, value=ALP_SLAVE)
            dmd.DevControl(controlType=ALP_TRIGGER_EDGE, value=ALP_EDGE_RISING)
        else:
            dmd.SetTiming(pictureTime=picture_time)

        dmd.SeqAlloc(nbImg=5000, bitDepth=bit_depth)
        dmd.SeqControl(ALP_BIN_MODE, ALP_BIN_UNINTERRUPTED)
        dmd.SetTiming(pictureTime=200)

        return dmd
    except Exception as e:
        logger.error(f"Failed to initialize DMD: {e}")
        raise


def cleanup_dmd(dmd):
    """
    Safely halt and free the DMD resources.
    """
    try:
        dmd.Halt()
        dmd.FreeSeq()
        dmd.Free()
    except Exception as e:
        logger.error(f"Error during DMD cleanup: {e}")


# ------------------- Image Handling -------------------


def prepare_image_sequence(dmd, num_images=2600):
    """
    Prepare a sequence of alternating black and white images for DMD projection.
    """
    try:
        img_black = np.zeros([dmd.nSizeY, dmd.nSizeX], dtype=np.uint8)
        img_white = np.ones([dmd.nSizeY, dmd.nSizeX], dtype=np.uint8) * (2**8 - 1)

        # Alternate black and white images
        return [img_black.ravel(), img_white.ravel()] * (num_images // 2)
    except Exception as e:
        logger.error(f"Error preparing image sequence: {e}")
        raise


def load_image_into_shared_memory(image_path, shm_name):
    """
    Load an image from file and place it into shared memory.
    """
    try:
        img = Image.open(image_path)
        img_array = np.array(img)

        if len(img_array.shape) == 2:
            img_array = img_array[:, :, np.newaxis].repeat(3, axis=2)

        shm = shared_memory.SharedMemory(
            name=shm_name, create=True, size=img_array.nbytes
        )
        shm_array = np.ndarray(img_array.shape, dtype=img_array.dtype, buffer=shm.buf)
        np.copyto(shm_array, img_array)

        return shm
    except Exception as e:
        logger.error(f"Failed to load image into shared memory: {e}")
        raise


# ------------------- Sequence Runner -------------------


def run_image_sequence(shm_name, dmd, e, shared_int, img_list):
    """
    Run the image sequence on the DMD and trigger the event when done.
    """
    try:
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        logger.info("Running image sequence...")

        dmd.SeqPutEx(imgData=img_list, LineOffset=0, LineLoad=0)
        e.set()
        dmd.Run(loop=False)

        time.sleep(6)
        cleanup_dmd(dmd)

        e.clear()
    except Exception as e:
        logger.error(f"Error during image sequence execution: {e}")
    finally:
        existing_shm.close()


# ------------------- Main Process -------------------


def main():
    """
    Main function to run the DMD sequence and shared memory operations.
    """
    try:
        shared_int = Value("i", 0)
        e = Event()
        image_path = "D:/project/control/log/01-31/green/20-53-39/Output0.9_load.png"
        shared_memory_name = "shared_image_2"

        # Load image into shared memory
        shm = load_image_into_shared_memory(image_path, shared_memory_name)
        shm_name = shm.name

        # Initialize DMD and prepare image sequence
        dmd = initialize_dmd(trigger=True)
        img_list = prepare_image_sequence(dmd)

        p = Process(
            target=event_recorder,
            args=(
                shm_name,
                e,
                shared_int,
            ),
        )
        p.start()

        # Start the DMD image sequence process
        process_dmd = Process(
            target=run_image_sequence, args=(shm_name, dmd, e, shared_int, img_list)
        )
        process_dmd.start()

        # Wait for the DMD process to finish
        process_dmd.join()

    except Exception as e:
        logger.error(f"Error in main process: {e}")
    finally:
        shm.unlink()


if __name__ == "__main__":
    main()
