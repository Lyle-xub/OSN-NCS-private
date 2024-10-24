import ctypes
from PIL import Image
import numpy as np
import torch
import cv2
from interface.imageInterface import *

# Load the DLL library
hpkSLMdaLV = ctypes.windll.LoadLibrary(
    "C:\\Users\\Administrator\\Desktop\\滨松SLM\\X15223-16\\LSH0803871_SLMControl3_DVDdata\\USB_Control_SDK\\hpkSLMdaLV_stdcall_64bit\\hpkSLMdaLV.dll"
)

# Define the ctypes function prototypes
Open_Dev = hpkSLMdaLV.Open_Dev
Open_Dev.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_int32]
Open_Dev.restype = ctypes.c_int32

Close_Dev = hpkSLMdaLV.Close_Dev
Close_Dev.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_int32]
Close_Dev.restype = ctypes.c_int32

Write_FMemArray = hpkSLMdaLV.Write_FMemArray
Write_FMemArray.argtypes = [
    ctypes.c_uint8,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_int32,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint32,
]
Write_FMemArray.restype = ctypes.c_int32

Check_HeadSerial = hpkSLMdaLV.Check_HeadSerial
Check_HeadSerial.argtypes = [ctypes.c_uint8, ctypes.c_char_p, ctypes.c_int32]
Check_HeadSerial.restype = ctypes.c_int32

Change_Disp_slot = hpkSLMdaLV.Change_DispSlot
Change_Disp_slot.argtypes = [ctypes.c_uint8, ctypes.c_uint32]
Change_Disp_slot.restype = ctypes.c_int32


# Function to open the device
def open_device():
    bIDList = (ctypes.c_uint8 * 10)()  # Array of 10 uint8_ts
    result = Open_Dev(bIDList, ctypes.c_int32(1))
    return result, bIDList


# Function to close the device
def close_device(bIDList):
    return Close_Dev(bIDList, ctypes.c_int32(1))


# Function to write to frame memory array
def write_fmem_array(bID, phase, XPixel, YPixel, SlotNo):
    # img = Image.open(image_path)
    # if img.mode != 'L':
    #     img = img.convert('L')
    image_data = np.array(phase)
    flat_image_data = image_data.flatten()
    ArrayIn = (ctypes.c_uint8 * len(flat_image_data))(*flat_image_data)
    ArraySize = ctypes.c_int32(XPixel * YPixel)
    return Write_FMemArray(
        bID,
        ArrayIn,
        ArraySize,
        ctypes.c_uint32(XPixel),
        ctypes.c_uint32(YPixel),
        ctypes.c_uint32(SlotNo),
    )


# Function to check head serial
def check_head_serial(bID, serial_number):
    CharSize = ctypes.c_int32(len(serial_number))
    serial_number_encoded = serial_number.encode("utf-8")
    return Check_HeadSerial(ctypes.c_uint8(bID), serial_number_encoded, CharSize)


# Function to change display slot
def change_disp_slot(bID, SlotNo):
    return Change_Disp_slot(ctypes.c_uint8(bID), ctypes.c_uint32(SlotNo))


def write_one_image_event_has(phase):
    """
    Write_one_image to the SLM.
    """
    if isinstance(phase, torch.Tensor):
        phase = ((phase) / (2 * np.pi) * 255).cpu().detach().numpy().astype(np.uint8)
    else:
        #
        phase = phase
        phase = phase.astype(np.uint8)
    print("ssss", phase.shape)
    pad1 = int((1272 - phase.shape[0]) / 2)
    pad2 = int((1024 - phase.shape[0]) / 2)

    phase = np.pad(
        phase, ((pad1, pad1), (pad2, pad2)), "constant", constant_values=(0, 0)
    )

    cropped_image1 = crop_center(Image.fromarray(phase))
    cropped_image2 = crop_center(
        Image.open("D:\project\control\img/reconstruction/BlazedGrating_Period2.bmp")
    )
    superimposed_image = superimpose_images(cropped_image1, cropped_image2)
    superimposed_image_pil = Image.fromarray(superimposed_image)
    phase = pad_image(superimposed_image_pil)
    phase = np.array(phase)

    phase = np.flipud(phase)
    cv2.imwrite("img/phase_load.bmp", phase)

    result, bIDList = open_device()
    # e.wait()

    result = write_fmem_array(5, phase, 1272, 1024, 1)
    change_disp_slot(5, 1)
    # result = close_device(bIDList)
    print("write success")
    # time.sleep(1)
    # close_device(bIDList)
