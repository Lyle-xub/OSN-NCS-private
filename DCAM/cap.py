from dcamapi4 import *
from dcam import Dcamapi, Dcam
import matplotlib.pyplot as plt
import cv2

def setup_camera_for_external_trigger(dcam: Dcam):
    """
    配置相机以使用外部触发器。
    """
    # 设置触发源为外部
    if not dcam.prop_setvalue(DCAM_IDPROP.TRIGGERSOURCE, DCAMPROP.TRIGGERSOURCE.EXTERNAL):
        raise RuntimeError("Error setting trigger source to external.")

    # 设置触发模式 - 根据您的相机可能需要调整
    if not dcam.prop_setvalue(DCAM_IDPROP.TRIGGER_MODE, DCAMPROP.TRIGGER_MODE.NORMAL):
        raise RuntimeError("Error setting trigger mode.")

    # 设置触发活动边缘 - 可能需要根据您的触发器硬件进行调整
    if not dcam.prop_setvalue(DCAM_IDPROP.TRIGGERACTIVE, DCAMPROP.TRIGGERACTIVE.EDGE):
        raise RuntimeError("Error setting trigger active edge.")

    print("Camera is set up for external triggering.")

def capture_image_on_trigger(dcam: Dcam):
    """
    等待外部触发并捕获图像。
    """
    # 分配内存以存储图像
    if not dcam.buf_alloc(1):
        raise RuntimeError("Error allocating buffer.")

    # 开始捕获（等待触发）
    if not dcam.cap_start():
        raise RuntimeError("Error starting capture.")

    print("Waiting for trigger...")

    try:
        while True:
            if not dcam.wait_capevent_frameready(10000):  # 10秒超时
                print("Timeout waiting for frame ready. Waiting for next trigger...")
                continue

            frame_data = dcam.buf_getlastframedata()
            if frame_data is False:
                raise RuntimeError("Error getting frame data.")

            # 显示图像
            cv2.imshow("Captured Image", frame_data)
            cv2.waitKey(1)  # 短暂等待，用于OpenCV处理内部事件

            print("Image captured on trigger.")

    except KeyboardInterrupt:
        print("Terminating capture due to interrupt.")
    finally:
        cv2.destroyAllWindows()
        dcam.cap_stop()
        dcam.buf_release()

def main():
    # 初始化DCAM API
    if not Dcamapi.init():
        raise RuntimeError("Error initializing DCAM API.")

    # 获取设备数量
    device_count = Dcamapi.get_devicecount()
    if device_count < 1:
        raise RuntimeError("No DCAM devices found.")

    # 打开第一个设备
    dcam = Dcam(0)
    if not dcam.dev_open():
        raise RuntimeError("Error opening DCAM device.")

    try:
        # 设置相机以使用外部触发器
        setup_camera_for_external_trigger(dcam)

        # 持续捕获图像
        capture_image_on_trigger(dcam)

    finally:
        # 清理
        dcam.dev_close()
        Dcamapi.uninit()


if __name__ == "__main__":
    main()
