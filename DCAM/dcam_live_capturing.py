from .dcam import *

import cv2


def dcamtest_show_framedata(data, windowtitle, iShown):
    """
    Show numpy buffer as an image

    Arg1:   NumPy array
    Arg2:   Window name
    Arg3:   Last window status.
        0   open as a new window
        <0  already closed
        >0  already openend
    """
    if iShown > 0 and cv2.getWindowProperty(windowtitle, cv2.WND_PROP_VISIBLE) == 0:
        return -1  # Window has been closed.
    if iShown < 0:
        return -1  # Window is already closed.

    if data.dtype == np.uint16:
        imax = np.amax(data)
        if imax > 0:
            imul = int(65535 / imax)
            # print('Multiple %s' % imul)
            data = data * imul

        cv2.imshow(windowtitle, data)
        return 1
    else:
        print("-NG: dcamtest_show_image(data) only support Numpy.uint16 data")
        return -1


def dcamtest_thread_live(dcam, callback, thread):
    """
    Show live image

    Arg1:   Dcam instance
    """
    if dcam.cap_start() is not False:

        timeout_milisec = 100
        iWindowStatus = 0
        while iWindowStatus >= 0 and (thread.running):
            if (
                dcam.wait_capevent_frameready(timeout_milisec)
                and thread.running is not False
            ):
                data = dcam.buf_getlastframedata()
                if callback == None:
                    iWindowStatus = dcamtest_show_framedata(data, "test", iWindowStatus)
                else:
                    callback(data)
            else:
                thread.stop()
                dcamerr = dcam.lasterr()
                if dcamerr.is_timeout():
                    print("===: timeout")
                else:
                    print("-NG: Dcam.wait_event() fails with error {}".format(dcamerr))
                    break

            key = cv2.waitKey(1)
            if key == ord("q") or key == ord(
                "Q"
            ):  # if 'q' was pressed with the live window, close it
                break

        dcam.cap_stop()
    else:
        print("-NG: Dcam.cap_start() fails with error {}".format(dcam.lasterr()))


def dcam_live_capturing(thread, iDevice=0, callback=None):
    """
    Capture and show a image
    """
    if Dcamapi.init() is not False:
        dcam = Dcam(iDevice)

        thread.dcam = dcam
        if dcam.dev_open() is not False:
            if dcam.buf_alloc(3) is not False:
                # th = threading.Thread(target=dcamtest_thread_live, args=(dcam,))
                # th.start()
                # th.join()
                dcamtest_thread_live(dcam, callback, thread)

                # release buffer
                dcam.buf_release()
            else:
                print(
                    "-NG: Dcam.buf_alloc(3) fails with error {}".format(dcam.lasterr())
                )
            dcam.dev_close()
        else:
            print("-NG: Dcam.dev_open() fails with error {}".format(dcam.lasterr()))
    else:
        print("-NG: Dcamapi.init() fails with error {}".format(Dcamapi.lasterr()))

    Dcamapi.uninit()


def capture_one_image_example():
    if Dcamapi.init() is not False:
        dcam = Dcam(0)

        if dcam.dev_open() is not False:
            print(dcam.prop_setgetvalue(DCAM_IDPROP.EXPOSURETIME, 0.001))
            if dcam.buf_alloc(3) is not False:
                dcam.cap_start()
                if dcam.wait_capevent_frameready(10000) is not False:
                    data = dcam.buf_getlastframedata()
                    cv2.imwrite("img/cmos_test.bmp", data)
                else:
                    dcamerr = dcam.lasterr()
                    if dcamerr.is_timeout():
                        print("===: timeout")
                    else:
                        print(
                            "-NG: Dcam.wait_event() fails with error {}".format(dcamerr)
                        )
                dcam.cap_stop()
                dcam.buf_release()
            else:
                print(
                    "-NG: Dcam.buf_alloc(3) fails with error {}".format(dcam.lasterr())
                )
            dcam.dev_close()
        else:
            print("-NG: Dcam.dev_open() fails with error {}".format(dcam.lasterr()))
    else:
        print("-NG: Dcamapi.init() fails with error {}".format(Dcamapi.lasterr()))

    Dcamapi.uninit()

    return data


if __name__ == "__main__":
    capture_one_image_example()
