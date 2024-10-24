import sys
import time
from FisbaReadyBeam import FisbaReadyBeam

# from crc import Crc16

# print(dir(Crc16))  # This will list all available attributes in Crc16.


def test():
    laser = FisbaReadyBeam(port="COM3")
    for i in range(30):
        laser.set_brightness([0.0, 0.0, i * 1.0])
    time.sleep(1.0)
    for i in range(30):
        laser.set_brightness([0.0, i * 1.0, 0.0])
    time.sleep(1.0)
    for i in range(30):
        laser.set_brightness([i * 1.0, 0.0, 0.0])
    time.sleep(1.0)
    laser.set_brightness([0, 0, 0])
    laser.close()
    assert True == True


if __name__ == "__main__":
    sys.exit(test())
