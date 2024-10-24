from multiprocessing.sharedctypes import Value, Array
from threading import Thread, Event
from ctypes import c_int

from FisbaReadyBeam import FisbaReadyBeam

from torchvision import transforms
import torchvision
import torch
from PIL import Image
from function import *
from torch.utils.data import Dataset
import h5py
import torchvision.transforms.functional as TF


class HDF5Dataset(Dataset):
    def __init__(self, file_path, transform=None):
        """
        Args:
            file_path (string): Path to the HDF5 file with images and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__()
        self.file_path = file_path
        self.file = h5py.File(self.file_path, "r")
        self.images = self.file["images"]
        self.labels = self.file["labels"]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Access the image and label from the HDF5 dataset
        image = self.images[idx]
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.int64)
        image = Image.fromarray(image.astype("uint8")).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, label

    def close(self):
        self.file.close()


def set_camera_params_for_kth(device):
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
    roi = {"x": 370, "y": 95, "width": 530, "height": 530}

    i_roi = device.get_i_roi()
    if i_roi is not None:
        dev_roi = i_roi.Window(roi["x"], roi["y"], roi["width"], roi["height"])
        i_roi.set_window(dev_roi)
        i_roi.enable(True)
    print(f"ROI:  {str(roi)}")


def initialize_dmd_II(version="4.3", bit_depth=1, picture_time=1000000, trigger=False):
    dmd = ALP4(version=version)
    dmd.Initialize()

    for i in range(6):
        dmd.SeqAlloc(nbImg=2, bitDepth=1)
        dmd.SeqControl(ALP_BIN_MODE, ALP_BIN_UNINTERRUPTED, ct.c_long(i + 1))
        dmd.SetTiming(ct.c_long(i + 1), pictureTime=30000)

        # if i % 2 == 0:
        #     dmd.SeqPut(
        #         imgData=np.zeros([dmd.nSizeY, dmd.nSizeX]).ravel(),
        #         SequenceId=ct.c_long(i + 2),
        #     )
    dmd.ProjControl(controlType=ALP_PROJ_QUEUE_MODE, value=ALP_PROJ_SEQUENCE_QUEUE)

    return dmd


model = DDNN(
    cfg.reconstruction.whole_dim,
    cfg.reconstruction.phase_dim,
    cfg.reconstruction.pixel_size,
    cfg.reconstruction.distance,
    cfg.reconstruction.wl,
    cfg=cfg,
    layer_num=6,
)


laser = FisbaReadyBeam(port="COM3")
laser.set_brightness([0.0, 10, 0.0])

ck = torch.load("D:\project\model\ck.pth")
model.load_state_dict(ck, strict=False)

model.to("cuda")
model.eval()


dmd = initialize_dmd_II()
ssize = 450
pad_size = 600
transform1 = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((ssize, ssize)),
        transforms.Pad(
            (
                (pad_size - ssize) // 2,
                (pad_size - ssize) // 2,
            )
        ),
    ]
)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((400, 400)),
        transforms.Pad([100, 100], fill=0, padding_mode="constant"),
    ]
)

size = int(cfg.reconstruction.whole_dim * 1.649)
dmd_transform = transforms.Compose(
    [
        transforms.Lambda(lambda x: x * 255),
        transforms.Resize(
            (size, size), interpolation=transforms.InterpolationMode.NEAREST
        ),
        # 将图像旋转-45度
        transforms.Pad(
            [(2560 - size) // 2, (1600 - size) // 2],  # 左右填充  # 上下填充
            fill=0,
            padding_mode="constant",
        ),
        transforms.Lambda(lambda x: TF.rotate(x, -48)),
    ]
)

val_dataset = HDF5Dataset(
    file_path=r"D:\\project\\model\\datasets\\dev.hdf5", transform=transform1
)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

imglist = []
n = np.random.randint(0, len(val_dataset))
img = val_dataset[n][0]
print(img.shape)


img2 = img
plt.imshow(img2.squeeze().numpy(), "plasma")
plt.savefig("D:\project\plot\\kth\img.png")
plt.show()

print(val_dataset[n][1])

output1 = model.input(img.cuda())
img = dmd_transform(output1.abs()).cpu().detach().numpy().astype(np.uint8)
print("ssss", img.max())

imgblack = np.zeros((1600, 2560), dtype=np.uint8)
imglist.append(imgblack.ravel())
imglist.append(img.ravel())
imgarray = np.concatenate(imglist)
laser.set_brightness([0.0, 15, 0.0])
time.sleep(0.1)

for i in range(6):
    layer = getattr(model, f"phase{i+1}")
    phase = dorefa_w(layer.w_p, 8)
    if isinstance(phase, torch.Tensor):
        phase = phase.cpu().detach().numpy()
        unique_values = np.sort(np.unique(phase))
        value_to_int = {v: i for i, v in enumerate(unique_values)}
        phase = np.vectorize(value_to_int.get)(phase)
    write_one_image_event_has(phase[0])

    dmd.SeqPut(imgData=imgarray, SequenceId=ct.c_long(i + 1))

    device = init_device()
    set_camera_params_for_kth(device)
    start_raw_data_logging(device)
    logger.info("event recorder process start")
    events_iterator = EventsIterator.from_device(device, delta_t=100000)
    time.sleep(0.1)
    dmd.Run(loop=False, SequenceId=ct.c_long(i + 1))
    for evs in events_iterator:
        if evs.size != 0:
            triggers = events_iterator.reader.get_ext_trigger_events()
            if len(triggers) > 0:
                print("there are " + str(len(triggers)) + " external trigger events!")
                # events_iterator.reader.clear_ext_trigger_events()
                if len(triggers) >= 3:
                    break

    device = init_device("out.raw")
    logger.info("event recorder process start")
    events_iterator = EventsIterator.from_device(device, delta_t=20000000)
    height, width = events_iterator.get_size()
    frame_gen = OnDemandFrameGenerationAlgorithm(width, height, accumulation_time_us=0)

    img = np.zeros((height, width, 3), dtype=np.uint8)
    output = []
    for evs in events_iterator:

        if evs.size != 0:
            triggers = events_iterator.reader.get_ext_trigger_events()
            if len(triggers) > 0:
                print("there are " + str(len(triggers)) + " external trigger events!)")
                for idx, trigger in enumerate(triggers):
                    img_bgr = np.zeros((height, width), dtype=np.uint8)

                    if trigger["p"] == 1 or (idx + 1) % 4 == 0:
                        continue
                    print(idx)
                    start_ts = triggers[idx]["t"]
                    end_ts = start_ts + 10000
                    selected_evs = evs[
                        (evs["t"] >= start_ts) & (evs["t"] <= end_ts) & (evs["p"] == 1)
                    ]

                    img2 = events_to_diff_image(
                        selected_evs, sensor_size=(height, width)
                    )
                    img_bgr[img2 > 0] = 1
                    img_bgr[img2 < 0] = 0
                    image = img_bgr[95:625, 370:900]
                    cv2.imwrite(f"D:\project\plot\\kth\\img_{i}.png", img_bgr * 255)
                    out = dmd_transform(transform(image)) * 255
                    if i == 5:
                        outputs = transform(image)
                        print(model.detector(outputs))
                    print("out", out.max())
                    cv2.imwrite(
                        f"D:\project\plot\\kth\\DAT_{i}_x.png",
                        out.numpy().squeeze(0),
                    )
    imglist = []
    imglist.append(imgblack.ravel())
    imglist.append(out.numpy().squeeze(0).ravel())
    imgarray = np.concatenate(imglist)
