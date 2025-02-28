from system.exputils import *
from laser import FisbaReadyBeam as Laser
import os
from system.utils import *
from system.parameter import ExpParams
from system.dataset import *
from system.parameter import *
from system.agent import BaseAgent
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from system.model import *
import ALP4
from torchvision.transforms import functional as TF
import pyiqa


class ExpBase(BaseAgent):
    params = config('exp')

    def __init__(self, exp_id):
        super().__init__(self.params)
        self.params.checkpoint_dir, self.params.exp_id = create_checkpoint_directory(self.params.exp)

        print("Parameters:")
        print(f"  Experiment ID: {self.params.exp_id}")
        print(f"  Model Type: {self.params.model_type}")
        print(f"  Whole Dim: {self.params.core['network']['whole_dim']}")
        print(f"  Device: {self.params.device}")

        self._load_params()
        self.dmd = DiMD(self.bs)
        self.slm = SLM()
        self.ec = EventCamera()
        self.Laser = Laser(port="COM3")
        self.camera = DcamCamera()
        # self.reset()
        self.root_dir = f"data\\{exp_id}\\"
        if self.mt == "s2nn":
            self._prepare_data()
            self._prepare_transform()
        elif self.mt == "s4nn":
            self._prepare_s4nn()
        else:
            assert False, "Model type not supported."

        os.makedirs(self.root_dir, exist_ok=True)

    def reset(self):
        self.dmd.reset()
        self.Laser.set_brightness([0.0, 0.0, 0.0])

    def test(self):
        self._laser_state(True, 1)
        self.dmd.put_white()
        self.slm.write_phase(np.zeros((1024, 1280)))
        return

    def _load_params(self):
        self.bs = self.params.batch_size
        c = self.params.core["network"]
        self.wd = c["whole_dim"]
        self.pd = c["phase_dim"]
        self.dt = self.params.core["data"]["detectors"]
        self.lan = c["layer_num"]
        self.pic_time = self.params.pic_time
        self.ill_time = self.params.ill_time
        self.mt = self.params.model_type

        if self.mt == "s4nn":
            self.dec_num = 2
            self.lan = 1
        elif self.mt == "s3nn":
            self.dec_num = 6
        elif self.mt == "s2nn":
            self.dec_num = 10
        else:
            assert False, "Model type not supported."

    def _prepare_data(self):
        subset_size = 12000
        pad = (self.wd - self.pd) // 2
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                (self.pd, self.pd),
                antialias=True,
            ),
            transforms.Pad([pad, pad], fill=(0), padding_mode="constant"),
        ])
        dev_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                (self.pd, self.pd),
                antialias=True,
            ),
            transforms.Pad([pad, pad], fill=(0), padding_mode="constant"),
        ])
        self.train_dataset = torchvision.datasets.MNIST("data", train=True, transform=train_transform, download=True)
        self.val_dataset = torchvision.datasets.MNIST("data", train=False, transform=dev_transform, download=True)
        self.trainloader = DataLoader(dataset=self.train_dataset, batch_size=self.bs, shuffle=True, pin_memory=True)
        self.testloader = DataLoader(dataset=self.val_dataset, batch_size=self.bs, shuffle=False, pin_memory=True)

        train_indices = torch.randperm(len(self.train_dataset))[:subset_size]
        val_indices = torch.randperm(len(self.val_dataset))[:min(len(self.val_dataset), subset_size)]

        train_dataset = Subset(self.train_dataset, train_indices)
        val_dataset = Subset(self.val_dataset, val_indices)

        self.subtrainloader = DataLoader(dataset=train_dataset, batch_size=self.bs, shuffle=True, pin_memory=True)
        self.subtestloader = DataLoader(dataset=val_dataset, batch_size=self.bs, shuffle=False, pin_memory=True)

        return

    def _prepare_transform(self):
        pad = (self.wd - self.pd) // 2
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.pd, self.pd), antialias=True),
            transforms.Pad([pad, pad], fill=0, padding_mode="constant"),
        ])

        size = int(self.wd * 1.575) + 1
        self.dmd_transform = transforms.Compose([
            transforms.Lambda(lambda x: x * 255),
            transforms.Resize(
                (size, size),
                interpolation=transforms.InterpolationMode.NEAREST,
                antialias=True,
            ),
            transforms.Pad(
                [(2560 - size) // 2, (1600 - size) // 2],
                fill=0,
                padding_mode="constant",
            ),
            transforms.Lambda(lambda x: TF.rotate(x, -45)),
        ])

    def _prepare_for_dat_train(self):
        self.criterion_pnn = nn.CrossEntropyLoss()

        self.params_pnn = [p for n, p in self.network.named_parameters() if "phase" in n or "w" == n or "dmd" in n]
        self.optimizer_pnn = torch.optim.SGD(self.params_pnn, lr=0.1, momentum=0.9)
        self.scheduler_pnn = torch.optim.lr_scheduler.StepLR(self.optimizer_pnn, 5, 0.1)

        if self.params.train == "bat":
            self.criterion_cn = diff_loss

        else:
            self.criterion_cn = cropped_loss(self.params.loss_slice)
        params_cn = [p for n, p in self.network.named_parameters() if "unet" in n]
        self.optimizer_cn = torch.optim.Adam(params_cn, lr=0.0001)
        self.scheduler_cn = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_cn, self.params.max_epoch,
                                                                       0.00001)

    def _prepare_s4nn(self,):
        filpath = S4NNFilePathManager(self.params.dataset).get_paths()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.pd, self.pd), antialias=True),
            transforms.Pad(((self.wd - self.pd) // 2, (self.wd - self.pd) // 2)),
        ])

        self.train_transform = transforms.Compose([
            transforms.Resize((self.pd, self.pd), antialias=True),
            transforms.Pad((
                (self.wd - self.pd) // 2,
                (self.wd - self.pd) // 2,
            )),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5)
        ])

        size = int(self.wd * 1.575)
        self.dmd_transform = transforms.Compose([
            transforms.Lambda(lambda x: x * 255),
            transforms.Resize(
                (size, size),
                interpolation=transforms.InterpolationMode.NEAREST,
                antialias=True,
            ),
            transforms.Pad(
                [(2560 - size) // 2, (1600 - size) // 2],
                fill=0,
                padding_mode="constant",
            ),
            transforms.Lambda(lambda x: TF.rotate(x, -45)),
        ])

        train_dataset = HDF5Dataset(file_path=filpath['train'], transform=self.train_transform)
        self.trainloader = DataLoader(train_dataset, batch_size=self.bs, shuffle=True, drop_last=True)
        val_dataset = HDF5Dataset(file_path=filpath['dev'], transform=self.transform)
        self.testloader = DataLoader(val_dataset, batch_size=self.bs, shuffle=False, drop_last=True)


class FullSystem(ExpBase):

    def __init__(self,):
        np.random.seed(25)
        torch.cuda.manual_seed_all(25)
        torch.manual_seed(25)
        super().__init__(self.params.exp_id)
        self.root_dir = f"data\\{self.params.exp_id}\\"
        self.load_checkpoint("")
        self.load_phase_patterns()

    def test(self):
        """Test Device"""
        self.dmd.put_white()
        self.Laser.set_brightness([0.0, 20, 0.0])
        return

    def load_phase_patterns(self):
        network = self.network
        self.phase_list_local = []
        for i in range(self.lan):
            phase = getattr(network, f"phase{i + 1}")
            phase = dorefa_w(phase.w_p, 8)
            if isinstance(phase, torch.Tensor):
                phase = phase.cpu().detach().numpy()
                unique_values = np.sort(np.unique(phase))
                value_to_int = {v: i for i, v in enumerate(unique_values)}
                phase = np.vectorize(value_to_int.get)(phase)
                cv2.imwrite(f"phase_{i + 1}.png", phase[0].astype(np.uint8))

    def phase_ready(self, phase):
        phase = dorefa_w(phase.w_p, 8)
        if isinstance(phase, torch.Tensor):
            phase = phase.cpu().detach().numpy()
            unique_values = np.sort(np.unique(phase))
            value_to_int = {v: i for i, v in enumerate(unique_values)}
            phase = np.vectorize(value_to_int.get)(phase)
        return phase[0]

    def mode2_physical_forward_cmos(self, x, layer_num=4):
        self.imgs_list = []
        phase = getattr(self.network, f"phase{layer_num}")
        phase = self.phase_ready(phase)
        self.slm.write_phase(phase)
        if layer_num == 1 or isinstance(x, torch.Tensor):
            x = self.dmd_transform(x).cpu().numpy().astype(np.uint8)
            imgs_list = [x[i] for i in range(x.shape[0])]
            self.dmd.put_imgs(imgs_list, transform=False, pic_time=self.pic_time, ill_time=self.ill_time)
        else:
            self.dmd.put_imgs(x, transform=False, pic_time=self.pic_time, ill_time=self.ill_time)

        captured_batch = self.camera.get_batched_frames(self.bs)
        valid_imgs = [img for img in captured_batch if img is not None]

        if valid_imgs:
            imgs_tensor_list = [self.transform(img) for img in valid_imgs]
            imgs_tensor = torch.stack(imgs_tensor_list).squeeze(1)
            if torch.cuda.is_available():
                self.imgs_tensor = imgs_tensor.cuda()

            processed_imgs_tensor = self.dmd_transform(self.imgs_tensor)
            processed_imgs_numpy = processed_imgs_tensor.cpu().numpy()
            for img in processed_imgs_numpy:
                self.imgs_list.append(img)
        else:
            print("get_batched_frames return None.")

        return self.imgs_list, self.imgs_tensor

    def mode2_physical_forward_event(self, x, layer_num=4):
        self.imgs_list = []
        phase = getattr(self.network, f"phase{layer_num}")
        phase = self.phase_ready(phase)
        self.slm.write_phase(phase)
        if layer_num == 1 or isinstance(x, torch.Tensor):
            print(x.shape)
            x = self.dmd_transform(x).cpu().numpy().astype(np.uint8)
            print(x.shape)
            imgs_list = [x[i] for i in range(x.shape[0])]
            self.dmd.put_imgs(imgs_list, transform=False, pic_time=self.pic_time, ill_time=self.ill_time)
        else:
            self.dmd.put_imgs(x, transform=False, pic_time=self.pic_time, ill_time=self.ill_time)
        captured_batch = self.eventcap.get_batched_frames(self.bs)
        valid_imgs = [img for img in captured_batch if img is not None]

        if valid_imgs:
            imgs_tensor_list = [self.transform(img) for img in valid_imgs]
            imgs_tensor = torch.stack(imgs_tensor_list).squeeze(1)
            if torch.cuda.is_available():
                self.imgs_tensor = imgs_tensor.cuda()
            processed_imgs_tensor = self.dmd_transform(self.imgs_tensor)
            processed_imgs_numpy = processed_imgs_tensor.cpu().numpy()
            for img in processed_imgs_numpy:
                self.imgs_list.append(img)
        else:
            assert False, "get_batched_frames return None."
        return self.imgs_list, self.imgs_tensor

    def mode3_adap_train_epoch(self, epoch, cur):
        self.network.train()
        running_loss = 0.0
        correct = 0
        correct_sim, correct_phy, total = 0, 0, 0
        for i, data in enumerate(self.subtrainloader, 0):
            self.load_phase_patterns()
            labels = (torch.nn.functional.one_hot(data[1], num_classes=10).float().to(self.device))
            pad_labels = pad_label(
                labels,
                self.wd,
                self.pd,
                **self.dt,
            )
            inputs = data[0].to(self.device)
            self.optimizer.zero_grad()
            sim_outputs = self.network(inputs.squeeze(1))
            for j in range(1, cur + 1):
                phy_outputs = inputs
                phy_outputs, phy_tensor = self.mode2_physical_forward_cmos(phy_outputs, layer_num=j)

            phy_outputs = self.mode3_adap_forward(phy_tensor, cur, layer_num=4)
            loss = self.loss_func(phy_outputs, pad_labels)
            loss.backward()
            self.optimizer.step()

            outputs = self.network.detector(phy_outputs)
            _, predicted = torch.max(outputs.data, 1)
            _, corrected = torch.max(labels.data, 1)

            total += labels.size(0)
            correct += (predicted == corrected).sum().item()

            running_loss += loss.item()

            if i % 25 == 24:
                self.logger.info(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 25:.4f}")
                running_loss = 0.0
                self.logger.info(f"Correct: {correct/total:.4f}")

        train_acc = 100 * correct / total
        self.logger.info(f"Epoch {epoch+1}: Training accuracy: {train_acc:.2f}%")
        return running_loss / total

    def mode3_adap_train(self, epoch_num):

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 5, 0.00001)
        self.loss_func = cropped_loss(self.params.loss_slice)
        self._prepare_for_dat_train()

        if self.params.camera == "event":
            self.Laser.set_brightness([0.0, 5, 0.0])
            self.eventcap = EventCapture()
            if not self.eventcap._continuous_capture_started:
                self.eventcap.start_continuous_capture(self.bs, self.pic_time, self.ill_time)
                time.sleep(5)
        else:
            self.Laser.set_brightness([0.0, 25, 0.0])
            if not self.camera._continuous_capture_started:
                self.camera.start_continuous_capture()
                time.sleep(5)

        self.network.cuda()
        self.network.train()
        for cur in range(1, self.lan + 1):
            best_test_acc = 0.0
            for epoch in range(self.params.max_epoch):
                print("start epoch")
                loss = self.mode3_adap_train_epoch(epoch, cur)
                self.scheduler.step()
                test_acc = self._evaluate(epoch)
                if test_acc > 90 and self.network.dmd1.beta.data <= 200:
                    for i in range(1, 2):
                        dmd = getattr(self.network, f"dmd{i}")
                        dmd.beta.data = dmd.beta.data + 3
                        print(f"beta{i}", dmd.beta.data)

                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    torch.save(
                        self.network.state_dict(),
                        f"{self.params.checkpoint_dir}/{best_test_acc:.2f}_{epoch+1:03d}.pth",
                    )
                self.logger.info(f"Best test accuracy so far: {best_test_acc:.2f}%")
            self.logger.info("Finished Training")
        return

    def mode3_adap_forward(self, x, cur=1, layer_num=4):
        for i in range(cur + 1, layer_num + 1):
            phase_layer = getattr(self.network, f"phase{i}")
            dmd = self.network.dmd1
            x = phase_layer(x)
            if i == self.lan:
                x = self.network.prop(x)
                break
            x = dmd(self.network.prop(x))
        out = self.network.w.cuda() * x
        return out

    def mode4_dat_forward(self, input_field, cn_weight=1.0):
        self.in_outs_sim = []
        x = self.network.input(dorefa_a(input_field, 1))
        for i in range(1, self.lan + 1):

            self.in_outs_sim.append(x)
            phase = getattr(self.network, f"phase{i}").cuda()
            x = phase(x)
            x = self.network.prop(x)
            if self.params.train == "bat":
                x = (x + getattr(self.network, f"unet{i}")(
                    (x + getattr(self, f"at_mask_phy{i}").cuda()) / 2) * cn_weight)

            setattr(self, f"at_mask{i}", x)

            if i < self.lan:
                setattr(self, f"at_mask_intensity{i}", CNormSq()(x))
                x = self.network.dmd1(x)
                pass
            else:
                # setattr(self, f"at_mask_intensity{i}", CNormSq()(x))
                # x = self.network.dmd(x)
                # x = x.abs()
                self.at_sensor = x
                self.at_sensor_intensity = x.abs()
                x = self.network.w * x
        return x

    def mode4_physical_forward(self, x):
        self.in_outs_phy = []
        with torch.no_grad():
            x = dorefa_a(x, 1)
            x_tensor = x
            for i in range(1, self.lan + 1):
                self.in_outs_phy.append(x_tensor)
                # phase = getattr(self.network, f"phase{i}")
                if self.params.camera == "event":
                    x, x_tensor = self.mode2_physical_forward_event(x, i)
                else:
                    x, x_tensor = self.mode2_physical_forward_cmos(x, i)

                setattr(self, f"at_mask_phy{i}", x_tensor)

                if i < self.lan:
                    setattr(self, f"at_mask_intensity_phy{i}", CNormSq()(self.network.input(x_tensor)))
                    pass
                else:
                    # setattr(self, f"at_mask_intensity_phy{i}", CNormSq()(self.network.input(x_tensor)))
                    # setattr(self, f"at_mask_intensity_phy{i}", x_tensor.abs())
                    self.at_sensor_phy = x_tensor
                    self.at_sensor_intensity_phy = x_tensor.abs()

                    x_tensor = self.network.w * self.at_sensor_intensity_phy
            return x_tensor

    def mode4_physical_forward_for_train(self, input_field_phy, iter_num=1):

        x_sim = getattr(self.network, f"phase{iter_num}")(self.network.input(input_field_phy))
        x_sim = self.network.prop(x_sim)

        if self.cfg.phy == "new":
            # x_sim = self.network.dmd(x_sim)
            x_sim = x_sim + getattr(self.network, f"unet{iter_num}")(
                (x_sim + getattr(self, f"at_mask_phy{iter_num}")) / 2)

        x_sim = self.network.dmd1(x_sim) if iter_num < self.lan else x_sim
        # x_sim = self.network.dmd(x_sim)
        x_sim = x_sim.abs()

        if self.params.camera == "event":
            _, x_phy = self.mode2_physical_forward_event(input_field_phy, iter_num)
        else:
            _, x_phy = self.mode2_physical_forward_cmos(input_field_phy, iter_num)
        # x_phy = x_phy.abs()
        # x_phy = self.network.dmd(self.network.input(x_phy))
        x_phy = x_phy.abs()

        print('sss')

        return x_sim, x_phy

    def phy_replace_sim(self):
        # state fusion
        if self.params.fusion == "new":
            with torch.no_grad():
                angle = torch.angle(self.at_sensor)
                angle2 = torch.angle(self.at_sensor_phy)
                amp = self.at_sensor_phy
                amp1 = self.at_sensor_intensity.cuda()

                modulus = (1 - self.params.alpha) * torch.abs(amp) + self.params.alpha * amp1

                new_data = modulus * torch.exp(1j * angle * self.params.beta)
                # new_data = modulus
                self.at_sensor.data.copy_(new_data.data)
                self.at_sensor_intensity.data.copy_(self.at_sensor_intensity_phy.data)
                # self.at_sensor_intensity.data.copy_(new_data.data)

                for i in range(1, self.lan):
                    angle = torch.angle(getattr(self, f"at_mask{i}"))
                    angle2 = torch.angle(getattr(self, f"at_mask_phy{i}"))
                    amp = getattr(self, f"at_mask_phy{i}")
                    # new_data = torch.abs(amp) * torch.exp(1j * angle)

                    amp1 = getattr(self, f"at_mask{i}").abs()

                    modulus = (1 - self.params.alpha) * torch.abs(amp) + self.params.alpha * amp1
                    new_data = modulus * torch.exp(1j * angle * self.params.beta)

                    getattr(self, f"at_mask{i}").data.copy_(new_data.data)
                    getattr(self, f"at_mask_intensity{i}").data.copy_(getattr(self, f"at_mask_intensity_phy{i}").data)
                    # getattr(self, f"at_mask_intensity{i}").data.copy_(new_data.data)
        elif self.params.fusion == "old":
            with torch.no_grad():
                angle = torch.angle(self.at_sensor)
                amp = self.at_sensor_phy
                new_data = torch.abs(amp) * torch.exp(1j * angle)

                self.at_sensor.data.copy_(new_data.data)
                self.at_sensor_intensity.data.copy_(self.at_sensor_intensity_phy.data)

                for i in range(1, self.lan):
                    angle = torch.angle(getattr(self, f"at_mask{i}"))
                    amp = getattr(self, f"at_mask_phy{i}")
                    new_data = torch.abs(amp) * torch.exp(1j * angle)

                    getattr(self, f"at_mask{i}").data.copy_(new_data.data)
                    getattr(self, f"at_mask_intensity{i}").data.copy_(getattr(self, f"at_mask_intensity_phy{i}").data)

    def mode4_dat_train(self):
        # total = 0
        # correct = 0

        self._prepare_for_dat_train()

        if self.params.camera == "event":
            self.Laser.set_brightness([0.0, 100, 0.0])
            self.eventcap = EventCapture()
            if not self.eventcap._continuous_capture_started:
                self.eventcap.start_continuous_capture(self.bs, self.pic_time, self.ill_time)
                time.sleep(5)
        else:
            self.Laser.set_brightness([0.0, 18, 0.0])
            if not self.camera._continuous_capture_started:
                self.camera.start_continuous_capture()
                time.sleep(5)

        self.network.cuda()
        self.network.train()

        for epoch in range(self.params.max_epoch):
            running_loss_pnn = []
            running_loss_cn = []

            cn_weight = 0.0 if epoch < 20 else 1.0
            loss_pnn = 0.0
            loss_cn = 0.0
            correct_sim = 0
            correct_phy = 0

            # cn_weight = 1.

            running_acc_sim = []
            running_acc_phy = []
            for i, data in tqdm(enumerate(self.trainloader, 0)):

                start = time.time()
                inputs, labels = data
                save_image(inputs, f"imgs_num.png")

                inputs, labels = inputs.to("cuda"), labels.to("cuda")
                inputs = inputs.squeeze(1)
                # one hot
                labels = F.one_hot(labels, self.dec_num).float()
                print('0th', time.time() - start)
                # pad_labels = pad_label(
                #     labels,
                #     self.wd,
                #     self.pd,
                #     **self.params.core["data"]["detectors"],
                # )

                # zero the parameter gradients
                self.optimizer_pnn.zero_grad()
                self.optimizer_cn.zero_grad()

                if self.params.train == "bat":
                    output_phy = self.mode4_physical_forward(inputs)
                    in_outs_phy = self.in_outs_phy
                if self.params.train == "bat":
                    if self.params.is_separable:

                        for num in range(1, self.params.core["network"]["layer_num"] + 1):
                            self.optimizer_cn.zero_grad()
                            outp_unit, outp_unit_phy = (self.mode4_physical_forward_for_train(
                                in_outs_phy[num - 1],
                                num,
                            ))
                            loss_cn_unit = self.criterion_cn(outp_unit, outp_unit_phy)
                            # print(output_sim.shape, inter_phy_detached.shape)
                            loss_cn_unit.backward()
                            self.optimizer_cn.step()

                        output_sim = self.mode4_dat_forward(inputs)
                        output_phy = self.mode4_physical_forward(inputs)
                        # print(torch.max(output_sim), torch.max(output_phy))
                        # save_image(output_sim.abs().unsqueeze(1), f"imgs_sim.png")
                        # save_image(output_phy.abs().unsqueeze(1), f"imgs_phy.png")

                        loss_cn = self.criterion_cn(output_sim.abs(), output_phy)
                        loss_cn.backward()
                        self.optimizer_cn.step()

                        # self.network.zero_grad()

                    elif self.params.unitary:
                        output_sim = self.mode4_dat_forward(inputs)
                        output_phy = self.mode4_physical_forward(inputs)
                        # save_image(output_sim.abs().unsqueeze(1), f"imgs_sim.png")
                        # save_image(output_phy.abs().unsqueeze(1), f"imgs_phy.png")

                        loss_cn = (self.criterion_cn(output_sim, output_phy) + self.criterion_cn(
                            self.at_mask_intensity1,
                            self.at_mask_intensity_phy1,
                        ) + self.criterion_cn(
                            self.at_mask_intensity2,
                            self.at_mask_intensity_phy2,
                        ) + self.criterion_cn(
                            self.at_mask_intensity3,
                            self.at_mask_intensity_phy3,
                        ))
                        loss_cn.backward()
                        self.optimizer_cn.step()

                    else:
                        loss_cn = self.criterion_cn(output_sim, output_phy)
                        loss_cn.backward()
                        self.optimizer_cn.step()
                        # self.network.zero_grad()
                else:
                    # pass
                    loss_pnn = self.criterion_pnn(output_sim_det, labels)
                    loss_pnn.backward()
                    self.optimizer_pnn.step()
                    running_loss_pnn.append(loss_pnn.item())
                    running_loss_cn.append(loss_cn)

                if self.params.train == "bat":
                    output_sim = self.mode4_dat_forward(inputs, cn_weight)
                    output_sim_det = self.network.detector(output_sim)
                    correct_sim = correct(output_sim_det, labels)
                    st3_5 = time.time()
                    with torch.no_grad():
                        output_phy = self.mode4_physical_forward(inputs)
                        output_phy_det = self.network.detector(output_phy)
                        correct_phy = correct(output_phy_det, labels)
                        self.phy_replace_sim()
                        output_sim_det.data.copy_(output_phy_det.data)

                    loss_pnn = self.criterion_pnn(output_sim_det, labels)
                    # b = 0.5
                    # loss_pnn = (loss_pnn - b).abs() + b
                    loss_pnn.backward()
                    # flood the loss

                    self.optimizer_pnn.step()
                    running_loss_pnn.append(loss_pnn.item())
                    running_loss_cn.append(loss_cn.item())
                    running_acc_sim.append(correct_sim)
                    running_acc_phy.append(correct_phy)
                    self.load_phase_patterns()
                # print(running_loss_cn)

                if (i + 1) % self.params.log_batch_num == 0:
                    content = (f"| epoch = {epoch + 1} " + f"| step = {i + 1:5d} " +
                               f"| loss_pnn = {np.mean(running_loss_pnn):.3f} " +
                               f"| loss_cn = {np.mean(running_loss_cn):.8f} " +
                               f"| acc_sim = {np.mean(running_acc_sim):.3f} " +
                               f"| acc_phy = {np.mean(running_acc_phy):.3f} ")
                    self.logger.info(content)

                    with torch.no_grad():
                        self.mode4_plot(inputs, 0)
                        self.mode4_plot(inputs, 1)

            acc = self.mode4_test()
            max_acc = 0
            if self.params.train == "sim":
                if self.network.dmd1.beta < 100:
                    self.network.dmd1.beta += 10
            # save model
            date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            if acc > max_acc:
                max_acc = acc
                torch.save(self.network.state_dict(), f"ck/model_{date}_{acc:.2f}.pth")
            self.scheduler_pnn.step()
            self.scheduler_cn.step()

    def mode4_plot(self, input_field, cn_weight=1.0):
        x = self.network.input(dorefa_a(input_field, 1))
        for i in range(1, self.lan + 1):
            phase = getattr(self.network, f"phase{i}")
            x = phase(x)
            x = self.network.prop(x)

            if self.params.train == "bat":

                x = (x + getattr(self.network, f"unet{i}")((x + getattr(self, f"at_mask_phy{i}")) / 2) * cn_weight)
                # x = x + getattr(self.network, f"unet{i}")(x) * cn_weight

            plt.figure()
            plt.imshow(
                x[0].abs().cpu().detach().numpy().reshape(self.wd, self.wd),
                cmap=parula,
            )
            plt.colorbar()
            plt.savefig(f"x_sim{i}_{cn_weight}.png")
            plt.close()
            x = self.network.dmd1(x) if i < self.lan else x
            # x = self.network.dmd(x)
            # x = x.abs()
            plt.figure()
            plt.imshow(
                x[0].abs().cpu().detach().numpy().reshape(self.wd, self.wd),
                cmap=parula,
            )
            plt.colorbar()
            plt.savefig(f"x_sim{i}_dmd_{cn_weight}.png")
            plt.close()
        output_amp = self.network.detector(self.network.w * x.abs())
        return output_amp

    def mode4_test(self,):
        self.network.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                # print(images.shape)
                images, labels = images.to("cuda"), labels.to("cuda")
                images = images.squeeze(1)
                # print(images.shape)
                labels = F.one_hot(labels, self.dec_num).float()
                # outputs = model.forward(images, train=True)
                outputs = self.network(images)
                # outputs = self.mode4_physical_forward(images)
                outputs = self.network.detector(outputs)
                _, predicted = torch.max(outputs.data, 1)
                _, corrected = torch.max(labels.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == corrected).sum().item()

        content = f"Accuracy of the network on the test images: {100 * correct_test / total_test:.2f}%"
        self.logger.info(content)

        return 100 * correct_test / total_test


if __name__ == "__main__":
    agent = FullSystem()
    agent.network.cuda()
    agent.mode4_dat_train()

    pass
