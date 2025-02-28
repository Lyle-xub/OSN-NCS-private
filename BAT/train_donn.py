from donn import *
import scipy.io as sio
from matplotlib import colors as color
import random

whole_dim = 700
phase_dim = 400
pixel_size = 12.5e-6
focal_length = 0.3
wave_lambda = 5.20e-07
scalar = 1
num_phases = 4
phase_prop = 0.28
batch_size = 10


def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(11)

from tqdm import tqdm
from omegaconf import OmegaConf

import os


def create_unique_experiment_folder(phase_error, training_method):
    experiment_id = 1
    folder_created = False
    attempt = 0

    while not folder_created:
        # 创建文件夹名称，包括实验ID和参数
        folder_name = f"log/{experiment_id:03}_error_{phase_error}_{training_method}"
        # 确定文件夹路径（这里是在当前工作目录下创建）
        folder_path = os.path.join(os.getcwd(), folder_name)

        # 检查文件夹是否已存在
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)  # 创建文件夹
            print(f"Folder created: {folder_path}")
            folder_created = True
        else:
            # print(
            #     f"Folder {folder_path} already exists, incrementing experiment ID and trying again."
            # )
            experiment_id += 1

    return folder_path


cfg = OmegaConf.load("config.yaml")
# 使用示例
folder_path = create_unique_experiment_folder(f"{phase_prop}", f"{cfg.train}")

if cfg.unitary:
    cfg.log_path = os.path.join(folder_path, "unitary.txt")
elif cfg.is_separable:
    cfg.log_path = os.path.join(folder_path, "separate.txt")
else:
    cfg.log_path = os.path.join(folder_path, "bat.txt")

# write cfg to log
OmegaConf.save(cfg, os.path.join(folder_path, "arg.yaml"))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((400, 400)),
    transforms.Pad([pad, pad], fill=(0), padding_mode="constant"),  # 填充100像素
])
testset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

subtrainset = torch.utils.data.Subset(trainset, range(100))
subtrainloader = torch.utils.data.DataLoader(subtrainset, batch_size=10, shuffle=True)

subtestset = torch.utils.data.Subset(testset, range(50))
subtestloader = torch.utils.data.DataLoader(subtestset, batch_size=1, shuffle=False)

# 取一张图
dataiter = iter(testloader)
images, labels = next(dataiter)
img = images[0]
print(img.shape)

# 初始化模型

# 生成400*400的随机相位
phase_error = torch.randn(1, phase_dim, phase_dim, dtype=torch.float32) * phase_prop
prop_error = torch.randn(1, whole_dim, whole_dim, dtype=torch.float32) * 0.4

# # # # # save phase_error and prop_error
torch.save(phase_error, "phase_error.pth")
torch.save(prop_error, "prop_error.pth")

# load phase_error and prop_error
phase_error = torch.load("phase_error.pth")
prop_error = torch.load("prop_error.pth")

model = DDNN(
    whole_dim,
    phase_dim,
    pixel_size,
    focal_length,
    wave_lambda,
    scalar,
    prop_error,
    phase_error,
    num_phases,
    cfg=cfg,
)

ck = torch.load("D:\project\BAT\ck\model_2024-01-03-11-02-08_94.10.pth")
# print(ck)
model.load_state_dict(ck, strict=False)

model.to("cuda")
img = img.to("cuda")
img = dorefa_a(img, 1)
plt.imshow(img.cpu().squeeze(0).numpy(), cmap=parula)
plt.savefig("img.png")
print(img.device)

loss_slice = slice(
    whole_dim // 2 - phase_dim // 2,
    whole_dim // 2 + phase_dim // 2,
)


class cropped_loss(nn.Module):

    def __init__(self, loss_slice):
        super(cropped_loss, self).__init__()
        self.loss_slice = loss_slice

    def forward(self, output, target):
        # print(self.loss_slice)
        diff = (output - target)[:, self.loss_slice, self.loss_slice]
        return torch.mean(torch.abs(diff)**2)


def diff_loss(x, y):
    return torch.mean(torch.abs(x - y))


def train():
    # total = 0
    # correct = 0
    model.train()

    criterion_pnn = cropped_loss(loss_slice)
    # 交叉熵
    # criterion_pnn = nn.CrossEntropyLoss()
    if cfg.train == "bat":
        params_pnn = [p for n, p in model.named_parameters() if "phase" in n or "w_scalar" in n]
    else:
        params_pnn = [p for n, p in model.named_parameters() if "phase" in n or "w_scalar" in n or "dmd" in n]
    optimizer_pnn = torch.optim.Adam(params_pnn, lr=0.001)
    # scheduler_pnn = torch.optim.lr_scheduler.StepLR(
    #     optimizer_pnn, step_size=3, gamma=0.5
    # )
    scheduler_pnn = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_pnn, 5, 0.00001)

    if cfg.train == "bat":
        criterion_cn = diff_loss
    else:
        criterion_cn = cropped_loss(loss_slice)
    params_cn = [p for n, p in model.named_parameters() if "unet" in n]
    optimizer_cn = torch.optim.Adam(params_cn, lr=0.001)
    # scheduler_cn = torch.optim.lr_scheduler.StepLR(optimizer_cn, step_size=3, gamma=0.5)
    scheduler_cn = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_pnn, 5, 0.00001)

    for epoch in range(5):  # loop over the dataset multiple times
        running_loss_pnn = []
        running_loss_cn = []

        cn_weight = 0.0 if epoch < 1 else 1.0
        loss_pnn = 0.0
        loss_cn = 0.0
        correct_sim = 0
        correct_phy = 0

        # cn_weight =  1.

        running_acc_sim = []
        running_acc_phy = []
        for i, data in tqdm(enumerate(trainloader, 0)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to("cuda"), labels.to("cuda")
            inputs = inputs.squeeze(1)
            # one hot
            labels = F.one_hot(labels, 10).float()
            pad_labels = pad_label(
                labels,
                whole_dim,
                phase_dim,
                det_x,
                det_y,
                square_size,
            )

            # zero the parameter gradients
            optimizer_pnn.zero_grad()
            optimizer_cn.zero_grad()

            if cfg.train == "bat":
                output_phy = model.physical_forward(inputs)
                in_outs_phy = model.in_outs_phy

            output_sim = model(inputs)
            in_outs_sim = model.in_outs_sim
            with torch.no_grad():
                output_sim_det = model.detector(output_sim)

            correct_sim = correct(output_sim_det, labels)
            if cfg.train == "bat":
                if cfg.is_separable:
                    loss_cn = criterion_cn(output_sim, output_phy)
                    loss_cn.backward()
                    optimizer_cn.step()

                    for num in range(1, num_phases + 1):
                        optimizer_cn.zero_grad()
                        outp_unit, outp_unit_phy = model.physical_forward_for_train(in_outs_phy[num - 1],
                                                                                    in_outs_sim[num - 1].detach(), num)
                        loss_cn_unit = criterion_cn(outp_unit, outp_unit_phy)
                        # print(output_sim.shape, inter_phy_detached.shape)
                        loss_cn_unit.backward()
                        optimizer_cn.step()

                    model.zero_grad()

                elif cfg.unitary:
                    loss_cn = (criterion_cn(output_sim, output_phy) + criterion_cn(
                        model.at_mask_intensity1,
                        model.at_mask_intensity_phy1,
                    ) + criterion_cn(
                        model.at_mask_intensity2,
                        model.at_mask_intensity_phy2,
                    ) + criterion_cn(
                        model.at_mask_intensity3,
                        model.at_mask_intensity_phy3,
                    ))
                    loss_cn.backward()
                    optimizer_cn.step()
                    model.zero_grad()
                else:
                    loss_cn = criterion_cn(output_sim, output_phy)
                    loss_cn.backward()
                    optimizer_cn.step()
                    model.zero_grad()
            else:
                # loss_pnn = criterion_pnn(output_sim, pad_labels)
                loss_pnn = criterion_pnn(output_sim_det, labels)
                loss_pnn.backward()
                optimizer_pnn.step()
                running_loss_pnn.append(loss_pnn.item())
                running_loss_cn.append(loss_cn)
                # model.zero_grad()

            if cfg.train == "bat":
                output_sim = model(inputs, cn_weight)
                with torch.no_grad():
                    output_sim_det = model.detector(output_sim)
                correct_sim = correct(output_sim_det, labels)
                with torch.no_grad():
                    output_phy = model.physical_forward(inputs)
                    output_phy_det = model.detector(output_phy)
                    correct_phy = correct(output_phy_det, labels)
                model.phy_replace_sim()
                output_sim.data.copy_(output_phy.data)

                loss_pnn = criterion_pnn(output_sim, pad_labels)
                loss_pnn.backward()
                optimizer_pnn.step()
                running_loss_pnn.append(loss_pnn.item())
                running_loss_cn.append(loss_cn.item())
                # model.zero_grad()

            running_acc_sim.append(correct_sim)
            running_acc_phy.append(correct_phy)
            # print(running_loss_cn)

            if (i + 1) % cfg.log_batch_num == 0:
                content = (f"| epoch = {epoch + 1} " + f"| step = {i + 1:5d} " +
                           f"| loss_pnn = {np.mean(running_loss_pnn):.3f} " +
                           f"| loss_cn = {np.mean(running_loss_cn):.8f} " +
                           f"| acc_sim = {np.mean(running_acc_sim):.3f} " +
                           f"| acc_phy = {np.mean(running_acc_phy):.3f} ")
                write_txt(cfg.log_path, content)
                with torch.no_grad():
                    model.physical_forward(img)
                    # in_outs_phy = model.in_outs_phy
                    # for num in range(1, num_phases):
                    #     outp_unit, outp_unit_phy = model.physical_forward_for_train(
                    #         in_outs_phy[num - 1], num
                    #     )
                    model.plot_sim(img)
                    model.plot_phy(img)
                    model.plot_sim(img, 0.0)
                # if epoch > 3:
                #     print('loss_cn:', loss_cn.item())

        acc = test()
        max_acc = 0
        if cfg.train == "sim":
            if model.dmd1.beta < 100:
                model.dmd1.beta += 10
        # save model
        date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        if acc > max_acc:
            max_acc = acc
            torch.save(model.state_dict(), f"ck/model_{date}_{acc:.2f}.pth")
        scheduler_pnn.step()
        scheduler_cn.step()


def test():
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to("cuda"), labels.to("cuda")
            images = images.squeeze(1)
            # print(images.shape)
            labels = F.one_hot(labels, 10).float()
            # outputs = model.forward(images, train=True)
            outputs = model.physical_forward(images)
            # outputs = model(images)
            outputs = model.detector(outputs)
            # print(outputs.shape)
            _, predicted = torch.max(outputs.data, 1)
            _, corrected = torch.max(labels.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == corrected).sum().item()

    content = f"Accuracy of the network on the 10000 test images: {100 * correct_test / total_test:.2f}%"
    write_txt(cfg.log_path, content)

    return 100 * correct_test / total_test


def test_one_image():

    dataiter = iter(testloader)
    # next 3 img
    for _ in range(5):
        images, labels = next(dataiter)
    img = images[0]

    model.eval()
    with torch.no_grad():
        img = img.to("cuda")
        img = img.squeeze(0)

        print(img.shape)
        out = model(img)
        out2 = model.forward(img, train=True)
        out3 = model.physical_forward(img)
        model.plot_train(img, True)
        model.plot_phy(img)

        print(out)
        print(out2)
        print(out3)


train()
# test_one_image()
# test()
