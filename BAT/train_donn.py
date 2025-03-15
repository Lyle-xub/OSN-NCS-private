from donn import *
import scipy.io as sio
from matplotlib import colors as color
import random
from tqdm import tqdm
from omegaconf import OmegaConf
import os
from torch.utils.data.distributed import DistributedSampler
from utils import parameter as para
import torch.distributed as dist

# cuda visible devices

# gpu_id = params.gpu_id
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device('cuda', local_rank)

torch.distributed.init_process_group(backend='nccl')

params = para.config()
# whole_dim = 700
# phase_dim = 400
# pad = (whole_dim - phase_dim) // 2
# pixel_size = 12.5e-6
# focal_length = 0.3
# wave_lambda = 5.20e-07
# scalar = 1
# num_phases = 4
# phase_prop = 0.28
# batch_size = 10


def prepare_data():

    pad = (params.whole_dim - params.phase_dim) // 2
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((params.phase_dim, params.phase_dim)),
        transforms.Pad([pad, pad], fill=(0), padding_mode="constant"),
    ])
    testset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False)

    trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    trainsampler = DistributedSampler(trainset)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        sampler=trainsampler,
        batch_size=params.batch_size,
        # shuffle=True,
    )

    subtrainset = torch.utils.data.Subset(trainset, range(100))
    subtrainloader = torch.utils.data.DataLoader(subtrainset, batch_size=10, shuffle=True)

    subtestset = torch.utils.data.Subset(testset, range(50))
    subtestloader = torch.utils.data.DataLoader(subtestset, batch_size=1, shuffle=False)

    # ck = torch.load("D:\project\BAT\ck\model_2024-01-03-11-02-08_94.10.pth")
    # # print(ck)
    # model.load_state_dict(ck, strict=False)

    model = load_params(params)
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank,
                                                      find_unused_parameters=True)

    return model, trainloader, testloader, trainsampler


def train():
    model, trainloader, testloader, trainsampler = prepare_data()
    # total = 0
    # correct = 0
    model.train()

    # criterion_pnn = cropped_loss(loss_slice)
    # criterion_pnn = nn.CrossEntropyLoss()

    criterion_pnn = nn.MSELoss()
    if params.train == "bat":
        params_pnn = [p for n, p in model.named_parameters() if "phase" in n or "w_scalar" in n or "dmd" in n]
    else:
        params_pnn = [p for n, p in model.named_parameters() if "phase" in n or "w_scalar" in n or "dmd" in n]
    optimizer_pnn = torch.optim.Adam(params_pnn, lr=params.pnn_lr)
    # scheduler_pnn = torch.optim.lr_scheduler.StepLR(
    #     optimizer_pnn, step_size=3, gamma=0.5
    # )
    scheduler_pnn = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_pnn, 5, 0.00001)

    if params.train == "bat":
        # criterion_cn = diff_loss
        criterion_cn = nn.MSELoss()
    else:
        criterion_cn = cropped_loss(params.whole_dim, params.phase_dim)
    params_cn = [p for n, p in model.named_parameters() if "unet" in n]
    optimizer_cn = torch.optim.Adam(params_cn, lr=params.cn_lr)
    # scheduler_cn = torch.optim.lr_scheduler.StepLR(optimizer_cn, step_size=3, gamma=0.5)
    scheduler_cn = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_pnn, 5, 0.00001)

    for epoch in range(5):  # loop over the dataset multiple times
        trainsampler.set_epoch(epoch)
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
            # pad_labels = pad_label(
            #     labels,
            #     whole_dim,
            #     phase_dim,
            #     det_x,
            #     det_y,
            #     square_size,
            # )

            # zero the parameter gradients
            optimizer_pnn.zero_grad()
            optimizer_cn.zero_grad()

            if params.train == "bat":
                output_phy = model.module.physical_forward(inputs)
                in_outs_phy = model.module.in_outs_phy

            output_sim = model(inputs)
            in_outs_sim = model.module.in_outs_sim
            with torch.no_grad():
                output_sim_det = model.module.detector(output_sim)

            correct_sim = correct(output_sim_det, labels)
            if params.train == "bat":
                if params.is_separable:
                    loss_cn = criterion_cn(output_sim, output_phy)
                    loss_cn.backward()
                    optimizer_cn.step()

                    for num in range(1, params.layer_num + 1):
                        optimizer_cn.zero_grad()
                        outp_unit, outp_unit_phy = model.module.physical_forward_for_train(
                            in_outs_phy[num - 1], in_outs_sim[num - 1].detach(), num)
                        loss_cn_unit = criterion_cn(outp_unit, outp_unit_phy)
                        # print(output_sim.shape, inter_phy_detached.shape)
                        loss_cn_unit.backward()
                        optimizer_cn.step()

                    model.zero_grad()

                elif params.unitary:
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

            if params.train == "bat":
                output_sim = model(inputs, cn_weight)
                output_sim_det = model.module.detector(output_sim)
                correct_sim = correct(output_sim_det, labels)
                with torch.no_grad():
                    output_phy = model.module.physical_forward(inputs)
                    output_phy_det = model.module.detector(output_phy)
                    correct_phy = correct(output_phy_det, labels)
                model.module.phy_replace_sim()
                output_sim_det.data.copy_(output_phy_det.data)

                loss_pnn = criterion_pnn(output_sim_det, labels)
                loss_pnn.backward()
                optimizer_pnn.step()
                running_loss_pnn.append(loss_pnn.item())
                running_loss_cn.append(loss_cn.item())
                # model.zero_grad()

            # correct_sim_tensor = torch.tensor(correct_sim, dtype=torch.float32, device="cuda")
            # correct_phy_tensor = torch.tensor(correct_phy, dtype=torch.float32, device="cuda")

            # dist.all_reduce(correct_sim_tensor, op=dist.ReduceOp.SUM)
            # dist.all_reduce(correct_phy_tensor, op=dist.ReduceOp.SUM)

            # running_acc_sim.append(correct_sim_tensor.cpu().numpy())
            # running_acc_phy.append(correct_phy_tensor.cpu().numpy())
            running_acc_sim.append(correct_sim)
            running_acc_phy.append(correct_phy)
            # print(running_loss_cn)

            if (i + 1) % params.log_batch_num == 0 and torch.distributed.get_rank() == 0:
                content = (f"| epoch = {epoch + 1} " + f"| step = {i + 1:5d} " +
                           f"| loss_pnn = {np.mean(running_loss_pnn):.3f} " +
                           f"| loss_cn = {np.mean(running_loss_cn):.8f} " +
                           f"| acc_sim = {np.mean(running_acc_sim):.3f} " +
                           f"| acc_phy = {np.mean(running_acc_phy):.3f} ")
                write_txt(params.log_path, content)
                with torch.no_grad():
                    dataiter = iter(testloader)
                    images, labels = next(dataiter)
                    img = images[0]
                    model.module.physical_forward(img)
                    # in_outs_phy = model.in_outs_phy
                    # for num in range(1, num_phases):
                    #     outp_unit, outp_unit_phy = model.physical_forward_for_train(
                    #         in_outs_phy[num - 1], num
                    #     )
                    model.module.plot_sim(img)
                    model.module.plot_phy(img)
                    model.module.plot_sim(img, 0.0)
                # if epoch > 3:
                #     print('loss_cn:', loss_cn.item())
        if torch.distributed.get_rank() == 0:
            acc = test(model, testloader)
            max_acc = 0
            if params.train == "sim":
                if model.dmd1.beta < 100:
                    model.dmd1.beta += 10
            # save model
            date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            if acc > max_acc and torch.distributed.get_rank() == 0:
                max_acc = acc
                torch.save(model.state_dict(), f"{params.folder_path}/model_{date}_{acc:.2f}.pth")
            scheduler_cn.step()


def test(model, testloader):
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
            outputs = model.module.physical_forward(images)
            # outputs = model(images)
            outputs = model.module.detector(outputs)
            # print(outputs.shape)
            _, predicted = torch.max(outputs.data, 1)
            _, corrected = torch.max(labels.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == corrected).sum().item()

    content = f"Accuracy of the network on the 10000 test images: {100 * correct_test / total_test:.2f}%"
    write_txt(params.log_path, content)

    return 100 * correct_test / total_test


def test_one_image(model, testloader):

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
        out2 = model.module.forward(img, train=True)
        out3 = model.module.physical_forward(img)
        model.module.plot_train(img, True)
        model.module.plot_phy(img)

        print(out)
        print(out2)
        print(out3)


train()
# test_one_image()
# test()
