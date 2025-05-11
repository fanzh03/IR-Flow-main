import argparse
import logging
import os.path
import sys
import time
from collections import OrderedDict
import torchvision.utils as tvutils

import numpy as np
import torch
from IPython import embed
import lpips
# import torch.hub

import basicsr.options as option
from basicsr.models import create_model

sys.path.insert(0, "../../")
import utilsss as util
from data import create_dataloader, create_dataset
from data.util import bgr2ycbcr
torch.cuda.empty_cache()
os.environ['TORCH_HOME'] = '/model/.cache/torch'
#### options
parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, required=True, help="Path to options YMAL file.")
opt = option.parse(parser.parse_args().opt, is_train=False)

opt = option.dict_to_nonedict(opt)

#### mkdir and logger
util.mkdirs(
    (
        path
        for key, path in opt["path"].items()
        if not key == "experiments_root"
           and "pretrain_model" not in key
           and "resume" not in key
    )
)

util.setup_logger(
    "base",
    opt["path"]["log"],
    "test_" + opt["name"],
    level=logging.INFO,
    screen=True,
    tofile=True,
)
logger = logging.getLogger("base")
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt["datasets"].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info(
        "Number of test images in [{:s}]: {:d}".format(
            dataset_opt["name"], len(test_set)
        )
    )
    test_loaders.append(test_loader)

# load pretrained model by default
model = create_model(opt)
device = model.device
timestep = False
if opt["network_G"]["which_model_G"] == "ConditionalUNet":
    timestep = True
else:
    timestep = False
    
sde = util.IRSDE(T=opt["sde"]["T"], device=device)
sde.set_model(model.model)
sde.set_timestep(timestep)
sde.set_XT(opt["method"]["isXT"])
solver = opt["method"]["ode"]
timesteps_ode = opt["timesteps_ode"]  # [1, 2, 3, 4, 5]

lpips_fn = lpips.LPIPS(net='alex').to(device)

scale = opt['degradation']['scale']

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt["name"]  # path opt['']
    logger.info("\nTesting [{:s}]...".format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(opt["path"]["results_root"], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results["psnr"] = OrderedDict()
    test_results["ssim"] = OrderedDict()
    test_results["psnr_y"] = OrderedDict()
    test_results["ssim_y"] = OrderedDict()
    test_results["lpips"] = OrderedDict()


    for i in timesteps_ode:
        test_results["psnr"][f'step_{i}'] = []
        test_results["ssim"][f'step_{i}'] = []
        test_results["psnr_y"][f'step_{i}'] = []
        test_results["ssim_y"][f'step_{i}'] = []
        test_results["lpips"][f'step_{i}'] = []
    test_times = []
    print("start test:*************************************************\n")
    for i, test_data in enumerate(test_loader):
        single_img_psnr = []
        single_img_ssim = []
        single_img_psnr_y = []
        single_img_ssim_y = []
        need_GT = False if test_loader.dataset.opt["dataroot_GT"] is None else True
        img_path = test_data["GT_path"][0] if need_GT else test_data["LQ_path"][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        #### input dataset_LQ
        LQ, GT = test_data["LQ"], test_data["GT"]
        model.feed_data(LQ, GT)
        tic = time.time()
        model.test(sde, i, timesteps_ode, solver)
        toc = time.time()
        test_times.append(toc - tic)

        visuals = model.get_current_visuals()
        output = {}
        for key in visuals["Output"]:
            output[key] = util.tensor2img(visuals["Output"][key].squeeze())  # uint8

        LQ_ = util.tensor2img(visuals["Input"].squeeze())  # uint8
        GT_ = util.tensor2img(visuals["GT"].squeeze())  # uint8
        for key, value in output.items():
            logger.info("********************************************************************************************************\n")
            if need_GT:
                gt_img = GT_
                sr_img = value

                crop_border = opt["crop_border"] if opt["crop_border"] else scale
                if crop_border == 0:
                    cropped_sr_img = sr_img
                    cropped_gt_img = gt_img
                else:
                    cropped_sr_img = sr_img[crop_border:-crop_border, crop_border:-crop_border]
                    cropped_gt_img = gt_img[crop_border:-crop_border, crop_border:-crop_border]

                psnr = util.calculate_psnr(cropped_sr_img, cropped_gt_img)
                ssim = util.calculate_ssim(cropped_sr_img, cropped_gt_img)


                sr_tensor = torch.from_numpy(sr_img).float().to(device)
                sr_tensor = (sr_tensor / 127.5) - 1  # Proper scaling to [-1, 1]
                if sr_tensor.ndim == 3:
                    sr_tensor = sr_tensor.permute(2, 0, 1).unsqueeze(0)
                
                gt_tensor = torch.from_numpy(gt_img).float().to(device)
                gt_tensor = (gt_tensor / 127.5) - 1
                if gt_tensor.ndim == 3:
                    gt_tensor = gt_tensor.permute(2, 0, 1).unsqueeze(0)

                # Now compute LPIPS
                # lp_score = lpips_fn(GT_tensor, sr_img_tensor).squeeze().item()
                lp_score = lpips_fn(gt_tensor, sr_tensor).squeeze().item()

                test_results["psnr"][key].append(psnr)
                test_results["ssim"][key].append(ssim)
                test_results["lpips"][key].append(lp_score)

                if len(gt_img.shape) == 3:
                    if gt_img.shape[2] == 3:  # RGB image
                        sr_img_y = bgr2ycbcr(sr_img, only_y=True)
                        gt_img_y = bgr2ycbcr(gt_img, only_y=True)
                        if crop_border == 0:
                            cropped_sr_img_y = sr_img_y
                            cropped_gt_img_y = gt_img_y
                        else:
                            cropped_sr_img_y = sr_img_y[crop_border:-crop_border, crop_border:-crop_border]
                            cropped_gt_img_y = gt_img_y[crop_border:-crop_border, crop_border:-crop_border]
                        psnr_y = util.calculate_psnr(cropped_sr_img_y, cropped_gt_img_y)
                        ssim_y = util.calculate_ssim(cropped_sr_img_y, cropped_gt_img_y)

                        test_results["psnr_y"][key].append(psnr_y)
                        test_results["ssim_y"][key].append(ssim_y)

                        logger.info(
                            "************************** ODE_{} **************************\n"
                            "img{:3d}:{:15s} - PSNR: {:.6f} dB; SSIM: {:.6f}; LPIPS: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.".format(
                                key, i, img_name, psnr, ssim, lp_score, psnr_y, ssim_y
                            )
                        )

                else:
                    logger.info(
                        "************************** ODE_{} **************************\n"
                        "img:{:15s} - PSNR: {:.6f} dB; SSIM: {:.6f}.".format(key, img_name, psnr, ssim)
                    )

                    test_results["psnr_y"][key].append(psnr)
                    test_results["ssim_y"][key].append(ssim)
            else:
                logger.info(img_name)
    for step in timesteps_ode:
        ave_lpips = sum(test_results["lpips"][f'step_{step}']) / len(test_results["lpips"][f'step_{step}'])
        ave_psnr = sum(test_results["psnr"][f'step_{step}']) / len(test_results["psnr"][f'step_{step}'])
        ave_ssim = sum(test_results["ssim"][f'step_{step}']) / len(test_results["ssim"][f'step_{step}'])

        logger.info(
            "************************** ODE_step:{} **************************\n"
            " ----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}; lpips: {:.6f}\n".format(
                step, test_set_name, ave_psnr, ave_ssim, ave_lpips
            )
        )
        if test_results["psnr_y"][f'step_{step}'] and test_results["ssim_y"][f'step_{step}']:
            ave_psnr_y = sum(test_results["psnr_y"][f'step_{step}']) / len(test_results["psnr_y"][f'step_{step}'])
            ave_ssim_y = sum(test_results["ssim_y"][f'step_{step}']) / len(test_results["ssim_y"][f'step_{step}'])
            logger.info(
                "----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}\n".format(
                    ave_psnr_y, ave_ssim_y
                )
            )

        logger.info("----average LPIPS\t: {:.6f}\n".format(ave_lpips))

        print(f"average test time: {np.mean(test_times):.4f}")
