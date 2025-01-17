import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import imageio
import nibabel as nib
from scipy import ndimage


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model
	
def fivetests(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    Vol_frac_test = 0
    Regions_test = 0
    Face_test = 0
    Continuity_test = 0
    Thickness_test = 0
    All_tests = 0
    threshold = 0.5
    binary_output = (x > threshold).astype(int)
    #print("binary_output max:", binary_output.max())
    #print("binary_output min:", binary_output.min())
    binary_output_invert = ~binary_output+2
    #print("binary_output invert max:", binary_output_invert.max())
    #print("binary_output invert min:", binary_output_invert.min())    
    #binary_output_invert = np.squeeze(binary_output_invert, axis=-1)
    #print("binary_output_invert shape", binary_output_invert.shape)
    structure_object = ndimage.generate_binary_structure(3, 2) 
    labeled_array, num_regions = ndimage.label(binary_output_invert, structure=structure_object)
    Vol_frac = np.sum(binary_output_invert)/(128*128*128)
    print("Vol_frac", Vol_frac)
    if Vol_frac*100 >= 10 and Vol_frac*100 <= 80:
        Vol_frac_test = 1
        #print("Vol_frac_test", Vol_frac_test)
        #print("num_regions", num_regions)
        if num_regions == 1:
            Regions_test = 1
            #print("Regions_test", Regions_test)

            if np.any(binary_output_invert[:, :, 0]) and np.any(binary_output_invert[:, :, -1]) and \
               np.any(binary_output_invert[:, 0, :]) and np.any(binary_output_invert[:, -1, :]) and \
               np.any(binary_output_invert[0, :, :]) and np.any(binary_output_invert[-1, :, :]):
                Face_test = 1
                
            
            if np.all(binary_output_invert[:, : ,0]) or  np.all(binary_output_invert[:, :, -1]) or  \
               np.all(binary_output_invert[:, 0, :]) or  np.all(binary_output_invert[:, -1, :]) or  \
               np.all(binary_output_invert[0, :, :]) or  np.all(binary_output_invert[-1, :, :]):
                Face_test = 0
       
            #print("Face_test", Face_test)


            if Face_test == 1: 

                face_1_a = np.sum(binary_output_invert[:, : ,0])
                face_1_b = np.sum(binary_output_invert[:, : ,-1])
                face_2_a = np.sum(binary_output_invert[:, 0 ,:])
                face_2_b = np.sum(binary_output_invert[:, -1 ,:])
                face_3_a = np.sum(binary_output_invert[0, : ,:])
                face_3_b = np.sum(binary_output_invert[-1, : ,:])
                
                if face_1_a <= 0.05*127*127 or face_1_b <= 0.05*127*127 or face_2_a <= 0.05*127*127 or \
                   face_2_b <= 0.05*127*127 or face_3_a <= 0.05*127*127 or face_3_b <= 0.05*127*127:
                    Thickness_test = 0
                    #print("Thickness_test", Thickness_test)
                else:
                    Thickness_test = 1
                    #print("Thickness_test", Thickness_test)
                    face_1_and = np.logical_and(binary_output_invert[:, : ,0], binary_output_invert[:, : ,-1]).astype(int)
                    face_1_and_sum = np.sum(face_1_and)
                    face_1_a_ratio = face_1_a/face_1_and_sum
                    face_1_b_ratio = face_1_b/face_1_and_sum
                    face_1_xnor = np.logical_xor(binary_output_invert[:, : ,0], binary_output_invert[:, : ,-1]).astype(int)
                    face_1_xnor_sum = np.sum(face_1_xnor)
                    
                    face_2_and = np.logical_and(binary_output_invert[:, 0 ,:], binary_output_invert[:, -1 ,:]).astype(int)
                    face_2_and_sum = np.sum(face_2_and)
                    face_2_a_ratio = face_2_a/face_2_and_sum
                    face_2_b_ratio = face_2_b/face_2_and_sum
                    face_2_xnor = np.logical_xor(binary_output_invert[:, 0 ,:], binary_output_invert[:, -1 ,:]).astype(int)
                    face_2_xnor_sum = np.sum(face_2_xnor)

                    face_3_and = np.logical_and(binary_output_invert[0, : ,:], binary_output_invert[-1, : ,:]).astype(int)
                    face_3_and_sum = np.sum(face_3_and)
                    face_3_a_ratio = face_3_a/face_3_and_sum
                    face_3_b_ratio = face_3_b/face_3_and_sum
                    face_3_xnor = np.logical_xor(binary_output_invert[0, : ,:], binary_output_invert[-1, : ,:]).astype(int)
                    face_3_xnor_sum = np.sum(face_3_xnor) 

                    if (0.95 <= face_1_a_ratio <= 1.05) and (0.95 <= face_1_b_ratio <= 1.05) and \
                       (0.95 <= face_2_a_ratio <= 1.05) and (0.95 <= face_2_b_ratio <= 1.05) and \
                       (0.95 <= face_3_a_ratio <= 1.05) and (0.95 <= face_3_b_ratio <= 1.05) and \
                       (face_1_xnor_sum <= face_1_a or face_1_xnor_sum <= face_1_b) and \
                       (face_2_xnor_sum <= face_2_a or face_2_xnor_sum <= face_2_b) and \
                       (face_3_xnor_sum <= face_3_a or face_3_xnor_sum <= face_3_b):          
                        Continuity_test = 1
                        #print("Continuity_test", Continuity_test)

    if Vol_frac_test == 1 and Regions_test == 1 and Face_test == 1 and Continuity_test == 1 and Thickness_test == 1:
        All_tests = 1
    return All_tests	


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=10,
        help="sample this often",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=128,
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=128,
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="how many samples to produce for the given prompt",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    opt = parser.parse_args()


    config = OmegaConf.load("configs/stable-diffusion/unet3D_16x16x16_textconditional_cellular.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
    # TODO: check path
    model = load_model_from_config(config, "logs/2024-10-09T20-11-17_unet3D_16x16x16_textconditional_cellular/checkpoints/epoch=000247.ckpt")
    #model = load_model_from_config(config, "logs/2024-05-13T00-15-59_unet3D_16x16x16_textconditional_small_new/checkpoints/epoch=000595.ckpt") 
    #model = load_model_from_config(config, "logs/2024-05-14T13-34-11_unet3D_16x16x16_textconditional_small_new/checkpoints/epoch=000743.ckpt") 
    #model = load_model_from_config(config, "logs/2024-05-16T22-02-06_unet3D_16x16x16_textconditional_small_new2/checkpoints/epoch=000123.ckpt")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    prompt = opt.prompt


    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    all_samples=list()
    with torch.no_grad():
        with model.ema_scope():
            uc = None
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning(opt.n_samples * [""])
            for n in trange(opt.n_iter, desc="Sampling"):
                c = model.get_learned_conditioning(opt.n_samples * [prompt])
                shape = [16, opt.H//8, opt.H//8, opt.W//8]
                samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                 conditioning=c,
                                                 batch_size=opt.n_samples,
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc,
                                                 eta=opt.ddim_eta)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp(x_samples_ddim, min=0.0, max=1.0)

                for x_sample in x_samples_ddim:
                    #x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c t h w -> t h w c')
                    #torch.clamp(x_sample, 0.0, 1.0)
                    x_sample = rearrange(x_sample.cpu().numpy(), 'c t h w -> t h w c')
                    video = []
                    nii_file = []
                    for i in range(x_sample.shape[0]):
                        video_grid = x_sample[i, :,:,:]
                        nii_grid =  (video_grid).astype(np.float32)
                        video_grid = (video_grid).astype(np.uint8)             
                        video.append(video_grid)   
                        nii_file.append(nii_grid) 
                    nii_data = np.array(nii_file)
                    nii_data = np.squeeze(nii_data, axis=-1)
                    print("niidata_shape: ", nii_data.shape)
                    nii_data = nii_data.transpose(1, 2, 0)
                    sod = fivetests(nii_data)
                    #if sod == 1:		
                    nii_img = nib.nifti1.Nifti1Image(nii_data, None)
                    filename = "{}-{}.mp4".format(base_count, i)
                    niiname = "{}-{}.nii".format(base_count, i)
                        #imageio.mimsave(os.path.join(sample_path, filename), video, fps=6)   		
                    nib.save(nii_img, os.path.join(sample_path, niiname)) 
                    base_count += 1
                all_samples.append(x_samples_ddim)


    print(f"Your samples are ready and waiting four you here: \n{outpath} \nEnjoy.")