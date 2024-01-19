import torch
import argparse
import os

from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler, DDIMScheduler
from huggingface_hub import hf_hub_download
from photomaker import PhotoMakerStableDiffusionXLPipeline


def main():
    from IPython import embed
    embed()
    # base model path
    base_model_path = 'SG161222/RealVisXL_V3.0'
    # download checkpoint
    photomaker_ckpt = hf_hub_download(repo_id="TencentARC/PhotoMaker",
                                      filename="photomaker-v1.bin",
                                      repo_type="model")

    pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        variant="fp16",
    ).to("cuda")

    pipe.load_photomaker_adapter(
        os.path.dirname(photomaker_ckpt),
        subfolder="",
        weight_name=os.path.basename(photomaker_ckpt),
        trigger_word="img"
    )
    pipe.id_encoder.to("cuda")

    # pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    # pipe.fuse_lora()

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    # pipe.set_adapters(["photomaker"], adapter_weights=[1.0])
    pipe.fuse_lora()

    # define and show the input ID images
    input_folder_name = args.input
    image_basename_list = os.listdir(input_folder_name)
    image_path_list = sorted([os.path.join(input_folder_name, basename) for basename in image_basename_list])

    input_id_images = []
    for image_path in image_path_list:
        input_id_images.append(load_image(image_path))

    ## Note that the trigger word `img` must follow the class word for personalization
    # prompt = "sci-fi, closeup portrait photo of an aisa man img wearing the sunglasses in Iron man suit, face, slim body, high quality, film grain"
    # negative_prompt = "(asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth"
    generator = torch.Generator(device="cuda").manual_seed(42)

    ## Parameter setting
    num_steps = 50
    style_strength_ratio = 20
    start_merge_step = int(float(style_strength_ratio) / 100 * num_steps)
    if start_merge_step > 30:
        start_merge_step = 30

    images = pipe(
        prompt=args.prompt,
        input_id_images=input_id_images,
        negative_prompt=args.neg_prompt,
        num_images_per_prompt=4,
        num_inference_steps=num_steps,
        start_merge_step=start_merge_step,
        generator=generator,
    ).images

    os.makedirs(args.output, exist_ok=True)
    for idx, image in enumerate(images):
        image.save(os.path.join(args.output, f"{args.exp_name}_{idx:02d}.png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Photomaker')
    parser.add_argument('--output', type=str, help='the path to save generated images')
    parser.add_argument('--input', type=str, help='the path to the ID images')
    parser.add_argument('--exp-name', type=str)
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--neg-prompt', type=str)
    args = parser.parse_args()

    main()
