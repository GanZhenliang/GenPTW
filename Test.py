import argparse
import os
os.environ["LAMA_MODEL"] = "/root/GenPTW/big-lama.pt"

import ssl
import lpips
from lama import SimpleLama
from accelerate.utils import set_seed

from JND import JND
from train_utils import *
from attack_methods.attack import *

ssl._create_default_https_context = ssl._create_unverified_context
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["WANDB_API_KEY"] = ""
os.environ["WANDB_MODE"] = "offline"


def parse_args():
    parser = argparse.ArgumentParser("Test GenPTW watermark adapter/localizer.")

    # Experiment
    parser.add_argument("--exp_name", type=str, default="Test")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-fs/output")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--seed", type=int, default=2026)

    # Data
    parser.add_argument("--test_img_dir", type=str, default="/autodl-tmp/Datasets/COCO2017/val2017")
    parser.add_argument("--test_ann_file", type=str, default="/autodl-tmp/Datasets/COCO2017/annotations/instances_val2017.json")
    parser.add_argument("--target_names", nargs="*", default=None)
    parser.add_argument("--val_ratio", type=float, default=0.001)
    parser.add_argument("--split_seed", type=int, default=34)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)

    # train
    parser.add_argument("--phi_dimension", type=int, default=64)
    parser.add_argument("--pretrained_vae", type=str, default="/root/GenPTW/vae")
    parser.add_argument("--pretrained_ConvNeXt", type=str, default="/root/GenPTW/pytorch_model.bin")
    parser.add_argument("--pretrained_pipe_sd", type=str, default="/root/autodl-fs/sdinpaint2")
    
    # Model / checkpoint
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pretrained_ckpt", type=str, default="./Checkpoint")
    parser.add_argument("--vae_ckpt_name", type=str, default="diffusion_pytorch_model.safetensors")
    parser.add_argument("--msg_decoder_ckpt_name", type=str, default="msg_decoder.pth")
    parser.add_argument("--localizer_ckpt_name", type=str, default="localizer.pth")
    parser.add_argument("--revision", type=str, default=None)

    # Optimizer / scheduler
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--max_train_steps", type=int, default=5000000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_restarts")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--cosine_cycle", type=int, default=1000)
    parser.add_argument("--scale_lr", action="store_true", default=False)

    # Runtime
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--allow_tf32", action="store_true")

    # Attack / validation
    parser.add_argument("--attack_ramp", type=int, default=1)
    parser.add_argument("--attack_prob_color", type=float, default=0.15)
    parser.add_argument("--attack_prob_noise", type=float, default=0.95)
    parser.add_argument("--attack_prob_jpeg", type=float, default=0.85)
    parser.add_argument("--edit_attack_prob", type=float, default=0.95)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--eval_max_batches", type=int, default=50)
    parser.add_argument("--eval_strength", type=float, default=1.0)

    return parser.parse_args()

def run_validation(
    args,
    test_loader,
    vae,
    msg_decoder,
    localized,
    accelerator,
    global_step,
    lpips_net,
    pipe_inpaint,
    attack_list,
    weight_dtype,
    best_acc,
    best_acc_aigc,
    save_dir,
):
    accelerator.print("🧪 Running validation on AIGC edits...")

    _, best_acc_aigc = test_strategies_aigc(
        test_loader=test_loader,
        vae=vae,
        msg_decoder=msg_decoder,
        localizer=localized,
        phi_dim=args.phi_dimension,
        accelerator=accelerator,
        global_step=global_step,
        lpips_net=lpips_net,
        args=args,
        pipe_lama=pipe_inpaint,
        pipe_cn=None,
        prompt="",
        resolution=args.resolution,
        strength=args.eval_strength,
        max_batches=args.eval_max_batches,
        best_acc=best_acc_aigc,
        do_aug=None,
        save_dir=save_dir,
    )

    maybe_empty_cuda_cache()

    accelerator.print("🧪 Running validation on common attacks...")

    _, best_acc = test_strategies_attack(
        test_loader=test_loader,
        vae=vae,
        msg_decoder=msg_decoder,
        localizer=localized,
        phi_dim=args.phi_dimension,
        accelerator=accelerator,
        global_step=global_step,
        lpips_net=lpips_net,
        args=args,
        attack_list=attack_list,
        max_batches=args.eval_max_batches,
        best_acc=best_acc,
        weight_dtype=weight_dtype,
        save_dir=save_dir,
    )

    return best_acc, best_acc_aigc


def main():
    args = parse_args()

    save_dir, logging_dir = prepare_experiment(args, train=False)
    accelerator = build_accelerator(args, logging_dir)

    if args.seed is not None:
        set_seed(args.seed)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    weight_dtype = get_weight_dtype(accelerator)

    vae, msg_decoder, localized = load_models(
        args=args,
        accelerator=accelerator,
        weight_dtype=weight_dtype,
    )

    test_loader = build_dataloaders_test(args)

    vae, msg_decoder, localized, test_loader = accelerator.prepare(
        vae,
        msg_decoder,
        localized,
        test_loader,
    )

    lpips_net = lpips.LPIPS(net="vgg").to(
        accelerator.device,
        dtype=weight_dtype,
    )

    attack_list, _ = build_attackers(
        args=args,
        device=accelerator.device,
        weight_dtype=weight_dtype,
    )

    pipe_inpaint = SimpleLama()

    global_step = 0
    best_acc = 0.0
    best_acc_aigc = 0.0

    save_val_dir = save_dir

    best_acc, best_acc_aigc = run_validation(
        args=args,
        test_loader=test_loader,
        vae=vae,
        msg_decoder=msg_decoder,
        localized=localized,
        accelerator=accelerator,
        global_step=global_step,
        lpips_net=lpips_net,
        pipe_inpaint=pipe_inpaint,
        attack_list=attack_list,
        weight_dtype=weight_dtype,
        best_acc=best_acc,
        best_acc_aigc=best_acc_aigc,
        save_dir=save_val_dir,
    )

    accelerator.print(f"✅ Validation finished.")
    accelerator.print(f"Best common attack acc: {best_acc}")
    accelerator.print(f"Best AIGC edit acc: {best_acc_aigc}")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()