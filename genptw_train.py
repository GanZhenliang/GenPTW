import argparse
import os
os.environ["LAMA_MODEL"] = "/root/GenPTW/big-lama.pt"
import ssl
import lpips
from lama import SimpleLama
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from JND import JND
from train_utils import *
from attack_methods.attack import *


ssl._create_default_https_context = ssl._create_unverified_context
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["WANDB_API_KEY"] = ""
os.environ["WANDB_MODE"] = "offline"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def parse_args():
    parser = argparse.ArgumentParser("Train GenPTW watermark adapter/localizer.")

    # Experiment
    parser.add_argument("--exp_name", type=str, default="1-1aaai_1sd_1final")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-fs/output")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--seed", type=int, default=2026)

    # Data
    parser.add_argument("--train_img_dir", type=str, default="/autodl-tmp/Datasets/COCO2017/train2017")
    parser.add_argument("--train_ann_file", type=str, default="/autodl-tmp/Datasets/COCO2017/annotations/instances_train2017.json")
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
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False)
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
    parser.add_argument("--eval_max_batches", type=int, default=5)
    parser.add_argument("--eval_strength", type=float, default=1.0)

    return parser.parse_args()


def apply_edit_attack(aug_img, gen_img_old, mask_coco, vae, pipe_inpaint, args, accelerator):
    bsz = aug_img.shape[0]
    mask = make_empty_mask(bsz, args, accelerator.device)

    if torch.rand(1).item() > args.edit_attack_prob:
        return aug_img, mask

    prop = torch.rand(1).item()
    unwrapped_vae = accelerator.unwrap_model(vae)

    #  无水印图替换
    if prop <= 0.25:
        aug_img = apply_mask_attack_ultra(aug_img, gen_img_old, mask_coco)
        mask = mask_coco
    # 局部vae
    elif prop <= 0.50:
        latent_a = unwrapped_vae.encode(aug_img).latent_dist.sample() * 0.18215
        vae_img = unwrapped_vae.old_decode(1 / 0.18215 * latent_a).sample
        aug_img = apply_mask_attack_ultra(aug_img, vae_img, mask_coco)
        mask = mask_coco
    # lama
    elif prop <= 0.75:
        inp_img = pipe_inpaint(normalize_img(aug_img), mask_coco).view(bsz, 3, args.resolution, args.resolution)
        aug_img = (inp_img - 0.5) * 2
        mask = mask_coco
    # 全图vae重构
    elif prop <= 0.80:
        latent_a = unwrapped_vae.encode(aug_img).latent_dist.sample() * 0.18215
        aug_img = unwrapped_vae.old_decode(1 / 0.18215 * latent_a).sample
        mask = make_full_mask(bsz, args, accelerator.device)
    # 全图vae重构 + 局部篡改为无水印的
    else:
        latent_a = unwrapped_vae.encode(aug_img).latent_dist.sample() * 0.18215
        vae_img = unwrapped_vae.old_decode(1 / 0.18215 * latent_a).sample
        aug_img = apply_mask_attack_ultra(vae_img, gen_img_old, mask_coco)
        mask = make_full_mask(bsz, args, accelerator.device)

    return aug_img, mask


def apply_common_attacks(aug_img, attackers, args, global_step):
    if torch.rand(1).item() <= args.attack_prob_color:
        aug_img = attackers["color"](aug_img, global_step)
    if torch.rand(1).item() <= args.attack_prob_noise:
        aug_img = attackers["noise"](aug_img, global_step)
    if torch.rand(1).item() <= args.attack_prob_jpeg:
        aug_img = attackers["jpeg"](aug_img, global_step)
    return aug_img


# def get_loss_weights(error_rate, mask_acc):
#     if error_rate > 0.98 and mask_acc > 0.94:
#         return [5, 10, 5, 1, 20, 15, 0.05, 0.05, 0.05, 0.05]
#     return [5, 1, 0.5, 1, 20, 1.5, 0.005, 0.005, 0.005, 0.005]

def get_loss_weights(error_rate, mask_acc):
    if error_rate > 0.98 and mask_acc > 0.94:
        return [5, 10, 5, 1, 20, 15]
    return [5, 1, 0.5, 1, 20, 1.5]


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
        save_dir = save_dir,
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
        save_dir = save_dir,
    )
    return best_acc, best_acc_aigc


def train_one_step(
    args,
    batch,
    vae,
    msg_decoder,
    localized,
    pipe_inpaint,
    attackers,
    attenuation,
    edge_gen,
    lpips_net,
    optimizer,
    lr_scheduler,
    accelerator,
    weight_dtype,
    global_step,
    decode_resize,
):
    vae.train()
    msg_decoder.train()
    localized.train()
    pipe_inpaint.model.train()

    pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
    mask_coco = batch["masks"].to(accelerator.device, dtype=weight_dtype).clamp(0, 1)
    unwrapped_vae = accelerator.unwrap_model(vae)

    latents = unwrapped_vae.encode(pixel_values).latent_dist.sample() * 0.18215
    bsz = latents.shape[0]

    with accelerator.autocast():
        gen_img_old, zo0, zo1, zo2, zo3 = unwrapped_vae.decode_plain(1 / 0.18215 * latents, return_dict=False)
        gen_img_old = decode_resize(gen_img_old.clamp(-1, 1))
        gen_img_old_nor = normalize_img(gen_img_old)

        phis = torch.bernoulli(torch.empty(bsz, args.phi_dimension).uniform_(0, 1)).to(
            accelerator.device, dtype=weight_dtype
        )
        gen_img, z0, z1, z2, z3 = unwrapped_vae.decode_wm(1 / 0.18215 * latents, phis, return_dict=False)
        gen_img = decode_resize(gen_img.clamp(-1, 1))
        gen_img_nor = normalize_img(gen_img)

        # loss_z0 = F.mse_loss(z0, zo0).mean()
        # loss_z1 = F.mse_loss(z1, zo1).mean()
        # loss_z2 = F.mse_loss(z2, zo2).mean()
        # loss_z3 = F.mse_loss(z3, zo3).mean()


        hmaps = attenuation(gen_img_old_nor).detach()
        cost = torch.exp(-8.0 * hmaps)
        loss_jnd = (cost * torch.abs(gen_img_nor - gen_img_old_nor)).mean()

        aug_img, mask = apply_edit_attack(gen_img, gen_img_old, mask_coco, vae, pipe_inpaint, args, accelerator)
        aug_img = apply_common_attacks(aug_img, attackers, args, global_step)

        maybe_empty_cuda_cache()

        detected_bits, wmf = msg_decoder(aug_img, mask, global_step)
        out = localized(aug_img, mask, wmf)

        maybe_empty_cuda_cache()

        loss_lpips = lpips_net(gen_img, gen_img_old).mean()
        loss_rec = F.mse_loss(gen_img, gen_img_old).mean()
        loss_key = F.binary_cross_entropy(detected_bits, phis).mean()
        edge = edge_gen(mask)
        loss_edge = F.binary_cross_entropy_with_logits(
            input=out["pred_mask_toloss"],
            target=mask,
            weight=edge,
        )
        loss_mask = out["backward_loss"]

        with torch.no_grad():
            metrics = compute_metrics(
                phis, detected_bits, gen_img, gen_img_old, gen_img_nor, gen_img_old_nor, out, mask
            )

        loss_weights = get_loss_weights(metrics["error_rate"], metrics["mask_acc"])
        loss = sum(
            weight * term
            for weight, term in zip(
                loss_weights,
                # [loss_key, loss_lpips, loss_rec, loss_mask, loss_edge, loss_jnd, loss_z0, loss_z1, loss_z2, loss_z3],
                [loss_key, loss_lpips, loss_rec, loss_mask, loss_edge, loss_jnd],
                # [5, 1, 0.5, 1, 20, 1.5]
            )
        )

    maybe_empty_cuda_cache()

    accelerator.backward(loss)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    logs = {
        "ac": metrics["error_rate"],
        "ma": metrics["mask_acc"].item(),
        "so": metrics["ssim"].item(),
        "pr": metrics["psnr"].item(),
        "iou": metrics["iou"].item(),
        "auc": metrics["auc"].item(),
        "f1": metrics["f1"].item(),
        "ls": loss.item(),
        "k": loss_key.item(),
        "fid": loss_lpips.item(),
        # "sifid": metrics["sifid"].item(),
        "lc": loss_rec.item(),
        "lj": loss_jnd.item(),
        "mk": loss_mask.item(),
        # "l0z": loss_z0.item(),
        "lr": lr_scheduler.get_last_lr()[0],
        "w": str(loss_weights),
    }
    sample_state = (mask, out, edge, gen_img_nor, gen_img_old_nor, aug_img, hmaps, cost)
    return logs, sample_state


def main():
    args = parse_args()
    save_dir, logging_dir = prepare_experiment(args)
    accelerator = build_accelerator(args, logging_dir)

    if args.seed is not None:
        set_seed(args.seed)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    weight_dtype = get_weight_dtype(accelerator)
    vae, msg_decoder, localized = load_models(args, accelerator, weight_dtype)
    optimizer = build_optimizer(args, vae, msg_decoder, localized)
    train_loader, test_loader = build_dataloaders(args)
    lr_scheduler = build_lr_scheduler(args, optimizer, train_loader)

    vae, msg_decoder, localized, optimizer, train_loader, test_loader, lr_scheduler = accelerator.prepare(
        vae, msg_decoder, localized, optimizer, train_loader, test_loader, lr_scheduler
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("fuser_train", config=vars(args))

    lpips_net = lpips.LPIPS(net="vgg").to(accelerator.device, dtype=weight_dtype)
    attenuation = JND(in_channels=3, out_channels=3, blue=False).to(accelerator.device, dtype=weight_dtype)
    edge_gen = EdgeGenerator(kernel_size=3).to(accelerator.device, dtype=weight_dtype)
    attack_list, attackers = build_attackers(args, accelerator.device, weight_dtype)

    pipe_inpaint = SimpleLama()
    decode_resize = torch.nn.Identity()

    global_step = 0
    first_epoch = 0
    best_acc = 0.0
    best_acc_aigc = 0.0

    for epoch in range(first_epoch, args.num_train_epochs):
        progress_bar = tqdm(train_loader, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch} Training")

        for step, batch in enumerate(progress_bar):
            if global_step >= args.max_train_steps:
                break

            with accelerator.accumulate(accelerator.unwrap_model(vae).decoder), accelerator.accumulate(
                msg_decoder
            ), accelerator.accumulate(localized):
                logs, sample_state = train_one_step(
                    args=args,
                    batch=batch,
                    vae=vae,
                    msg_decoder=msg_decoder,
                    localized=localized,
                    pipe_inpaint=pipe_inpaint,
                    attackers=attackers,
                    attenuation=attenuation,
                    edge_gen=edge_gen,
                    lpips_net=lpips_net,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    accelerator=accelerator,
                    weight_dtype=weight_dtype,
                    global_step=global_step,
                    decode_resize=decode_resize,
                )

            accelerator.log(logs, step=global_step)
            progress_bar.set_postfix(**logs)

            global_step += 1
            if global_step >= args.max_train_steps:
                break

            if global_step % args.eval_interval == 0 and accelerator.is_main_process:

                save_steps_dir = os.path.join(save_dir, f"Train_{global_step}")
                os.makedirs(save_steps_dir, exist_ok=True)

                save_train_dir = os.path.join(save_steps_dir, f"train")
                os.makedirs(save_train_dir, exist_ok=True)
                save_training_samples(save_train_dir, global_step, *sample_state)

                save_val_dir = os.path.join(save_steps_dir, f"val")
                os.makedirs(save_val_dir, exist_ok=True)
                
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
                    save_dir = save_steps_dir
                )
                
                

        maybe_empty_cuda_cache()

    accelerator.wait_for_everyone()
    accelerator.end_training()
    wandb.finish()


if __name__ == "__main__":
    main()