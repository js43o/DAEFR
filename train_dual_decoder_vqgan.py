import argparse, os, sys, datetime
import torch
import torchvision
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm.auto import tqdm  # Progress bar

# Accelerate
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed

# 작성한 모듈 임포트 (경로는 환경에 맞게 유지)
from DAEFR.modules.vqvae.vqvae_arch import SharedEncDualDecVQModel
from DAEFR.modules.losses.vqperceptual import DualTaskVQLoss
from dataset_multipie import MultiPIEDataset


def save_sample_images(
    inputs, reconstructions, targets, step, save_dir, prefix="restoration"
):
    """
    학습 중 샘플 이미지를 그리드 형태로 저장하는 헬퍼 함수
    Input, Recon, Target 순서로 저장
    """

    # 텐서 범위가 [-1, 1]이라고 가정하고 [0, 1]로 정규화
    def norm(x):
        # return (x.clamp(-1, 1) + 1) / 2
        return x

    with torch.no_grad():
        # (B, C, H, W) -> Grid
        # 배치 내 첫 4장만 저장
        n_vis = min(inputs.shape[0], 4)

        # [Input, Recon, Target] 순서로 쌓기
        vis_list = []
        for i in range(n_vis):
            vis_list.append(norm(inputs[i]))
            vis_list.append(norm(reconstructions[i]))
            vis_list.append(norm(targets[i]))

        grid = torchvision.utils.make_grid(
            vis_list, nrow=3, padding=2
        )  # 3 columns: In, Out, GT
        save_path = os.path.join(save_dir, f"{prefix}_step_{step:06d}.png")
        torchvision.utils.save_image(grid, save_path)


def train(args):
    # -------------------------------------------------------------------------
    # 1. Accelerate Setup
    # -------------------------------------------------------------------------
    # mixed_precision="fp16" 등의 설정은 'accelerate config' CLI 명령어로 제어하거나
    # 여기서 Accelerator(mixed_precision='fp16')으로 강제할 수 있습니다.
    accelerator = Accelerator(
        gradient_accumulation_steps=args.accumulate_grad_batches,
        log_with="tensorboard",
        project_dir=args.log_dir,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
    )

    # 시드 고정 (모든 프로세스 동기화)
    set_seed(42)

    # Main process에서만 디렉토리 생성
    if accelerator.is_main_process:
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(os.path.join(args.log_dir, "samples"), exist_ok=True)
        os.makedirs(os.path.join(args.log_dir, "checkpoints"), exist_ok=True)

        # Config 백업
        OmegaConf.save(
            OmegaConf.load(args.config_path), os.path.join(args.log_dir, "config.yaml")
        )

    # -------------------------------------------------------------------------
    # 2. Load Configuration & Initialize
    # -------------------------------------------------------------------------
    config = OmegaConf.load(args.config_path)
    ddconfig = config.model.params.ddconfig.params
    loss_config = config.model.params.lossconfig.params

    # Model Init
    model = SharedEncDualDecVQModel(ddconfig)  # .to(device) 제거 (Prepare가 처리)

    # Loss Init (Discriminators 포함)
    loss_module = DualTaskVQLoss(
        loss_config,
        disc_start=loss_config.disc_start,
        codebook_weight=loss_config.codebook_weight,
    )  # .to(device) 제거

    # Dataset & Loader
    # num_workers는 CPU 코어 수에 맞게 설정
    dataset = MultiPIEDataset(**config.data.params.train.params)
    dataloader = DataLoader(
        dataset,
        batch_size=config.data.params.batch_size,
        shuffle=True,
        num_workers=config.data.params.num_workers,
        # pin_memory=True,
    )
    
    print("🔥", len(dataloader))

    # Optimizers
    opt_g = torch.optim.Adam(
        model.parameters(), lr=config.model.base_learning_rate, betas=(0.5, 0.9)
    )
    opt_d = torch.optim.Adam(
        loss_module.parameters(), lr=config.model.base_learning_rate, betas=(0.5, 0.9)
    )

    # -------------------------------------------------------------------------
    # 3. Accelerate Prepare
    # -------------------------------------------------------------------------
    # 모든 모듈을 accelerator가 관리하도록 래핑
    model, loss_module, opt_g, opt_d, dataloader = accelerator.prepare(
        model, loss_module, opt_g, opt_d, dataloader
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("daefr_experiment")

    global_step = 0
    total_steps = config.model.max_epochs * len(dataloader)

    # Progress Bar (Main process only)
    progress_bar = tqdm(total=total_steps, disable=not accelerator.is_main_process)

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    for epoch in range(config.model.max_epochs):
        model.train()
        loss_module.train()

        for batch in dataloader:
            # Data Load (Accelerator가 device 할당 자동 처리)
            img_lq_nf = batch[0]  # LQ Non-Frontal
            img_lq_f = batch[1]  # LQ Frontal
            img_hq_nf = batch[2]  # HQ Non-Frontal

            # ====================================================
            # Phase 1: Generator Training
            # ====================================================
            # Gradient Accumulation Context
            with accelerator.accumulate(model):
                opt_g.zero_grad()

                # Forward Pass
                # output: (rec_restore, rec_frontal, quant_loss)
                out = model(img_lq_nf)
                rec_restore, rec_frontal, quant_loss = out[0], out[1], out[2]

                # [중요] DDP 사용 시 model은 wrapper로 감싸져 있으므로,
                # 커스텀 메서드 접근 시 unwrap하거나 .module 등을 써야 함.
                # 하지만 매번 unwrap은 비효율적이므로, forward시 리턴받도록 모델을 수정하거나
                # 아래처럼 unwrap_model을 사용해 접근합니다.
                raw_model = accelerator.unwrap_model(model)

                loss_g, log_g = loss_module(
                    codebook_loss=quant_loss,
                    inputs_lq_nf=img_lq_nf,
                    recons_restore=rec_restore,
                    target_hq_nf=img_hq_nf,
                    recons_frontal=rec_frontal,
                    target_lq_f=img_lq_f,
                    optimizer_idx=0,
                    global_step=global_step,
                    last_layer_restore=raw_model.get_last_layer_restoration(),  # Unwrapped model access
                    last_layer_frontal=raw_model.get_last_layer_frontalization(),
                    split="train",
                )

                # Backward
                accelerator.backward(loss_g)
                opt_g.step()

            # ====================================================
            # Phase 2: Discriminator Training
            # ====================================================
            with accelerator.accumulate(loss_module):
                opt_d.zero_grad()

                # Discriminator Forward (requires detach)
                loss_d, log_d = loss_module(
                    codebook_loss=quant_loss,
                    inputs_lq_nf=img_lq_nf,
                    recons_restore=rec_restore.detach(),
                    target_hq_nf=img_hq_nf,
                    recons_frontal=rec_frontal.detach(),
                    target_lq_f=img_lq_f,
                    optimizer_idx=1,
                    global_step=global_step,
                    last_layer_restore=raw_model.get_last_layer_restoration(),
                    last_layer_frontal=raw_model.get_last_layer_frontalization(),
                    split="train",
                )

                accelerator.backward(loss_d)
                opt_d.step()

            # ====================================================
            # Logging & Sampling
            # ====================================================
            progress_bar.update(1)
            global_step += 1

            if accelerator.is_main_process:
                # 1. Logging
                if global_step % args.log_freq == 0:
                    logs = {
                        "loss/g_total": loss_g.item(),
                        "loss/d_total": loss_d.item(),
                        "loss/rec_restore": log_g["train/rec_loss_restore"].item(),
                        "loss/rec_frontal": log_g["train/rec_loss_frontal"].item(),
                        "epoch": epoch,
                    }
                    accelerator.log(logs, step=global_step)
                    progress_bar.set_postfix(**logs)

                # 2. Image Sampling
                if global_step % args.sample_freq == 0:
                    save_dir = os.path.join(args.log_dir, "samples")

                    # Restoration Task Sample (LQ NF -> HQ NF)
                    save_sample_images(
                        img_lq_nf,
                        rec_restore,
                        img_hq_nf,
                        global_step,
                        save_dir,
                        prefix="restore",
                    )

                    # Frontalization Task Sample (LQ NF -> LQ F)
                    save_sample_images(
                        img_lq_nf,
                        rec_frontal,
                        img_lq_f,
                        global_step,
                        save_dir,
                        prefix="frontal",
                    )

        # ====================================================
        # Epoch End: Checkpointing
        # ====================================================
        accelerator.wait_for_everyone()  # 모든 GPU 동기화
        if accelerator.is_main_process:
            save_path = os.path.join(args.log_dir, "checkpoints", f"epoch_{epoch:04d}")
            # 전체 상태 저장 (Model, Optimizer, LR Scheduler 포함)
            accelerator.save_state(save_path)
            print(f"Saved checkpoint to {save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, default="configs/Dual_Decoder_VQGAN.yaml"
    )
    parser.add_argument("--log_dir", type=str, default="experiments/02")
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument(
        "--sample_freq",
        type=int,
        default=100,
        help="Iterations between saving sample images",
    )
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    args = parser.parse_args()

    if not os.path.exists(args.config_path):
        print(f"Warning: Config file {args.config_path} not found.")

    train(args)
