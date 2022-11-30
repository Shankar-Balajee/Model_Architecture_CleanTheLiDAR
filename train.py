import random
import numpy as np
import torch
import utils
import dataset
import transforms
from torch.utils.data import DataLoader, DistributedSampler

from torch.nn.parallel import DistributedDataParallel
import os
import json
import wandb
# import CleanPoint.cleanpoint as cleanpoint
import cleanpoint
from Chamfer3D.dist_chamfer_3D import chamfer_3DDist

import argparse
from datetime import datetime
from pointnet2_ops import pointnet2_utils
import yaml


class ChamferLoss:
    def __init__(self):
        self.kernel = chamfer_3DDist()

    def __call__(self, pc1, pc2):
        dist1, dist2, _, _ = self.kernel(pc1, pc2)
        return torch.mean(dist1) + torch.mean(dist2)

# class EmdLoss:
#     def __init__(self):
#         self.kernel = EMD()

#     def __call__(self, pc1, pc2):
#         loss = self.kernel(pc1, pc2)
#         return loss.mean()
    
def index(pc, indx):
    device = pc.device
    B = pc.shape[0]
    view_shape = list(indx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(indx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    )
    new_pc = pc[batch_indices, indx, :]
    return new_pc

def fps(pc, n_pts, indx_only=False):
    fps_indx = pointnet2_utils.furthest_point_sample(pc, n_pts).long()
    if indx_only:
        return fps_indx
    pc = index(pc, fps_indx)
    return pc

class Trainer:
    def __init__(self, args):
        self.args = args

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        
        self.device, local_rank = utils.setup_device(args.dist)
        self.main_thread = True if local_rank == 0 else False
        if self.main_thread:
            print(f"\nsetting up device, args.dist = {args.dist}")
        print(f" | {self.device}")
        
        # train_transform = transforms.Compose(
        #     [
        #         transforms.RandomPermute(),
        #         transforms.RandomMirror(),
        #         transforms.RandomScale(args.scale_low, args.scale_high),
        #         transforms.ToTensor(),
        #     ]
        # )
        # val_transform = transforms.Compose([transforms.ToTensor()])
        
        if args.dset == "shapenetchairs":
            train_dset = dataset.ShapeNetChairs(
                root=args.data_root,
                split="train",
                # transform=train_transform,
            )
            val_dset = dataset.ShapeNetChairs(
                root=args.data_root,
                split="test",
                # transform=val_transform,
            )
        else:
            raise ValueError(f"args.dset = {args.dset} not implemented")

        if self.main_thread:
            print(f"setting up dataset, train: {len(train_dset)}, val: {len(val_dset)}")
        
        if args.dist:
            train_sampler = DistributedSampler(train_dset)
            self.train_loader = DataLoader(
                train_dset,
                batch_size=args.batch_size,
                sampler=train_sampler,
                num_workers=args.n_workers,
            )
        else:
            self.train_loader = DataLoader(
                train_dset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.n_workers,
            )
        self.val_loader = DataLoader(
            val_dset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.n_workers,
        )
        # with open('pt_model.yaml') as f:
        #     cfg = yaml.safe_load(f)
        if args.net == "cleanpoint":
            model = cleanpoint.Model()
        else:
            raise ValueError(f"args.net = {args.net} not implemented")
        
        if args.dist:
            torch.set_num_threads(1)
            self.model = DistributedDataParallel(
                model.to(self.device),
                device_ids=[local_rank],
                output_device=local_rank,
            )
        else:
            self.model = model.to(self.device)
        
        if self.main_thread:
            print(f"# of model parameters: {sum(p.numel() for p in self.model.parameters())/1e6}M")
        
        # self.p_loss_model = get_model(num_classes=50, normal_channel=False).to(self.device)
        # self.p_loss_model.load_state_dict(torch.load('pointnet2_part_seg_msg/checkpoints/best_model.pth', map_location=self.device), strict=False)
        # for param in self.p_loss_model.parameters():
        #     param.requires_grad = False
        # self.perceptual_weight = args.perceptual_weight
        self.chamfer_criterion = ChamferLoss()
        
        if args.optim == "sgd":
            self.optim = torch.optim.SGD(
                self.model.parameters(),
                lr=args.lr,
                momentum=0.9,
                weight_decay=args.weight_decay,
            )
        elif args.optim == "adam":
            self.optim = torch.optim.Adam(
                self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
        elif args.optim == "adamw":
            self.optim = torch.optim.AdamW(
                self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
        else:
            raise ValueError(f"args.optim = {args.optim} not implemented")

        if self.args.lr_step_mode == "epoch":
            total_steps = args.epochs - args.warmup
        else:
            total_steps = int(args.epochs * len(self.train_loader) - args.warmup)
        if args.warmup > 0:
            for group in self.optim.param_groups:
                group["lr"] = 1e-12 * group["lr"]
        if args.lr_sched == "cosine":
            self.lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, total_steps)
        elif args.lr_sched == "multi_step":
            milestones = [
                int(milestone) - total_steps for milestone in args.lr_decay_step.split(",")
            ]
            self.lr_sched = torch.optim.lr_scheduler.MultiStepLR(
                self.optim, milestones=milestones, gamma=args.lr_decay
            )
        elif args.lr_sched == "step":
            self.lr_sched = torch.optim.lr_scheduler.StepLR(
                self.optim, step_size=args.lr_decay_step, gamma=args.lr_decay,
            )
        else:
            raise ValueError(f"args.lr_sched = {args.lr_sched} not implemented")

        if os.path.exists(os.path.join(args.out_dir, "last.ckpt")):
            if args.resume == False and self.main_thread:
                raise ValueError(
                    f"directory {args.out_dir} already exists, change output directory or use --resume argument"
                )
            ckpt = torch.load(os.path.join(args.out_dir, "last.ckpt"), map_location=self.device)
            model_dict = ckpt["model"]
            if "module" in list(model_dict.keys())[0] and args.dist == False:
                model_dict = {
                    key.replace("module.", ""): value for key, value in model_dict.items()
                }
            self.model.load_state_dict(model_dict)
            self.optim.load_state_dict(ckpt["optim"])
            self.lr_sched.load_state_dict(ckpt["lr_sched"])
            self.start_epoch = ckpt["epoch"] + 1
            if self.main_thread:
                print(
                    f"loaded checkpoint, resuming training expt from {self.start_epoch} to {args.epochs} epochs."
                )
        else:
            if args.resume == True and self.main_thread:
                raise ValueError(
                    f"resume training args are true but no checkpoint found in {args.out_dir}"
                )
            os.makedirs(args.out_dir, exist_ok=True)
            with open(os.path.join(args.out_dir, "args.txt"), "w") as f:
                json.dump(args.__dict__, f, indent=4)
            self.start_epoch = 0
            if self.main_thread:
                print(f"starting fresh training expt for {args.epochs} epochs.")
        self.train_steps = self.start_epoch * len(self.train_loader)

        self.log_wandb = False
        self.metric_meter = utils.AvgMeter()
        if self.main_thread:
            self.log_f = open(os.path.join(args.out_dir, "logs.txt"), "w")
            print(f"start file logging @ {os.path.join(args.out_dir, 'logs.txt')}")
            if args.wandb:
                self.log_wandb = True
                run = wandb.init(project="CleanPoint")
                print(f"start wandb logging @ {run.get_url()}")
                self.log_f.write(f"\nwandb url @ {run.get_url()}\n")
    
    def train_epoch(self):
        self.metric_meter.reset()
        self.model.train()
        for indx, complete in enumerate(self.train_loader):
            complete = complete.to(self.device)
            if self.args.n_pts < complete.shape[1]:
                complete = utils.sample_pc(complete, self.args.n_pts)
            noise = torch.rand(1)*0.05 + 0.05
            noisy_pc = complete + torch.randn(complete.shape).to("cuda")*noise.to("cuda")
            noisy_pc = noisy_pc.transpose(2, 1)
            
            coarse, pred = self.model(noisy_pc)
            
            loss_chamfer1 = self.chamfer_criterion(pred, complete)
            loss_chamfer2 = self.chamfer_criterion(coarse, complete)
            total_loss = (loss_chamfer1 + loss_chamfer2)/2
            # if self.args.emd_lam > 0:
            #     loss_emd1 = self.emd_criterion(pred, complete)
            #     # loss_emd2 = self.emd_criterion(pred2, complete)
            #     total_loss = total_loss + self.args.emd_lam * loss_emd1# + loss_emd2)
                
            self.optim.zero_grad()
            total_loss.backward()
            self.optim.step()

            metrics = {
                "total_train_loss": total_loss.item(),
                "train_chamfer" : total_loss.item(),
                "train_chamfer1" : loss_chamfer1.item(),
                "train_chamfer2" : loss_chamfer2.item(),
                # 'train_emd': self.args.emd_lam * loss_emd1.item(),
                # "train_perceptual_loss": p_loss.item() * args.perceptual_weight
                }
            self.metric_meter.add(metrics)

            if self.main_thread and indx % self.args.log_every == 0:
                if self.log_wandb:
                    wandb.log({"train step": self.train_steps, **metrics})
                utils.pbar(indx / len(self.train_loader), msg=self.metric_meter.msg())

            if self.args.lr_step_mode == "step":
                if self.train_steps < self.args.warmup and self.args.warmup > 0:
                    self.optim.param_groups[0]["lr"] = (
                        self.train_steps / (self.args.warmup) * self.args.lr
                    )
                else:
                    self.lr_sched.step()

            self.train_steps += 1
        if self.main_thread:
            utils.pbar(1, msg=self.metric_meter.msg())

    @torch.no_grad()
    def eval(self):
        self.metric_meter.reset()
        self.model.eval()
        n_vis = 0
        for indx, complete in enumerate(self.val_loader):
            complete = complete.to(self.device)
            noise = torch.rand(1)*0.05 + 0.05
            noisy_pc = complete + torch.randn(complete.shape).to("cuda")*noise.to("cuda")
            noisy_pc_2048 = noisy_pc
            if self.args.n_pts < noisy_pc.shape[1]:
                noisy_pc = utils.sample_pc(noisy_pc, self.args.n_pts)
            noisy_pc = noisy_pc.transpose(2, 1)
            
            coarse, pred = self.model(noisy_pc)
        
            loss_chamfer1 = self.chamfer_criterion(pred, complete)
            loss_chamfer2 = self.chamfer_criterion(coarse, complete)
            total_loss = (loss_chamfer1 + loss_chamfer2)/2
            
            # if self.args.emd_lam > 0:
            #     loss_emd1 = self.emd_criterion(pred, complete)
            #     total_loss = total_loss + self.args.emd_lam * loss_emd1
               
            metrics = {
                "total_val_loss": total_loss.item(),
                "val_chamfer" : total_loss.item(),
                "val_chamfer1" : loss_chamfer1.item(),
                "val_chamfer2" : loss_chamfer2.item(),
                # "val_emd": self.args.emd_lam * loss_emd1.item(), 
                # "val_emb_loss": emb_loss.item() * args.emb_loss_weight,
                # "val_perceptual_loss": p_loss.item() * args.perceptual_weight
                
            }
            
            self.metric_meter.add(metrics)
            
            if self.main_thread:
                if self.log_wandb and random.random() < 0.8 and n_vis < self.args.n_vis:
                    vis_indx = random.randint(0, complete.shape[0] - 1)
                    vis = []
                    vis.append(complete[vis_indx].cpu().detach().numpy())
                    
                    temp = noisy_pc_2048[vis_indx].cpu().detach().numpy()
                    temp[:, 0] += 2
                    vis.append(temp)
                    
                    # temp = coarse[vis_indx].cpu().detach().numpy()
                    # temp[:, 0] += 4
                    # vis.append(temp)
                    
                    temp = pred[vis_indx].cpu().detach().numpy()
                    temp[:, 0] += 4
                    vis.append(temp)
                    
                    wandb.log(
                        {f"sample_{n_vis}": wandb.Object3D(np.concatenate(vis, axis=0), axis=0)}
                    )
                    n_vis += 1

                if indx % self.args.log_every == 0:
                    utils.pbar(indx / len(self.val_loader), msg=self.metric_meter.msg())

        # self.log_f.write(str(list(info[2].detach().cpu().numpy())))
        # self.log_f.flush()

        if self.main_thread:
            utils.pbar(1, msg=self.metric_meter.msg())

    def train(self):
        best_train, best_val = float("inf"), float("inf")

        for epoch in range(self.start_epoch, self.args.epochs):
            if self.main_thread:
                print(f"\nepoch: {epoch}")
                print("---------------")

            self.train_epoch()

            if self.main_thread:
                train_metrics = self.metric_meter.get()
                if train_metrics["train_chamfer"] < best_train:
                    print(
                        "\x1b[34m"
                        + f"train chamfer improved from {round(best_train, 5)} to {round(train_metrics['train_chamfer'], 5)}"
                        + "\033[0m"
                    )
                    best_train = train_metrics["train_chamfer"]
                msg = f"epoch: {epoch}, last train: {round(train_metrics['train_chamfer'], 5)}, best train: {round(best_train, 5)}"

                val_metrics = {}
                if epoch % self.args.eval_every == 0:
                    self.eval()
                    val_metrics = self.metric_meter.get()
                    if val_metrics["val_chamfer"] < best_val:
                        print(
                            "\x1b[33m"
                            + f"val chamfer improved from {round(best_val, 5)} to {round(val_metrics['val_chamfer'], 5)}"
                            + "\033[0m"
                        )
                        best_val = val_metrics["val_chamfer"]
                        torch.save(
                            self.model.state_dict(),
                            os.path.join(self.args.out_dir, f"best.ckpt"),
                        )
                    msg += f", last val: {round(val_metrics['val_chamfer'], 5)}, best val: {round(best_val, 5)}"

                print(msg)
                self.log_f.write(msg + f", lr: {round(self.optim.param_groups[0]['lr'], 5)}\n")
                
                self.log_f.flush()

                if self.log_wandb:
                    train_metrics = {"epoch " + key: value for key, value in train_metrics.items()}
                    val_metrics = {"epoch " + key: value for key, value in val_metrics.items()}
                    wandb.log(
                        {
                            "epoch": epoch,
                            **train_metrics,
                            **val_metrics,
                            "lr": self.optim.param_groups[0]["lr"],
                        }
                    )

                torch.save(
                    {
                        "model": self.model.state_dict(),
                        "optim": self.optim.state_dict(),
                        "lr_sched": self.lr_sched.state_dict(),
                        "epoch": epoch,
                    },
                    os.path.join(self.args.out_dir, "last.ckpt"),
                )

            if self.args.lr_step_mode == "epoch":
                if epoch <= self.args.warmup and self.args.warmup > 0:
                    self.optim.param_groups[0]["lr"] = epoch / self.args.warmup * self.args.lr
                else:
                    self.lr_sched.step()     
            

if __name__ == "__main__":

    # wandb.init(project="CLIPointEditor")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out_dir",
        type=str,
        default=f"output/{datetime.now().strftime('%Y-%m-%d_%H-%M')}",
        help="path to output directory [def: output/year-month-date_hour-minute]",
    )
    parser.add_argument("--seed", type=int, default=1486, help="set experiment seed [def: 42]")
    parser.add_argument(
        "--dist", action="store_true", help="start distributed training [def: false]"
    )
    parser.add_argument(
        "--n_pts", type=int, default=512, help="set number of input points [def: 1024]"
    )
    parser.add_argument("--dset", type=str, default="shapenetchairs", help="dataset name [def: shapenetchairs]")
    parser.add_argument("--data_root", type=str, required=True, help="dataset directory")
    parser.add_argument("--batch_size", type=int, default=24, help="batch size [def: 16]")
    parser.add_argument(
        "--n_workers", type=int, default=4, help="number of workers for dataloading [def: 4]"
    )
    parser.add_argument("--net", type=str, default="point_mae", help="network name [def: pcn]")
    
    parser.add_argument("--optim", type=str, default="adam", help="optimizer name [def: adam]")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam learning rate [def: 0.0001]")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="sgd optimizer weight decay [def: 0.0001]"
    )

    parser.add_argument(
        "--lr_step_mode",
        type=str,
        default="epoch",
        help="choose lr step mode, one of [epoch, step] [def: epoch]",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="lr warmup in epochs/steps based on epoch step mode [def: 0]",
    )

    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs [def: 300]")
    parser.add_argument(
        "--lr_sched", type=str, default="step", help="lr scheduler name [def: step]"
    )

    parser.add_argument(
        "--lr_decay_step",
        type=str,
        default=50,
        help="multi step lr scheduler milestones [def: 120, 160]",
    )
    parser.add_argument(
        "--lr_decay",
        type=float,
        default=0.7,
        help="multi step lr scheduler decay gamma [def: 0.1]",
    )
    parser.add_argument(
        "--resume", action="store_true", help="resume training from checkpoint [def: false]"
    )
    parser.add_argument("--wandb", action="store_true", help="start wandb logging [def: false]")
    parser.add_argument("--eval_every", type=int, default=1, help="eval frequency [def: 1]")
    parser.add_argument("--log_every", type=int, default=1, help="logging frequency [def: 1]")
    parser.add_argument(
        "--emd_lam", type=float, default= 1.0, help="emd loss lambda [def: 0]"
    )
    parser.add_argument(
        "--n_vis", type=int, default=10, help="num of pcs to visualize during eval [def: 5]"
    )
    parser.add_argument(
        "--nneighbor", type=int, default=16, help="num of pcs to visualize during eval [def: 5]"
    )
    parser.add_argument(
        "--nblocks", type=int, default=4, help="num of pcs to visualize during eval [def: 5]"
    )
    parser.add_argument(
        "--transformer_dim", type=int, default=512, help="num of pcs to visualize during eval [def: 5]"
    )
    parser.add_argument(
        "--num_class", type=int, default=3, help="num of pcs to visualize during eval [def: 5]"
    )
    parser.add_argument(
        "--input_dim", type=int, default=3, help="num of pcs to visualize during eval [def: 5]"
    )
    parser.add_argument(
        "--perceptual_weight", type=float, default=0.0, help="num of pcs to visualize during eval [def: 5]"
    )
    parser.add_argument(
        "--emb_loss_weight", type=float, default=1.0, help="num of pcs to visualize during eval [def: 5]"
    )
    
    args = parser.parse_args()
    utils.print_args(args)

    trainer = Trainer(args)
    trainer.train()

    if args.dist:
        torch.distributed.destroy_process_group()