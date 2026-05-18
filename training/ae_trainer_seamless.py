import torch
import smplx
import numpy as np
from optimizers.loss_factory import GeodesicLoss, get_loss_func
from data_processing.utils import get_smplh_model
from utils import rotation_conversions as rc
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
from models_dyadit.vqvae_downsampled import TemporalVertexCodecDownsampled

class SeamlessStage1Trainer:
    def __init__(self, args, gpu_id):
        self.args = args
        self.device = torch.device(f"cuda:{gpu_id}")

        m = args.model_params
        self.model = TemporalVertexCodecDownsampled(
            n_vertices=m["nb_joints"],
            latent_dim=m["output_emb_width"],
            categories=m["code_dim"],
            residual_depth=m["depth"],
            downsample_factor=m["downsample_factor"],
        ).to(self.device)

        self.model = DDP(self.model, device_ids=[gpu_id])

        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.training_params.lr_base)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, 
            mode='min', 
            factor=0.75,
            patience=999, 
            verbose=True  
        )

        self.smplh = get_smplh_model(self.device, 'male').eval()

        if args.training_params.rec_mse:
            self.rec_loss = torch.nn.MSELoss(reduction='mean')
        else:
            self.rec_loss = GeodesicLoss()

        self.vel_loss = torch.nn.L1Loss(reduction='mean')
        self.ver_loss = torch.nn.MSELoss(reduction='mean')

    def train_step(self, dict_data):
        self.model.train()
        self.opt.zero_grad()

        target_pose = dict_data["pose"].to(self.device, non_blocking=True)
        batch_size, frames = target_pose.shape[0], target_pose.shape[1]
        target_pose_matrix = rc.axis_angle_to_matrix(target_pose.reshape(batch_size, frames, 43, 3))
        target_pose_6d = rc.matrix_to_rotation_6d(target_pose_matrix).reshape(batch_size, frames, 258)

        reconstructed_pose, loss_commit, perplexity = self.model(target_pose_6d)

        reconstructed_pose_matrix = rc.rotation_6d_to_matrix(reconstructed_pose.reshape(batch_size, frames, 43, 6))
        
        loss_rec = self.rec_loss(reconstructed_pose_matrix, target_pose_matrix) * self.args.loss_weights.rec_weight

        loss_vel = self.vel_loss(reconstructed_pose_matrix[:, 1:] - reconstructed_pose_matrix[:, :-1], target_pose_matrix[:, 1:] - target_pose_matrix[:, :-1])
        loss_acc = self.vel_loss(reconstructed_pose_matrix[:, 2:] + reconstructed_pose_matrix[:, :-2] - 2 * reconstructed_pose_matrix[:, 1:-1], target_pose_matrix[:, 2:] + target_pose_matrix[:, :-2] - 2 * target_pose_matrix[:, 1:-1])
        
        loss_ver = 0
        if self.args.loss_weights.ver_weight > 0:
            reconstructed_axis_angle = rc.matrix_to_axis_angle(reconstructed_pose_matrix).reshape(batch_size * frames, 129)
            target_axis_angle = target_pose.reshape(batch_size * frames, 129)
            
            predicted_verts = self.smplh(**self.prepare_smpl_input(reconstructed_axis_angle)).vertices
            target_verts = self.smplh(**self.prepare_smpl_input(target_axis_angle)).vertices
            loss_ver = self.ver_loss(predicted_verts, target_verts) * self.args.loss_weights.ver_weight

            # predicted_verts = predicted_verts.reshape(batch_size, frames, -1, 3)
            # target_verts = target_verts.reshape(batch_size, frames, -1, 3)

            vertices_vel_loss = self.vel_loss(predicted_verts[:, 1:] - predicted_verts[:, :-1], target_verts[:, 1:] - target_verts[:, :-1])
            vertices_acc_loss = self.vel_loss(predicted_verts[:, 2:] + predicted_verts[:, :-2] - 2 * predicted_verts[:, 1:-1], target_verts[:, 2:] + target_verts[:, :-2] - 2 * target_verts[:, 1:-1])
            loss_ver += (vertices_vel_loss + vertices_acc_loss) * self.args.loss_weights.ver_weight
        
        loss_vel = loss_vel * self.args.loss_weights.vel_weight
        loss_acc = loss_acc * self.args.loss_weights.acc_weight
        loss_commit = loss_commit * self.args.loss_weights.commit_weight
        
        total_loss = loss_rec + loss_vel + loss_acc + loss_commit + loss_ver
        total_loss.backward()
        
        if self.args.training_params.grad_norm != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.training_params.grad_norm)
        self.opt.step()

        return {
            "total": total_loss.item(),
            "perplexity": perplexity,
            "rec": loss_rec.item(),
            "vel": loss_vel.item(),
            "acc": loss_acc.item(),
            "ver": loss_ver.item() if torch.is_tensor(loss_ver) else loss_ver,
            "commit": loss_commit.item()
        }

    def val_step(self, dict_data):
        self.model.eval()
        with torch.no_grad():
            target_pose = dict_data["pose"].to(self.device, non_blocking=True)
            batch_size, frames = target_pose.shape[0], target_pose.shape[1]
            target_pose_matrix = rc.axis_angle_to_matrix(target_pose.reshape(batch_size, frames, 43, 3))
            target_pose_6d = rc.matrix_to_rotation_6d(target_pose_matrix).reshape(batch_size, frames, 258)

            reconstructed_pose, loss_commit, perplexity = self.model(target_pose_6d)

            reconstructed_pose_matrix = rc.rotation_6d_to_matrix(reconstructed_pose.reshape(batch_size, frames, 43, 6))
            
            loss_rec = self.rec_loss(reconstructed_pose_matrix, target_pose_matrix) * self.args.loss_weights.rec_weight

            loss_vel = self.vel_loss(reconstructed_pose_matrix[:, 1:] - reconstructed_pose_matrix[:, :-1], target_pose_matrix[:, 1:] - target_pose_matrix[:, :-1])
            loss_acc = self.vel_loss(reconstructed_pose_matrix[:, 2:] + reconstructed_pose_matrix[:, :-2] - 2 * reconstructed_pose_matrix[:, 1:-1], target_pose_matrix[:, 2:] + target_pose_matrix[:, :-2] - 2 * target_pose_matrix[:, 1:-1])
            
            loss_ver = 0
            if self.args.loss_weights.ver_weight > 0:
                reconstructed_axis_angle = rc.matrix_to_axis_angle(reconstructed_pose_matrix).reshape(batch_size * frames, 129)
                target_axis_angle = target_pose.reshape(batch_size * frames, 129)
                
                predicted_verts = self.smplh(**self.prepare_smpl_input(reconstructed_axis_angle)).vertices
                target_verts = self.smplh(**self.prepare_smpl_input(target_axis_angle)).vertices
                loss_ver = self.ver_loss(predicted_verts, target_verts) * self.args.loss_weights.ver_weight

                # predicted_verts = predicted_verts.reshape(batch_size, frames, -1, 3)
                # target_verts = target_verts.reshape(batch_size, frames, -1, 3)

                vertices_vel_loss = self.vel_loss(predicted_verts[:, 1:] - predicted_verts[:, :-1], target_verts[:, 1:] - target_verts[:, :-1])
                vertices_acc_loss = self.vel_loss(predicted_verts[:, 2:] + predicted_verts[:, :-2] - 2 * predicted_verts[:, 1:-1], target_verts[:, 2:] + target_verts[:, :-2] - 2 * target_verts[:, 1:-1])
                loss_ver += (vertices_vel_loss + vertices_acc_loss) * self.args.loss_weights.ver_weight
            
            loss_vel = loss_vel * self.args.loss_weights.vel_weight
            loss_acc = loss_acc * self.args.loss_weights.acc_weight
            loss_commit = loss_commit * self.args.loss_weights.commit_weight
            
            total_loss = loss_rec + loss_vel + loss_acc + loss_commit + loss_ver
            
        return {
            "total": total_loss.item(),
            "perplexity": perplexity,
            "rec": loss_rec.item(),
            "vel": loss_vel.item(),
            "acc": loss_acc.item(),
            "ver": loss_ver.item() if torch.is_tensor(loss_ver) else loss_ver,
            "commit": loss_commit.item()
        }
    
    def prepare_smpl_input(self, model_output):
        batch_size = model_output.shape[0]
        #
        full_body_pose = torch.zeros((batch_size, 63), device=self.device)
        left_hand  = torch.zeros((batch_size, 45), device=self.device)
        right_hand = torch.zeros((batch_size, 45), device=self.device)
        #
        full_body_pose[:, self.body_indices] = model_output[:, 0:39]
        left_hand = model_output[:, 39:84]
        right_hand = model_output[:, 84:129]
        
        current_batch = {
            'body_pose': full_body_pose, # size: (batch_size, 63)
            'left_hand_pose': left_hand, # size: (batch_size, 45)
            'right_hand_pose': right_hand, # size: (batch_size, 45)
            'global_orient': torch.zeros((batch_size, 3), device=self.device),
            'transl': torch.zeros((batch_size, 3), device=self.device),
            'expression': torch.zeros((batch_size, 100), device=self.device),
            'jaw_pose': torch.zeros((batch_size, 3), device=self.device),
            'leye_pose': torch.zeros((batch_size, 3), device=self.device),
            'reye_pose': torch.zeros((batch_size, 3), device=self.device),
            'betas': self.smplh.betas.expand(batch_size, -1),
            'rot2pose': True
        }

        return current_batch