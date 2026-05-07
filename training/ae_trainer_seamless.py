import torch
import smplx
import numpy as np
from models.rvq import RVQVAE
from optimizers.loss_factory import GeodesicLoss, get_loss_func
from data_processing.utils import get_smplh_model
from utils import rotation_conversions as rc
from torch.nn.parallel import DistributedDataParallel as DDP

class SeamlessStage1Trainer:
    def __init__(self, args, gpu_id):
        self.args = args
        self.device = torch.device(f"cuda:{gpu_id}")

        self.model = RVQVAE(args, **args.model_params).to(self.device)
        self.model = DDP(self.model, device_ids=[gpu_id])

        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.training_params.lr_base)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=args.training_params.decay_epochs, gamma=0.1)

        # load smpl-h model
        self.smplh = get_smplh_model(self.device, 'male').eval()

        # loss Functions
        self.rec_loss = GeodesicLoss()
        self.vel_loss = torch.nn.L1Loss()
        self.ver_loss = torch.nn.MSELoss()

        self.body_mask = [0, 1, 3, 4, 6, 7, 9, 10]
        self.keep_indices = [i for i in range(21) if i not in self.body_mask]
        self.body_indices = []
        for idx in self.keep_indices:
            self.body_indices.extend([idx*3, idx*3+1, idx*3+2])
        self.body_indices = torch.tensor(self.body_indices, device=self.device)

    def train_step(self, dict_data):
        """
        dict_data is the dictionairy created by seamless_segmenter.py saved in lmdb format
        Performs one step of the training process
        """
        self.model.train()
        self.opt.zero_grad()
        
        target_pose = dict_data["pose"].to(self.device, non_blocking=True) # (batch_size, 64, 129)
        if self.args.mode == "upper_body":
            target_pose = target_pose[:, :, :39]
            joint_count = 13
        elif self.args.mode == "hands":
            target_pose = target_pose[:, :, 39:129]
            joint_count = 30
        else: # full
            joint_count = 43
        batch_size, frames = target_pose.shape[0], target_pose.shape[1]

        # conversion from axis-angle to 6D
        target_pose_matrix = rc.axis_angle_to_matrix(target_pose.reshape(batch_size, frames, joint_count, 3))
        target_pose_6d = rc.matrix_to_rotation_6d(target_pose_matrix).reshape(batch_size, frames, joint_count * 6)
        
        # do a forward pass to the model
        model_output = self.model(target_pose_6d)
        reconstructed_pose_6d = model_output["rec_pose"].reshape(batch_size, frames, joint_count, 6)
        
        # convert back to matrix form for the loss calculations
        reconstructed_pose_matrix = rc.rotation_6d_to_matrix(reconstructed_pose_6d)
        
        # compute the geodesic loss which compares rotation matrices directly
        loss_rec = self.rec_loss(reconstructed_pose_matrix, target_pose_matrix) * self.args.loss_weights.rec_weight * self.args.loss_weights.rec_pos_weight

        # compute the temporal loss which is the L1 loss that checks the difference between consecutive frames
        loss_vel = self.vel_loss(reconstructed_pose_matrix[:, 1:] - reconstructed_pose_matrix[:, :-1], target_pose_matrix[:, 1:] - target_pose_matrix[:, :-1])
        # acceleration loss (2nd derivative)
        loss_acc = self.vel_loss(reconstructed_pose_matrix[:, 2:] + reconstructed_pose_matrix[:, :-2] - 2 * reconstructed_pose_matrix[:, 1:-1], target_pose_matrix[:, 2:] + target_pose_matrix[:, :-2] - 2 * target_pose_matrix[:, 1:-1])
        
        # the vertices loss (kinematic loss)
        loss_ver = 0
        if self.args.loss_weights.rec_ver_weight > 0:
            # convert predicted and target matrices to axis angle for prepare_smpl_input
            reconstructed_pose_axis_angle = rc.matrix_to_axis_angle(reconstructed_pose_matrix).reshape(batch_size * frames, -1)
            target_pose_axis_angle = rc.matrix_to_axis_angle(target_pose_matrix).reshape(batch_size * frames, -1)
            # get vertices
            predicted_verts = self.smplh(**self.prepare_smpl_input(reconstructed_pose_axis_angle, self.args.mode)).vertices
            target_verts = self.smplh(**self.prepare_smpl_input(target_pose_axis_angle, self.args.mode)).vertices
            loss_ver = self.ver_loss(predicted_verts, target_verts) * self.args.loss_weights.rec_weight * self.args.loss_weights.rec_ver_weight

            # reshape for temporal losses
            predicted_verts = predicted_verts.reshape(batch_size, frames, -1, 3)
            target_verts = target_verts.reshape(batch_size, frames, -1, 3)

            vertices_vel_loss = self.vel_loss(predicted_verts[:, 1:] - predicted_verts[:, :-1], target_verts[:, 1:] - target_verts[:, :-1])
            vertices_acc_loss = self.vel_loss(predicted_verts[:, 2:] + predicted_verts[:, :-2] - 2 * predicted_verts[:, 1:-1], target_verts[:, 2:] + target_verts[:, :-2] - 2 * target_verts[:, 1:-1])
            loss_ver += (vertices_vel_loss + vertices_acc_loss) * self.args.loss_weights.rec_weight * self.args.loss_weights.rec_ver_weight
            
        # commitment loss
        loss_commit = model_output["embedding_loss"]
        
        # backward pass
        total_loss = loss_rec + loss_vel + loss_acc + loss_commit + loss_ver
        total_loss.backward()
        if self.args.training_params.grad_norm != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.training_params.grad_norm)
        self.opt.step()

        return {
            "total": total_loss.item(),
            "perplexity": model_output["perplexity"].item(),
            "rec": loss_rec.item(),
            "vel": loss_vel.item(),
            "acc": loss_acc.item(),
            "ver": loss_ver.item() if torch.is_tensor(loss_ver) else loss_ver,
            "commit": loss_commit.item()
        }

    def val_step(self, dict_data):
        """
        performs one validation step without gradient updates
        """
        self.model.eval()
        with torch.no_grad():
            raw_pose = dict_data["pose"].to(self.device, non_blocking=True)
            if self.args.mode == "upper_body":
                target_pose = raw_pose[:, :, :39]
                joint_count = 13
            elif self.args.mode == "hands":
                target_pose = raw_pose[:, :, 39:129]
                joint_count = 30
            else:
                target_pose = raw_pose
                joint_count = 43
                
            batch_size, frames = target_pose.shape[0], target_pose.shape[1]

            target_pose_matrix = rc.axis_angle_to_matrix(target_pose.reshape(batch_size, frames, joint_count, 3))
            target_pose_6d = rc.matrix_to_rotation_6d(target_pose_matrix).reshape(batch_size, frames, joint_count * 6)
            
            model_output = self.model(target_pose_6d)["rec_pose"].transpose(1, 2)
            reconstructed_pose_6d = model_output.reshape(batch_size, frames, joint_count, 6)
            reconstructed_pose_matrix = rc.rotation_6d_to_matrix(reconstructed_pose_6d)
            
            # reconstruction loss
            loss_rec = self.rec_loss(reconstructed_pose_matrix, target_pose_matrix) * self.args.loss_weights.rec_weight * self.args.loss_weights.rec_pos_weight
            
            # 4. vertex los
            loss_ver = 0
            if self.args.loss_weights.rec_ver_weight > 0:
                # convert predicted and target matrices to axis angle for prepare_smpl_input
                reconstructed_pose_axis_angle = rc.matrix_to_axis_angle(reconstructed_pose_matrix).reshape(batch_size * frames, -1)
                target_pose_axis_angle = rc.matrix_to_axis_angle(target_pose_matrix).reshape(batch_size * frames, -1)
                # get vertices
                predicted_verts = self.smplh(**self.prepare_smpl_input(reconstructed_pose_axis_angle, self.args.mode)).vertices
                target_verts = self.smplh(**self.prepare_smpl_input(target_pose_axis_angle, self.args.mode)).vertices
                loss_ver = self.ver_loss(predicted_verts, target_verts) * self.args.loss_weights.rec_weight * self.args.loss_weights.rec_ver_weight

                # reshape for temporal losses
                predicted_verts = predicted_verts.reshape(batch_size, frames, -1, 3)
                target_verts = target_verts.reshape(batch_size, frames, -1, 3)

                vertices_vel_loss = self.vel_loss(predicted_verts[:, 1:] - predicted_verts[:, :-1], target_verts[:, 1:] - target_verts[:, :-1])
                vertices_acc_loss = self.vel_loss(predicted_verts[:, 2:] + predicted_verts[:, :-2] - 2 * predicted_verts[:, 1:-1], target_verts[:, 2:] + target_verts[:, :-2] - 2 * target_verts[:, 1:-1])
                loss_ver += (vertices_vel_loss + vertices_acc_loss) * self.args.loss_weights.rec_weight * self.args.loss_weights.rec_ver_weight

            loss_commit = model_output["embedding_loss"]
            total_loss = loss_rec + loss_commit + loss_ver
            
        return {
            "total": total_loss.item(), 
            "rec": loss_rec.item(), 
            "ver": loss_ver.item()
        }
    
    def prepare_smpl_input(self, model_output, mode="full"):
        """
        model_output (batch_size, 39) if mode=="upper_body" or (batch_size, 90) if mode=="hands"
        mode "upper_body" or "hands" or "full"
        returns a dict ready for self.body_model(**current_batch) so you can easily load a smpl model with it
        """
        batch_size = model_output.shape[0]
        
        full_body_pose = torch.zeros((batch_size, 63), device=self.device)
        left_hand  = torch.zeros((batch_size, 45), device=self.device)
        right_hand = torch.zeros((batch_size, 45), device=self.device)

        if mode == "upper_body":
            # remap the 13 upper body joints into the 21-joint body_pose
            # flatten the indices
            full_body_pose[:, self.body_indices] = model_output
        elif mode == "hands":
            left_hand = model_output[:, 0:45]
            right_hand = model_output[:, 45:90]
        else:
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