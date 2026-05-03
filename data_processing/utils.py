import smplx
import torch
import numpy as np
from dataclasses import dataclass
import os
import cv2 
import pyrender
import trimesh
from pathlib import Path
import json
import pandas as pd

_model_cache = {}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
_model_dir = str(PROJECT_ROOT / "smpl_models")
_assets_dir = PROJECT_ROOT / "data" / "seamless" / "assets"
_participants = pd.read_csv(_assets_dir / "participants.csv", dtype=str)
_relationships = pd.read_csv(_assets_dir / "relationships.csv", dtype=str)
_filelist = pd.read_csv(_assets_dir / "filelist.csv", dtype=str)
_interactions = pd.read_csv(_assets_dir / "interactions.csv", dtype=str)

HAND_MAP = {
    'left_hand': range(91, 112),
    'right_hand': range(112, 133)
}

def split_full_id(full_id):
    """
    Simply splits the full id, into separate parts
    """
    parts = full_id.split('_')
    if len(parts) < 4:
        print("Error: ID format should be Vxx_Sxxxx_Ixxxxxxxx_Pxxxx")
        return None

    vendor_id = parts[0][1:] # This removes V from the prefix
    session_id = parts[1][1:] # This removes S from the prefix
    interaction_id = parts[2][1:] # This removes I from the prefix
    participant_id = parts[3][1:] # This removes P from the prefix
    return vendor_id, session_id, interaction_id, participant_id

def get_partner_full_id(full_id):
    """
    Returns the full id of the partner of the interaction related to the given full_id
    """
    vendor_id, session_id, interaction_id, _ = split_full_id(full_id)
    if _filelist is not None:
        interaction_prefix = f"V{vendor_id}_S{session_id}_I{interaction_id}"
        partners = _filelist[
            (_filelist['file_id'].str.startswith(interaction_prefix)) & 
            (_filelist['file_id'] != full_id)
        ]
        if not partners.empty:
            partner_full_id = partners.iloc[0]['file_id']
            return partner_full_id
    
    print(f"No partner found for {full_id}")

def get_seamless_meta_data(base_path, data):
    """
    Returns a filled SeamlessMetaData object based on the base_path of a seamless interaction sample and a loaded .npz data
    """
    meta_data = SeamlessMetaData(full_id=base_path.stem)
    meta_data.vendor_id, meta_data.session_id, meta_data.interaction_id, meta_data.participant_id = split_full_id(base_path.stem)
    # Load the transcript json file
    json_path = base_path.with_suffix('.json')
    if json_path.exists():
        with open(json_path, 'r') as file:
            meta_data.transcript = json.load(file)
    # Save the .wav audio path
    audio_path = base_path.with_suffix('.wav')
    if audio_path.exists():
        meta_data.audio_file = audio_path
    # Save the .mp4 video path
    video_path = base_path.with_suffix('.mp4')
    if video_path.exists():
        meta_data.video_file = video_path
    #
    participant_data_row = _participants[(_participants['vendor_id'] == meta_data.vendor_id) & (_participants['participant_id'] == meta_data.participant_id)]
    if not participant_data_row.empty:
        row = participant_data_row.iloc[0]
        meta_data.agreeableness_raw = row['agreeableness_raw']
        meta_data.conscientiousness_raw = row['conscientiousness_raw']
        meta_data.extraversion_raw = row['extraversion_raw']
        meta_data.neuroticism_raw = row['neuroticism_raw']
        meta_data.openness_raw = row['openness_raw']
    #
    relationship_data_row = _relationships[(_relationships['vendor_id'] == 'V' + meta_data.vendor_id) & (_relationships['session_id'] == meta_data.session_id)]
    if not relationship_data_row.empty:
        row = relationship_data_row.iloc[0]
        meta_data.relationship = row['relationship']
        meta_data.relationship_detail = row['relationship_detail']
    #
    meta_data.partner_full_id = get_partner_full_id(meta_data.full_id)
    #
    interactions_data_row = _interactions[(_interactions['prompt_hash'] == meta_data.interaction_id)]
    if not interactions_data_row.empty:
        row = interactions_data_row.iloc[0]
        meta_data.interaction_data = row
    #
    filelist_data_row = _filelist[(_filelist['file_id'] == meta_data.full_id)]
    if not filelist_data_row.empty:
        row = filelist_data_row.iloc[0]
        meta_data.has_annotation_1p = row['has_annotation_1p']
        meta_data.has_annotation_3p = row['has_annotation_3p']
    #
    if data != None:
        meta_data.keypoints = data.get('boxes_and_keypoints:keypoints', None)
        meta_data.emotion_scores = data.get('movement_v4:emotion_scores', data.get('movement:emotion_scores', None))
        meta_data.boxes = data.get('boxes_and_keypoints:box', None)
        #
        num_frames = len(meta_data.keypoints) if meta_data.keypoints is not None else 0
        meta_data.movement_intensity = np.ones(num_frames, dtype=int) # Will be overwritten below if data is present
        meta_data.hand_offscreen = np.zeros(num_frames, dtype=int) # Will be overwritten below if data is present
        for prefix in HAND_MAP.keys():
            setattr(meta_data, f"{prefix}_offscreen_percentage", np.zeros(num_frames))
        #
        if meta_data.keypoints is not None:
            # Hand movement intensity detection
            hand_indices = range(91, 133) # These are the hands based on coco wholebody
            hand_coords = meta_data.keypoints[:, hand_indices, :2] # only use the x and y coords
            displacements = np.linalg.norm(hand_coords[1:] - hand_coords[:-1], axis=2) # calculate movement (temporal variance)
            intensity = np.mean(displacements, axis=1)
            meta_data.movement_intensity = np.concatenate(([0], intensity))
            meta_data.movement_intensity_peak_threshold = np.percentile(intensity, 75)
            # Hand off screen detection
            if meta_data.boxes is not None:
                limit_x = np.max(meta_data.boxes[:, [0, 2]])
                limit_y = np.max(meta_data.boxes[:, [1, 3]])
                offscreen_values = []
                for prefix, indices in HAND_MAP.items():
                    coords = meta_data.keypoints[:, indices, :2]
                    off_boundary = (coords[..., 0] <= 0) | (coords[..., 0] >= limit_x) | (coords[..., 1] <= 0) | (coords[..., 1] >= limit_y)
                    is_offscreen = np.mean(off_boundary, axis=1) # The percentage of off screen points
                    setattr(meta_data, f"{prefix}_offscreen_percentage", is_offscreen)
                    offscreen_values.append(np.any(off_boundary, axis=1)) # Create a 0/1 instead of percentage due to np.any
                meta_data.hand_offscreen = (np.maximum.reduce(offscreen_values)).astype(int) # sets a frame to 0 if it is not offscreen for either hand and 1 if either hand has even 1 point offscreen

    return meta_data

def prune_seamless_npz_file(input_path, frames, output_folder):
    """
    Functions that can be used to prune a input file
    """
    npz_path = os.path.expanduser(input_path)
    if not os.path.exists(npz_path):
        print(f"Error: Source file not found at {npz_path}")
        return

    data = np.load(npz_path, allow_pickle=True)
    size = len(data['smplh:body_pose'])

    small_data = {}
    for key in data.files:
        val = data[key]
        # prune frames
        if isinstance(val, np.ndarray) and len(val) == size:
            small_data[key] = val[:frames]
        else:
            small_data[key] = val
    
    os.makedirs(output_folder, exist_ok=True)

    base_name = os.path.basename(input_path).replace(".npz", "")
    filename = f"{base_name}_test_{frames}_frames.npz"
    final_save_path = os.path.join(output_folder, filename)

    np.savez(final_save_path, **small_data)

    print(f"Saved {len(small_data['smplh:body_pose'])} frames to {final_save_path}")

@dataclass
class SMPLModelParams:
    num_frames: int
    orient: torch.Tensor
    body: torch.Tensor
    left_hand: torch.Tensor
    right_hand: torch.Tensor
    expression: torch.Tensor = None
    jaw_pose: torch.Tensor = None
    leye_pose: torch.Tensor = None
    reye_pose: torch.Tensor = None
    translation: torch.Tensor = None
    betas: torch.Tensor = None

@dataclass
class SeamlessMetaData:
    full_id : str = None
    vendor_id : str = None 
    session_id : str = None 
    interaction_id : str = None 
    participant_id : str = None 
    relationship : str = None
    relationship_detail : str = None
    extraversion_raw : float = None
    agreeableness_raw : float = None
    conscientiousness_raw : float = None
    neuroticism_raw : float = None
    openness_raw : float = None
    partner_full_id : str = None
    interaction_data : pd.Series = None
    transcript : json = None
    has_annotation_1p : bool = False
    has_annotation_3p : bool = False
    audio_file : Path = None
    video_file : Path = None
    keypoints : np.array = None
    boxes : np.array = None
    emotion_scores : np.array = None
    movement_intensity : np.array = None
    movement_intensity_peak_threshold : float = None
    hand_offscreen : np.array = None
    left_hand_offscreen_percentage : np.array = None
    right_hand_offscreen_percentage : np.array = None

def get_smplx_model(device):
    if 'smplx' not in _model_cache:
        _model_cache['smplx'] = smplx.create(
            model_path=_model_dir, 
            model_type='smplx',
            gender='neutral', 
            batch_size=1, 
            use_pca=False, 
            num_expression_coeffs=100
        ).to(device)
    return _model_cache['smplx']

def get_smplh_model(device, gender='male'):
    model_name = f"smplh_{gender}"
    if model_name not in _model_cache:
        _model_cache[model_name] = smplx.create(
            model_path=_model_dir, 
            model_type='smplh',
            gender=gender, 
            batch_size=1, 
            ext='pkl', 
            use_pca=False, 
        ).to(device)
    return _model_cache[model_name]

class GenericLoader:
    def __init__(self, npz_path, device='cpu'):
        self.device = device
        self.npz_path = npz_path
        if not os.path.exists(npz_path):
            print(f"NPZ File not found: {npz_path}")
            self.data = None
            return
        self.sample_name = os.path.basename(npz_path).replace('.npz', '')
        self.data = np.load(npz_path, allow_pickle=True)
        #
        self.num_frames = None
        self.body_model = None
        self.vertices = None
        self.SMPLParams = None

    def get_smpl_vertices_from_SMPLParams(self, chunk_size=512):
        all_verts = []
        
        for i in range(0, self.num_frames, chunk_size):
            end = min(i + chunk_size, self.num_frames)
            curr_batch_size = end - i
            self.body_model.batch_size = curr_batch_size

            current_batch = {
                'global_orient': self.SMPLParams.orient[i:end].to(self.device).view(curr_batch_size, -1) if self.SMPLParams.orient is not None else None,
                'body_pose': self.SMPLParams.body[i:end].to(self.device).view(curr_batch_size, -1) if self.SMPLParams.body is not None else None,
                'left_hand_pose': self.SMPLParams.left_hand[i:end].to(self.device).view(curr_batch_size, -1) if self.SMPLParams.left_hand is not None else None,
                'right_hand_pose': self.SMPLParams.right_hand[i:end].to(self.device).view(curr_batch_size, -1) if self.SMPLParams.right_hand is not None else None,
                'transl': self.SMPLParams.translation[i:end].to(self.device).view(curr_batch_size, -1) if self.SMPLParams.translation is not None else None,
                'expression': self.SMPLParams.expression[i:end].to(self.device).view(curr_batch_size, -1) if self.SMPLParams.expression is not None else None,
                'jaw_pose': self.SMPLParams.jaw_pose[i:end].to(self.device).view(curr_batch_size, -1) if self.SMPLParams.jaw_pose is not None else None,
                'leye_pose': self.SMPLParams.leye_pose[i:end].to(self.device).view(curr_batch_size, -1) if self.SMPLParams.leye_pose is not None else None,
                'reye_pose': self.SMPLParams.reye_pose[i:end].to(self.device).view(curr_batch_size, -1) if self.SMPLParams.reye_pose is not None else None,
                'betas': self.SMPLParams.betas[i:end].to(self.device).view(curr_batch_size, -1) if self.SMPLParams.betas is not None else self.body_model.betas.expand(self.body_model.batch_size, -1),
                'rot2pose' : True
            }

            current_batch = {k: v for k, v in current_batch.items() if v is not None} # Get rid of any none (some datasets dont have all options, or we have smpl-h vs smpl-x)
            with torch.no_grad():
                output = self.body_model(**current_batch)
                all_verts.append(output.vertices.detach().cpu().numpy())

        return np.concatenate(all_verts, axis=0)

    def generate_video(self, width=720, height=720, output_directory='cache/video_cache', rot_y=[0, 1, 0], rot_x=[1, 0, 0], cam_pose=[0, 0, 2.5], rot_angle=[0.0, 0.0], use_cache=True, progress=None):
        os.makedirs(output_directory, exist_ok=True)
        video_path = os.path.join(output_directory, f"{self.sample_name}_sample_video.mp4")
        if use_cache and os.path.exists(video_path):
            if progress is not None: progress(1.0, desc="Loaded from cache!")
            return video_path

        if self.vertices is None:
            self.vertices = self.get_smpl_vertices_from_SMPLParams()

        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0])
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        camera_pose = np.eye(4); camera_pose[:3, 3] = cam_pose
        scene.add(camera, pose=camera_pose)
        
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        scene.add(light, pose=camera_pose)

        renderer = pyrender.OffscreenRenderer(width, height)
        faces = self.body_model.faces

        rx_mat = trimesh.transformations.rotation_matrix(rot_angle[0], rot_x)
        ry_mat = trimesh.transformations.rotation_matrix(rot_angle[1], rot_y)
        full_rot = trimesh.transformations.concatenate_matrices(rx_mat, ry_mat)

        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.5, 0.5, 0.5, 1.0], # Darker gray base
            metallicFactor=0.2, roughnessFactor=0.8
        )

        mesh_node = None
        for i in range(self.num_frames):
            curr_vertices = self.vertices[i]
            
            temp_tm = trimesh.Trimesh(vertices=curr_vertices, faces=faces, process=False)
            curr_normals = temp_tm.vertex_normals

            primitive = pyrender.Primitive(positions=curr_vertices, normals=curr_normals, indices=faces, material=material)
            mesh = pyrender.Mesh(primitives=[primitive])
            
            if mesh_node is not None:
                scene.remove_node(mesh_node)
            mesh_node = scene.add(mesh, pose=full_rot)
            
            color, _ = renderer.render(scene)
            video_writer.write(cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
            if i % 100 == 0 and progress is not None:
                progress(0.6 + (0.4 * (i/self.num_frames)), desc=f"Writing Video {i}/{self.num_frames}")

        renderer.delete()
        video_writer.release()
        return video_path

class BeatLoader(GenericLoader):
    def __init__(self, npz_path, device='cpu'):
        super().__init__(npz_path, device=device)
        if self.data is None:
            return
        self.num_frames = self.data['poses'].shape[0]
        self.body_model = get_smplx_model(self.device)
        
        all_poses = torch.from_numpy(self.data['poses']).float()
        self.SMPLParams = SMPLModelParams(
            num_frames=self.num_frames,
            orient=all_poses[:, 0:3].view(-1, 1, 3),
            body=all_poses[:, 3:66].view(-1, 21, 3),
            jaw_pose=all_poses[:, 66:69].view(-1, 1, 3),
            leye_pose=all_poses[:, 69:72].view(-1, 1, 3),
            reye_pose=all_poses[:, 72:75].view(-1, 1, 3),
            left_hand=all_poses[:, 75:120].view(-1, 15, 3),
            right_hand=all_poses[:, 120:165].view(-1, 15, 3),
            expression=torch.from_numpy(self.data['expressions']).float()
        )
        self.vertices = self.get_smpl_vertices_from_SMPLParams()
    
    def generate_video(self, width=720, height=720, output_directory='cache/video_cache', rot_y=[0, 1, 0], rot_x=[1, 0, 0], cam_pose=[0, 0.0, 2.5], rot_angle=[0.0, 0.0], use_cache=True, progress=None):
        return super().generate_video(width, height, output_directory, rot_y, rot_x, cam_pose, rot_angle=rot_angle, use_cache=use_cache, progress=progress)


class SeamlessSMPLHLoader(GenericLoader):
    """
    This class is able to load a single example of the seamless interaction dataset
    """
    def __init__(self, npz_path, device='cpu', gender='male', discard_lower_body=True, load_vertices=True, with_meta_data=False):
        super().__init__(npz_path, device=device)
        if self.data is None:
            return
        self.path = npz_path
        self.num_frames = self.data['smplh:global_orient'].shape[0]
        self.body_model = get_smplh_model(self.device, gender=gender)
        self.body_mask = [0, 1, 3, 4, 6, 7, 9, 10]
        self.keep_indices = [i for i in range(21) if i not in self.body_mask]
        # Turn all lower body joints to 0
        if discard_lower_body:
            # To mimic what Dyadit did, we will set the lower body joints to 0. this mask is all lower body joints (8) total resulting in 13 leftover joits
            body_data = self.data['smplh:body_pose'].copy()
            body_data[:,self.body_mask,:] = 0

            global_orient = self.data['smplh:global_orient'].copy()
            global_orient[:] = 0

            translation = self.data['smplh:translation'].copy()
            translation[:] = 0
        #
        self.SMPLParams = SMPLModelParams(
            num_frames=self.num_frames,
            orient=torch.from_numpy(global_orient if discard_lower_body else self.data['smplh:global_orient']).float(),
            body=torch.from_numpy(body_data if discard_lower_body else self.data['smplh:body_pose']).float(),
            left_hand=torch.from_numpy(self.data['smplh:left_hand_pose']).float(),
            right_hand=torch.from_numpy(self.data['smplh:right_hand_pose']).float(),
            translation=torch.from_numpy(translation if discard_lower_body else None).float()
        )
        if load_vertices:
            self.vertices = self.get_smpl_vertices_from_SMPLParams()

        if with_meta_data:
            base_path = Path(npz_path)
            self.meta_data = get_seamless_meta_data(base_path, self.data)

    
    def generate_video(self, width=720, height=720, output_directory='cache/video_cache', rot_y=[0, 1, 0], rot_x=[1, 0, 0], cam_pose=[0, 0.0, 2.5], rot_angle=[0, 0.0], use_cache=True, progress=None):
        return super().generate_video(width, height, output_directory, rot_y, rot_x, cam_pose, rot_angle=rot_angle, use_cache=use_cache, progress=progress)

    def get_trainable_pose(self):
        """
        Converts SMPLParams into a flat 129-dim vector for training purposes
        """
        # get the .body (21 joints) as the 13 joints we want to train (upper body only)
        body_filtered = self.SMPLParams.body[:, self.keep_indices, :]
        
        # Flatten each component
        flattened_body = body_filtered.reshape(body_filtered.shape[0], -1)
        flattened_left_hand = self.SMPLParams.left_hand.reshape(self.SMPLParams.left_hand.shape[0], -1)
        flattened_right_hand = self.SMPLParams.right_hand.reshape(self.SMPLParams.right_hand.shape[0], -1)
        
        # concatenate into the final 129 dim vector used for training (order is body->left_hand->right_hand)
        final_vector = torch.cat([flattened_body, flattened_left_hand, flattened_right_hand], dim=-1)
        
        return final_vector



def generate_interaction_video(loader1, loader2, width=1280, height=720, output_directory='cache/video_cache/interactions', progress=None, use_cache=True):
    os.makedirs(output_directory, exist_ok=True)
    sample_name = f"{loader1.sample_name}_vs_{loader2.sample_name}"
    video_path = os.path.join(output_directory, f"{sample_name}_interaction.mp4")
    if use_cache and os.path.exists(video_path):
        if progress is not None: progress(1.0, desc="Loaded from cache!")
        return video_path
    
    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0])

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = [0, 0.5, 3.5]  # raised slightly and moved back
    scene.add(camera, pose=camera_pose)
    
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(width, height)
    faces = loader1.body_model.faces # assuming same body model for both

    # Transformations for Facing Each Other
    p1_trans = trimesh.transformations.translation_matrix([-0.7, 0, 0])
    p1_rot = trimesh.transformations.rotation_matrix(np.radians(90), [0, 1, 0])
    p1_pose = trimesh.transformations.concatenate_matrices(p1_trans, p1_rot)

    p2_trans = trimesh.transformations.translation_matrix([0.7, 0, 0])
    p2_rot = trimesh.transformations.rotation_matrix(np.radians(-90), [0, 1, 0])
    p2_pose = trimesh.transformations.concatenate_matrices(p2_trans, p2_rot)

    node1, node2 = None, None
    num_frames = min(loader1.num_frames, loader2.num_frames)

    for i in range(num_frames):
        mesh1 = pyrender.Mesh.from_trimesh(trimesh.Trimesh(loader1.vertices[i], faces, process=False))
        mesh2 = pyrender.Mesh.from_trimesh(trimesh.Trimesh(loader2.vertices[i], faces, process=False))

        if node1: scene.remove_node(node1)
        if node2: scene.remove_node(node2)
            
        node1 = scene.add(mesh1, pose=p1_pose)
        node2 = scene.add(mesh2, pose=p2_pose)

        color, _ = renderer.render(scene)
        video_writer.write(cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
        
        if progress is not None and i % 50 == 0:
            progress(i/num_frames, desc=f"Rendering Interaction: {i}/{num_frames}")

    renderer.delete()
    video_writer.release()
    return video_path