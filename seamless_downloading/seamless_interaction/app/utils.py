# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import streamlit as st

from seamless_interaction.fs import DatasetConfig, SeamlessInteractionFS


@st.cache_resource
def get_fs_instance(
    label: str = "improvised", split: str = "dev", local_dir: str | None = None
) -> SeamlessInteractionFS:
    """Initialize the SeamlessInteractionFS instance."""
    config = DatasetConfig(label=label, split=split, local_dir=local_dir)
    return SeamlessInteractionFS(config=config)


def download_archive_files(
    fs: SeamlessInteractionFS, label: str, split: str, batch: str, archive: str
) -> tuple[bool, str]:
    """Download and extract a specific archive from HuggingFace."""
    try:
        with st.spinner(
            f"Downloading archive {batch:04d}/{archive:04d}.tar (size: {fs.get_tar_archive_size(label, split, batch, archive):.2f} GB)..."
        ):
            success, extract_path = fs.download_archive_from_hf(
                label=label, split=split, batch=batch, archive=archive, extract=True
            )

            if success:
                # Update the local directory in fs to point to extracted directory
                fs._local_dir = os.path.dirname(extract_path)
                return True, extract_path
            else:
                return False, ""

    except Exception as e:
        st.error(f"Error downloading archive: {e}")
        return False, ""


def check_local_files(fs: SeamlessInteractionFS, participant_id: str) -> dict:
    """Check what files are available locally for a participant."""
    try:
        paths = fs.get_path_list_for_file_id_local(participant_id)

        file_status = {
            "video": None,
            "audio": None,
            "json": [],
            "npz": [],
            "total_files": 4,  # video, audio, json, npz
            "available_files": 0,
            "directory": os.path.dirname(paths[0]) if paths else None,
        }

        for path in paths:
            if os.path.exists(path):
                file_status["available_files"] += 1

                if path.endswith(".mp4"):
                    file_status["video"] = path
                elif path.endswith(".wav"):
                    file_status["audio"] = path
                elif ".json" in path:
                    file_status["json"].append(path)
                elif ".npz" in path:
                    file_status["npz"].append(path)

        return file_status

    except Exception as e:
        return {
            "video": None,
            "audio": None,
            "json": [],
            "npz": [],
            "total_files": 4,  # video, audio, json, npz
            "available_files": 0,
            "error": str(e),
            "directory": None,
        }
