# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Interaction viewer page for the Seamless Interaction dataset.
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import streamlit as st

from seamless_interaction.app.config import CSS
from seamless_interaction.app.utils import check_local_files, get_fs_instance
from seamless_interaction.fs import SeamlessInteractionFS

# Page configuration
st.set_page_config(
    page_title="Seamless Interaction Interaction Viewer",
    page_icon="🎭",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(CSS, unsafe_allow_html=True)


def display_file_status_widget(file_status: dict):
    """Display a compact file status widget."""
    if "error" in file_status:
        st.error(f"Error checking files: {file_status['error']}")
        return

    total = file_status["total_files"]
    available = file_status["available_files"]

    # Status indicator
    if available == 0:
        status_color = "🔴"
        status_text = "No files downloaded\n(Click on the `Download` button)"
    elif available == total:
        status_color = "🟢"
        status_text = "All files available"
    else:
        status_color = "🟡"
        status_text = f"{available}/{total} files available"

    st.markdown(f"**{status_color} {status_text}**")

    # Quick file type status
    file_types = []
    if file_status["video"]:
        file_types.append("📹 Video")
    if file_status["audio"]:
        file_types.append("🎵 Audio")
    if file_status["json"]:
        file_types.append("📝 JSON")
    if file_status["npz"]:
        file_types.append("🤖 Numpy")

    if file_types:
        st.markdown("Available: " + " | ".join(file_types))


def display_interaction_videos(fs: SeamlessInteractionFS, interaction_key: str):
    """Display videos for both participants side by side with local file checking."""
    participant_0_id, participant_1_id = fs.get_interaction_pairs(
        interaction_keys=[interaction_key]
    )[0]
    st.markdown('<div class="interaction-header">', unsafe_allow_html=True)
    st.markdown(
        f"<h3>🎭 Interaction: {participant_0_id.split('_')[2]} | Session: {participant_0_id.split('_')[1]}</h3>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Check local files for both participants
    status_0 = check_local_files(fs, participant_0_id)
    status_1 = check_local_files(fs, participant_1_id)

    cols = st.columns(2)

    with cols[0]:
        st.markdown("### 👤 Participant 0")
        with st.expander(f"Directory: {participant_0_id}", expanded=False):
            st.text(f"{status_0['directory']}")

        # Display file status
        display_file_status_widget(status_0)

        # Show video if available
        if status_0["video"] and os.path.exists(status_0["video"]):
            st.video(status_0["video"], format="video/mp4")
        else:
            st.warning("Video not available locally")

    with cols[1]:
        st.markdown("### 👤 Participant 1")
        with st.expander(f"Directory: {participant_1_id}", expanded=False):
            st.text(f"{status_1['directory']}")

        # Display file status
        display_file_status_widget(status_1)

        # Show video if available
        if status_1["video"] and os.path.exists(status_1["video"]):
            st.video(status_1["video"], format="video/mp4")
        else:
            st.warning("Video not available locally")

    # Sync play button - only show if both videos exist
    if (
        status_0["video"]
        and os.path.exists(status_0["video"])
        and status_1["video"]
        and os.path.exists(status_1["video"])
    ):
        if st.button("🎬 Sync Play All Videos", use_container_width=True):
            st.components.v1.html(
                """<script>
                let videos = parent.document.querySelectorAll("video");
                videos.forEach(v => {
                    v.currentTime = 0;
                    v.play();
                })
                </script>""",
                width=0,
                height=0,
            )


def display_interaction_info(fs: SeamlessInteractionFS, interaction_key: str):
    """Display interaction info for a participant."""
    file_ids = fs.get_interaction_pairs(interaction_keys=[interaction_key])[0]

    col1, col2 = st.columns(2)

    with col1:
        display_multimodal_data(fs, file_ids[0])

    with col2:
        display_multimodal_data(fs, file_ids[1])


def display_multimodal_data(fs: SeamlessInteractionFS, file_id: str):
    """Display multimodal data for a participant."""

    try:
        paths = fs.get_path_list_for_file_id_local(file_id)

        # Organize paths by modality
        modalities = {
            "Audio": [],
            "JSON": [],
            "Numpy": [],
        }

        for path in paths:
            if ".wav" in path:
                modalities["Audio"].append(path)
            elif ".json" in path:
                modalities["JSON"].append(path)
            elif ".npz" in path:
                modalities["Numpy"].append(path)

        st.subheader("📊 Multimodal Data")

        if not any(modalities.values()):
            st.warning("No local files found.")
            return

        for modality, files in modalities.items():
            if files:
                with st.expander(f"{modality}"):
                    for file_path in files:
                        exists = os.path.exists(file_path)

                        # Try to load and preview some files
                        if (
                            exists
                            and modality == "JSON"
                            and file_path.endswith(".json")
                        ):
                            try:
                                with open(file_path, "r") as f:
                                    data = json.load(f)
                                    st.json(data, expanded=False)
                            except Exception as e:
                                st.error(f"Error reading {file_path}: {e}")

                        elif (
                            exists
                            and modality == "Numpy"
                            and file_path.endswith(".npz")
                        ):
                            try:
                                data = np.load(file_path)
                                for key in list(data.keys()):
                                    arr = data[key]
                                    st.text(
                                        f"  {key}: shape={arr.shape}, dtype={arr.dtype}"
                                    )
                            except Exception as e:
                                st.error(f"Error reading {file_path}: {e}")

                        elif exists and modality == "Audio":
                            st.audio(file_path)

    except Exception as e:
        st.error(f"Error loading multimodal data: {e}")


def get_interactions(fs: SeamlessInteractionFS, label: str, split: str) -> list[str]:
    df = fs._cached_filelist
    df = df.loc[(df["label"] == label) & (df["split"] == split), ["file_id"]].copy()
    df["interaction_key"] = df["file_id"].str.extract(r"(V\d+_S\d+_I\d+)")[0]
    return df["interaction_key"].unique().tolist()


def download_interaction_pair(fs: SeamlessInteractionFS, interaction_key: str):
    """
    Download a pair of interactions from the same session (~100-200MB).

    Ideal for studying conversational dynamics between participants.
    Auto-samples interaction pairs if no interaction_key provided.

    :param interaction_key: Interaction key (V00_S0809_I00000582) or None to auto-sample
    """
    # Use specific interaction key
    pairs = fs.get_interaction_pairs(interaction_keys=[interaction_key])
    file_ids = pairs[0] if pairs else []

    if not file_ids:
        st.error("❌ No interaction pairs found")
        return

    # Download both participants from same interaction
    with st.spinner(f"Downloading interaction pair: {interaction_key}"):
        fs.download_batch_from_s3(file_ids)
        st.success(f"✅ Downloaded interaction pair: {file_ids}")


def main():
    st.header("🎬 Interaction Viewer")

    st.sidebar.title("Configuration")
    label = st.sidebar.selectbox("Label", ["improvised", "naturalistic"])
    split = st.sidebar.selectbox("Split", ["dev", "train", "test"])
    local_dir = st.sidebar.text_input(
        "Local directory", value=str(Path.home() / "datasets/seamless_interaction")
    )

    fs = get_fs_instance(label=label, split=split, local_dir=local_dir)
    interactions = get_interactions(fs, label, split)

    if interactions:
        # pick up an interaction or randomly select one
        st.session_state.selected_interaction = st.selectbox(
            "Select an interaction:",
            interactions,
        )
        if st.button("🎲 Randomly select an interaction", use_container_width=True):
            st.session_state.selected_interaction = random.choice(interactions)

        if st.session_state.selected_interaction is not None:
            if st.button("🔄 Download interaction", use_container_width=True):
                download_interaction_pair(fs, st.session_state.selected_interaction)

            display_interaction_videos(fs, st.session_state.selected_interaction)

            st.markdown("---")

            display_interaction_info(fs, st.session_state.selected_interaction)
    else:
        st.warning("No interaction data available for the selected configuration.")


if __name__ == "__main__":
    main()
