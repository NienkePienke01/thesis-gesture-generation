# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Download page for the Seamless Interaction dataset.
"""

import os
from pathlib import Path

import streamlit as st

from seamless_interaction.app.config import CSS
from seamless_interaction.app.utils import download_archive_files, get_fs_instance
from seamless_interaction.constants import ALL_LABELS, ALL_SPLITS
from seamless_interaction.fs import SeamlessInteractionFS

# Page configuration
st.set_page_config(
    page_title="Seamless Interaction Download",
    page_icon="🗳️",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(CSS, unsafe_allow_html=True)


def display_hf_browser(fs: SeamlessInteractionFS):
    """Browse the HuggingFace dataset structure and download archives."""
    st.subheader("🤗 HuggingFace Dataset Browser")
    st.markdown(
        "Browse and download tar archives from the HuggingFace dataset repository."
    )

    # Dataset structure info
    with st.expander("📁 Dataset Structure", expanded=False):
        st.code("""
        datasets/facebook/seamless-interaction/
        ├── improvised/
        │   ├── train/
        │   │   ├── 0000/
        │   │   │   ├── 0000.tar
        │   │   │   ├── 0001.tar
        │   │   │   └── ...
        │   │   ├── 0001/
        │   │   └── ...
        │   ├── dev/
        │   └── test/
        └── naturalistic/
            ├── train/
            ├── dev/
            └── test/
        
        Each tar file contains:
        - .mp4 files (video)
        - .wav files (audio) 
        - .json files (annotations, metadata)
        - .npz files (imitator/movement, smplh, keypoints data)
        """)

    local_dir = st.text_input(
        "Local directory",
        value=str(Path.home() / "datasets/seamless_interaction"),
        key="local_dir",
    )
    fs.config.local_dir = local_dir

    # Selection interface
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        label = st.selectbox("Label:", ALL_LABELS, key="hf_label")

    with col2:
        split = st.selectbox("Split:", ALL_SPLITS, key="hf_split")

    with col3:
        batch = st.selectbox("Batch:", fs.list_batches(label, split), key="hf_batch")

    with col4:
        archive = st.selectbox(
            "Archive:", fs.list_archives(label, split, batch), key="hf_archive"
        )

    with st.container():
        _col_l, col_c, _col_r = st.columns([1, 2, 1])
        with col_c:
            st.markdown(f"## {label} - {split} - {batch:04d}/{archive:04d}.tar")
            st.markdown(
                f"Size: {fs.get_tar_archive_size(label, split, batch, archive):.2f} GB"
            )

            if st.button(
                f"📥 Download `{batch:04d}/{archive:04d}.tar`", use_container_width=True
            ):
                success, extract_path = download_archive_files(
                    fs, label, split, batch, archive
                )
                if success:
                    st.success(
                        f"✅ Archive downloaded and extracted to: `{extract_path}`"
                    )

                    # Show extracted files
                    try:
                        files = os.listdir(extract_path)
                        st.markdown("### 📄 Extracted Files")
                        for file in sorted(files)[:20]:  # Show first 20 files
                            file_path = os.path.join(extract_path, file)
                            file_size = os.path.getsize(file_path)
                            st.text(f"📄 {file} ({file_size:,} bytes)")

                        if len(files) > 20:
                            st.text(f"... and {len(files) - 20} more files")

                    except Exception as e:
                        st.error(f"Error listing extracted files: {e}")
                else:
                    st.error("❌ Failed to download archive")


def main():
    st.header("🗳️ Download")

    display_hf_browser(get_fs_instance())


if __name__ == "__main__":
    main()
