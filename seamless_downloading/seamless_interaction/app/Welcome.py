# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Seamless Interaction",
    page_icon="👋",
)


def main():
    st.write("# Welcome to Seamless Interaction!")

    st.image(
        Path(__file__).parent.parent.parent.parent / "assets/banner.gif",
        width=1000,
    )

    st.markdown(
        """
        **Seamless Interaction** is a large-scale multimodal dataset of 
        4,000+ hours of human interactions for AI research
        """
    )

    # Quick access links
    blog_url = (
        "https://ai.meta.com/blog/"
        "seamless-interaction-dataset-natural-conversation-dynamics"
    )
    website_url = "https://ai.meta.com/research/seamless-interaction/"
    demo_url = "https://www.aidemos.meta.com/seamless_interaction_dataset"
    hf_url = "https://huggingface.co/datasets/facebook/seamless-interaction"
    paper_url = (
        "https://ai.meta.com/research/publications/"
        "seamless-interaction-dyadic-audiovisual-motion-modeling-"
        "and-large-scale-dataset"
    )

    st.markdown(
        f"""
        <table style="width:100%; text-align:center; margin:20px 0;">
        <tr>
        <td align="center">
        <a href="{blog_url}" target="_blank">
        🖼️ Blog
        </a>
        </td>
        <td align="center">
        <a href="{website_url}" target="_blank">
        🌐 Website
        </a>
        </td>
        <td align="center">
        <a href="{demo_url}" target="_blank">
        🎮 Demo
        </a>
        </td>
        <td align="center">
        <a href="{hf_url}" target="_blank">
        🤗 HuggingFace
        </a>
        </td>
        <td align="center">
        <a href="{paper_url}" target="_blank">
        📄 Paper
        </a>
        </td>
        </tr>
        </table>
        """,
        unsafe_allow_html=True,
    )

    # Overview section
    st.markdown("## 🔍 Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Hours", "4,000+", help="Hours of recorded interactions")
        st.metric("Participants", "4,000+", help="Unique participants across sessions")

    with col2:
        modality_help = "Video, Audio, Transcripts, Motion, Annotations"
        st.metric("Modalities", "6+", help=modality_help)
        st.metric("Total Size", "~27TB", help="Complete dataset size")

    # Key features
    st.markdown("## ✨ Key Features")

    feature_cols = st.columns(3)

    with feature_cols[0]:
        st.markdown("""
        **🎥 Rich Multimodal Data**
        - High-definition video (1080p, 30/29.97fps)
        - Denoised audio (48kHz, 16-bit)
        - Time-aligned transcripts
        """)

    with feature_cols[1]:
        st.markdown("""
        **🤖 AI-Ready Features**
        - SMPL-H body model parameters
        - Facial keypoints and expressions
        - Emotion and gaze encodings
        """)

    with feature_cols[2]:
        st.markdown("""
        **📊 Human Annotations**
        - Internal state annotations
        - Behavioral analysis
        - Personality assessments (BFI-2)
        """)

    # Applications section
    st.markdown("## 🧪 Research Applications")

    app_cols = st.columns(2)

    with app_cols[0]:
        st.markdown("""
        - 🤖 **Virtual agents and embodied AI**
        - 🎭 **Natural human-computer interaction**
        - 📡 **Advanced telepresence experiences**
        """)

    with app_cols[1]:
        st.markdown("""
        - 📊 **Multimodal content analysis tools**
        - 🎬 **Animation and synthetic content generation**
        - 🧠 **Social dynamics understanding**
        """)

    # Quick start section
    st.markdown("## 🚀 Quick Start")

    tab1, tab2, tab3 = st.tabs(["📥 Download", "💻 Code Example", "📖 Documentation"])

    with tab1:
        st.markdown("""
        **Choose your download method:**
        
        - **🔍 Single Example** (~100MB): Perfect for exploration
        - **📂 Sample Set** (~1GB): Initial prototyping  
        - **📦 Single Batch** (~50GB): Local development
        - **🌍 Whole Dataset** (~27TB): Complete research
        
        Use the **Download** page in the sidebar to get started!
        """)

    with tab2:
        st.code(
            """
# Quick setup
from seamless_interaction.fs import SeamlessInteractionFS, DatasetConfig

# Initialize filesystem
config = DatasetConfig(label="improvised", split="dev")
fs = SeamlessInteractionFS(config=config)

# Download a sample
file_ids = fs.sample_random_file_ids(num_samples=1)
fs.gather_file_id_data_from_s3(file_ids[0])
        """,
            language="python",
        )

    with tab3:
        st.markdown("""
        **Navigate through the app:**
        
        1. **📊 Overview**: Explore dataset statistics and structure
        2. **🗳️ Download**: Download specific batches or samples
        3. **🎭 Interaction**: View and analyze individual interactions
        
        Each page provides interactive tools for dataset exploration!
        """)

    # Dataset structure
    st.markdown("## 📂 Dataset Structure")

    with st.expander("🔍 Click to view dataset organization"):
        st.markdown("""
        ```
        seamless_interaction/
        ├── improvised/          # Guided prompt interactions
        │   ├── dev/             # Development split
        │   ├── test/            # Test split  
        │   └── train/           # Training split
        └── naturalistic/        # Spontaneous conversations
            ├── dev/
            ├── test/
            └── train/
        ```
        
        **File naming convention:**
        - `V<vendor>`: Collection site identifier
        - `S<session>`: Session identifier  
        - `I<interaction>`: Interaction within session
        - `P<participant>`: Individual participant
        """)

    # License and citation
    st.markdown("## 📄 License & Citation")

    st.info("""
    **License:** CC-BY-NC 4.0 (Creative Commons Attribution-NonCommercial)
    
    **Citation:** If you use this dataset, please cite our technical report and this repository.
    
    ```bibtex
    @inproceedings{seamless2025,
        title={Seamless Interaction: Dyadic Audiovisual Motion Modeling and Large-Scale Dataset},
        author={Seamless Next Team},
        year={2025},
    }
    ```
    """)

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; color: #666;">
    Made with ❤️ by the FAIR Seamless Team<br>
    © Meta Platforms, Inc. and affiliates
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
