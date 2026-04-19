# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Overview page for the Seamless Interaction dataset.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from seamless_interaction.app.config import CSS, DatasetStats
from seamless_interaction.app.utils import get_fs_instance

# Page configuration
st.set_page_config(
    page_title="Seamless Interaction Overview",
    page_icon="📊",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(CSS, unsafe_allow_html=True)


def calculate_dataset_stats(df: pd.DataFrame) -> DatasetStats:
    """Calculate dataset statistics."""
    if df.empty:
        return DatasetStats(0, 0, [], {}, {})

    total_interactions = df["interaction_key"].nunique()
    participants = df["participant"].unique().tolist()
    total_participants = len(participants)
    vendors = df["vendor"].unique().tolist()

    sessions_per_vendor = df.groupby("vendor")["session"].nunique().to_dict()
    interactions_per_session = df.groupby(["vendor", "session"]).size().to_dict()

    return DatasetStats(
        total_interactions=total_interactions,
        total_participants=total_participants,
        vendors=vendors,
        sessions_per_vendor=sessions_per_vendor,
        interactions_per_session=interactions_per_session,
    )


def display_overview_stats(stats: DatasetStats):
    """Display overview statistics."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Interactions", stats.total_interactions)
    with col2:
        st.metric("Total Participants", stats.total_participants)
    with col3:
        st.metric("Vendors", len(stats.vendors))
    with col4:
        avg_interactions = (
            np.mean(list(stats.interactions_per_session.values()))
            if stats.interactions_per_session
            else 0
        )
        st.metric("Avg Interactions/Session", f"{avg_interactions:.1f}")


def create_distribution_plots(df: pd.DataFrame):
    """Create distribution plots for the dataset."""
    if df.empty:
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Interactions per Vendor")
        vendor_counts = df["vendor"].value_counts()
        fig = px.bar(
            x=vendor_counts.index,
            y=vendor_counts.values,
            labels={"x": "Vendor", "y": "Number of Interactions"},
            color=vendor_counts.values,
            color_continuous_scale="viridis",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Sessions per Vendor")
        session_counts = df.groupby("vendor")["session"].nunique()
        fig = px.pie(
            values=session_counts.values,
            names=session_counts.index,
            title="Distribution of Sessions",
        )
        st.plotly_chart(fig, use_container_width=True)


def display_session_analysis(df: pd.DataFrame):
    st.subheader("Session Analysis")
    session_interaction_counts = (
        df.groupby(["vendor", "session"]).size().reset_index(name="interaction_count")
    )

    fig = px.box(
        session_interaction_counts,
        x="vendor",
        y="interaction_count",
        title="Distribution of Interactions per Session by Vendor",
    )
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data
def get_df() -> pd.DataFrame:
    df = get_fs_instance()._cached_filelist
    df = df.copy()
    df["vendor"] = df["file_id"].str.extract(r"V(\d+)_")[0]
    df["session"] = df["file_id"].str.extract(r"S(\d+)_")[0]
    df["interaction"] = df["file_id"].str.extract(r"I(\d+)_")[0]
    df["participant"] = df["file_id"].str.extract(r"P(\d+)")[0]
    df["interaction_key"] = df["file_id"].str.extract(r"(V\d+_S\d+_I\d+)_P\d+")[0]
    return df


def main():
    st.header("📊 Dataset Overview")

    df = get_df()

    if not df.empty:
        stats = calculate_dataset_stats(df)
        display_overview_stats(stats)

        st.markdown("---")
        create_distribution_plots(df)

        st.markdown("---")
        st.subheader("📋 Sample Data")
        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("---")
        display_session_analysis(df)

    else:
        st.warning("No metadata available for the selected configuration.")


if __name__ == "__main__":
    main()
