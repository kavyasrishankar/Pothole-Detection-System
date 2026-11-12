
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import folium
from streamlit_folium import st_folium

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Pothole Detection & Monitoring",
    page_icon="🛣️",
    layout="wide"
)

OUTPUT_DIR = (
    "/content/drive/MyDrive/Pothole_Project/Batch_Results"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR, "Gamma_YOLO_Image_Summary.csv"
)

FEATURES_FILE = os.path.join(
    OUTPUT_DIR, "Gamma_YOLO_Pothole_Features.csv"
)

INTEGRATED_FILE = os.path.join(
    OUTPUT_DIR, "Pothole_Integrated_Features.csv"
)

SEVERITY_FILE = os.path.join(
    OUTPUT_DIR, "Pothole_Severity_Results.csv"
)

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

summary_df = load_csv(SUMMARY_FILE)
features_df = load_csv(FEATURES_FILE)
integrated_df = load_csv(INTEGRATED_FILE)
severity_df = load_csv(SEVERITY_FILE)

# Prefer severity results, then integrated results,
# then raw YOLO features.
if not severity_df.empty:
    data = severity_df.copy()
elif not integrated_df.empty:
    data = integrated_df.copy()
else:
    data = features_df.copy()

st.title("🛣️ Road Pothole Detection & Monitoring System")
st.caption(
    "YOLOv8n-Segmentation | GeoCLIP Location Estimation | "
    "Visual Severity Assessment"
)

if data.empty:
    st.error(
        "No pothole feature data found. "
        "Please generate the integrated or YOLO feature CSV first."
    )
    st.stop()

# --------------------------------------------------
# 3. DATA CLEANING
# --------------------------------------------------

for col in [
    "confidence", "area_pixels", "area_ratio",
    "bbox_width", "bbox_height", "aspect_ratio",
    "perimeter", "circularity", "solidity",
    "latitude", "longitude", "geoclip_score"
]:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

if "split" not in data.columns:
    data["split"] = "unknown"

if "visual_severity" not in data.columns:
    data["visual_severity"] = "Not classified"

# --------------------------------------------------
# 4. SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.header("Dashboard Filters")

available_splits = sorted(
    data["split"].dropna().astype(str).unique().tolist()
)

selected_splits = st.sidebar.multiselect(
    "Dataset split",
    options=available_splits,
    default=available_splits
)

filtered = data[
    data["split"].astype(str).isin(selected_splits)
].copy()

if "visual_severity" in filtered.columns:
    available_severities = sorted(
        filtered["visual_severity"].dropna().astype(str).unique()
    )

    selected_severities = st.sidebar.multiselect(
        "Visual severity",
        options=available_severities,
        default=available_severities
    )

    filtered = filtered[
        filtered["visual_severity"].astype(str).isin(
            selected_severities
        )
    ]

# --------------------------------------------------
# 5. KEY PERFORMANCE INDICATORS
# --------------------------------------------------

if not summary_df.empty and "status" in summary_df.columns:
    successful = summary_df[
        summary_df["status"].astype(str).str.upper() == "SUCCESS"
    ]

    total_images = successful["image_path"].nunique()
else:
    total_images = (
        filtered["image_path"].nunique()
        if "image_path" in filtered.columns else 0
    )

total_potholes = len(filtered)

low_count = (
    (filtered["visual_severity"] == "Low").sum()
    if "visual_severity" in filtered.columns else 0
)

medium_count = (
    (filtered["visual_severity"] == "Medium").sum()
    if "visual_severity" in filtered.columns else 0
)

high_count = (
    (filtered["visual_severity"] == "High").sum()
    if "visual_severity" in filtered.columns else 0
)

st.subheader("Project Overview")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Images processed", f"{total_images:,}")
c2.metric("Potholes detected", f"{total_potholes:,}")
c3.metric("Low severity", f"{low_count:,}")
c4.metric("Medium severity", f"{medium_count:,}")
c5.metric("High severity", f"{high_count:,}")

st.divider()

# --------------------------------------------------
# 6. SEVERITY DISTRIBUTION
# --------------------------------------------------

st.subheader("Visual Severity Distribution")

severity_counts = (
    filtered["visual_severity"]
    .fillna("Unknown")
    .value_counts()
    .rename_axis("Severity")
    .reset_index(name="Pothole Count")
)

col1, col2 = st.columns(2)

with col1:
    if not severity_counts.empty:
        fig = px.bar(
            severity_counts,
            x="Severity",
            y="Pothole Count",
            color="Severity",
            title="Potholes by Visual Severity",
            text="Pothole Count"
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    if not severity_counts.empty:
        fig = px.pie(
            severity_counts,
            names="Severity",
            values="Pothole Count",
            title="Severity Percentage",
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Severity is an experimental visual size-based proxy "
    "derived from segmentation area. It is not a measured "
    "physical depth or engineering-standard road rating."
)

st.divider()

# --------------------------------------------------
# 7. YOLO CONFIDENCE ANALYSIS
# --------------------------------------------------

st.subheader("YOLO Segmentation Results and Confidence")

if "confidence" in filtered.columns:
    confidence_data = filtered.dropna(subset=["confidence"])

    if not confidence_data.empty:
        avg_confidence = confidence_data["confidence"].mean()

        st.metric(
            "Average YOLO detection confidence",
            f"{avg_confidence:.3f}"
        )

        fig = px.histogram(
            confidence_data,
            x="confidence",
            nbins=20,
            title="YOLO Detection Confidence Distribution",
            labels={"confidence": "Detection Confidence"}
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No confidence values are available.")
else:
    st.info("Confidence column is unavailable.")

st.divider()

# --------------------------------------------------
# 8. DATASET SPLIT AND IMAGE PROCESSING
# --------------------------------------------------

st.subheader("Dataset Split and Processing Performance")

col1, col2 = st.columns(2)

with col1:
    if not summary_df.empty and "split" in summary_df.columns:
        split_summary = (
            summary_df[
                summary_df.get(
                    "status",
                    pd.Series("SUCCESS", index=summary_df.index)
                ).astype(str).str.upper() == "SUCCESS"
            ]
            .groupby("split")["image_path"]
            .nunique()
            .reset_index(name="Images Processed")
        )

        fig = px.bar(
            split_summary,
            x="split",
            y="Images Processed",
            title="Successfully Processed Images by Split",
            text="Images Processed"
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Image summary CSV is unavailable.")

with col2:
    if not summary_df.empty and "status" in summary_df.columns:
        status_summary = (
            summary_df["status"]
            .fillna("Unknown")
            .value_counts()
            .rename_axis("Status")
            .reset_index(name="Image Count")
        )

        fig = px.pie(
            status_summary,
            names="Status",
            values="Image Count",
            title="Image Processing Status"
        )

        st.plotly_chart(fig, use_container_width=True)

st.divider()

# --------------------------------------------------
# 9. VALIDATION PERFORMANCE
# --------------------------------------------------

st.subheader("YOLO Model Validation Performance")

# Validation metrics from the recorded experiments.
# Original and CLAHE: up to 100 epochs.
# Gamma: 50 epochs, so comparison is not perfectly controlled.

metrics = pd.DataFrame({
    "Model": ["Original YOLOv8n-Seg", "CLAHE YOLOv8n-Seg",
              "Gamma YOLOv8n-Seg"],
    "Precision": [0.7178, 0.7386, 0.7445],
    "Recall": [0.6617, 0.6418, 0.7104],
    "mAP50": [0.7351, 0.7188, 0.7356],
    "mAP50-95": [0.4302, 0.4053, 0.4427],
    "Training Epochs": ["Up to 100", "Up to 100", "50"]
})

st.dataframe(
    metrics.style.format({
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "mAP50": "{:.4f}",
        "mAP50-95": "{:.4f}"
    }),
    use_container_width=True,
    hide_index=True
)

fig = px.bar(
    metrics,
    x="Model",
    y=["Precision", "Recall", "mAP50", "mAP50-95"],
    barmode="group",
    title="Validation Metric Comparison"
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "These are validation metrics, not independent test results. "
    "Gamma was trained for 50 epochs, while the other experiments "
    "were trained for up to 100 epochs."
)

st.divider()

# --------------------------------------------------
# 10. PER-POTHOLE FEATURE TABLE
# --------------------------------------------------

st.subheader("Individual Pothole Feature Analysis")

display_columns = [
    "image_path", "split", "instance_id",
    "confidence", "area_pixels", "area_ratio",
    "bbox_width", "bbox_height", "aspect_ratio",
    "perimeter", "circularity", "solidity",
    "latitude", "longitude", "visual_severity"
]

display_columns = [
    col for col in display_columns if col in filtered.columns
]

st.dataframe(
    filtered[display_columns],
    use_container_width=True,
    hide_index=True
)

csv_data = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download filtered pothole results",
    data=csv_data,
    file_name="Pothole_Filtered_Results.csv",
    mime="text/csv"
)

st.divider()

# --------------------------------------------------
# 11. OPTIONAL GEOGRAPHIC MAP
# --------------------------------------------------

st.subheader("Estimated Geographic Distribution")

st.warning(
    "GeoCLIP produces image-level location estimates, not "
    "verified GPS coordinates. Multiple potholes in one image "
    "share the same estimated coordinates. Do not interpret "
    "these points as confirmed road locations."
)

show_map = st.checkbox(
    "I have independently checked the coordinate quality. "
    "Show the estimated-location map.",
    value=False
)

if show_map:

    if not {"latitude", "longitude"}.issubset(filtered.columns):
        st.error("Latitude and longitude columns are unavailable.")
    else:
        map_df = filtered.dropna(
            subset=["latitude", "longitude"]
        ).copy()

        map_df = map_df[
            map_df["latitude"].between(-90, 90) &
            map_df["longitude"].between(-180, 180)
        ]

        if map_df.empty:
            st.info("No valid coordinates are available.")
        else:
            # Limit the number of plotted points for responsiveness.
            map_df = map_df.head(500)

            center_lat = map_df["latitude"].median()
            center_lon = map_df["longitude"].median()

            road_map = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=5
            )

            for _, row in map_df.iterrows():

                popup_text = (
                    f"Image: {os.path.basename(str(row.get('image_path', '')))}"
                    f"<br>Severity: {row.get('visual_severity', 'Unknown')}"
                    f"<br>YOLO confidence: {row.get('confidence', 'N/A')}"
                )

                folium.CircleMarker(
                    location=[
                        row["latitude"],
                        row["longitude"]
                    ],
                    radius=5,
                    popup=popup_text,
                    tooltip="Estimated pothole location",
                    fill=True
                ).add_to(road_map)

            st_folium(
                road_map,
                width=None,
                height=500,
                use_container_width=True
            )

            st.caption(
                "Map points represent GeoCLIP image-level estimates. "
                "They are not verified pothole GPS measurements."
            )

# --------------------------------------------------
# 12. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Road Pothole Detection and Monitoring System | "
    "YOLOv8n-Seg + GeoCLIP | Experimental visual severity assessment"
)
