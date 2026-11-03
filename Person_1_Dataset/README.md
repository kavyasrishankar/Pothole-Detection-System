# Person 1 – Dataset Collection and Preparation

## Project Title

SmartRoad AI – A Spatiotemporal Geospatial Intelligence Framework for Vision-Based Road Defect Characterization and Optimization Techniques

## Objective

To collect, preprocess, verify and prepare the pothole dataset for YOLOv8n-Segmentation training.

## 1. Dataset Information

* **Dataset Name:** Iraqi Road Defect Dataset (IRD)
* **Image Source:** Dashcam only
* **Target Class:** Pothole
* **Original Class ID:** 3
* **Final YOLO Class ID:** 0
* **Image Format:** JPEG
* **Image Resolution:** 3840 × 2160

## 2. Dataset Selection

Only Dashcam images containing at least one pothole were selected.

Images containing potholes along with other road defects were also retained. However, only pothole segmentation annotations were included in the processed dataset.

Images without potholes and empty annotations were excluded.

## 3. Dataset Preprocessing

The following preprocessing activities were performed:

* Extracted Dashcam images and polygon annotations.
* Identified pothole images using the original class ID.
* Extracted corresponding images and annotations.
* Converted pothole class ID from 3 to 0.
* Checked image formats and dimensions.
* Identified corrupted images.
* Identified and removed duplicate images.
* Checked missing images and annotations.
* Verified polygon coordinates.
* Visually inspected sample segmentation annotations.

## 4. Final Dataset Statistics

| Description                | Count |
| -------------------------- | ----: |
| Original Dashcam images    | 1,006 |
| Images containing potholes |   132 |
| Duplicate images removed   |     1 |
| Final unique images        |   131 |
| Annotation files           |   131 |
| Total pothole polygons     |   173 |
| Corrupted images           |     0 |
| Missing annotations        |     0 |
| Invalid annotations        |     0 |

## 5. Dataset Split

The dataset was divided into three sets using a 70:20:10 ratio.

| Dataset    |  Images | Percentage |
| ---------- | ------: | ---------: |
| Training   |      91 |     69.47% |
| Validation |      26 |     19.85% |
| Testing    |      14 |     10.69% |
| **Total**  | **131** |   **100%** |

## 6. Folder Structure

```text
Prepared_Dataset/
│
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
│
├── data.yaml
└── dataset_statistics.csv
```

## 7. Technologies Used

* Python
* Google Colab
* Google Drive
* ZIP File Handling
* OpenCV
* Pillow (PIL)
* YOLOv8 Segmentation Annotation Format

## 8. Deliverables

1. `01_dataset_preparation.ipynb`
2. `data.yaml`
3. `dataset_statistics.csv`
4. Prepared pothole dataset uploaded to the team's shared Google Drive.

## 9. Dataset Storage

The original IRD dataset and prepared dataset are stored in the team's shared Google Drive.

The actual images and segmentation annotations are not uploaded to GitHub.

## 10. Conclusion

The IRD Dashcam dataset was successfully collected, filtered, cleaned and verified. A total of 131 unique pothole images containing 173 segmentation polygons were prepared.

The dataset was organised into training, validation and testing sets and configured for single-class YOLOv8n-Segmentation training.

The prepared dataset is available in the team's shared Google Drive for further model training.
