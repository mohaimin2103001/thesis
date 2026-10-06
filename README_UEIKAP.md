# UEIKAP Dataset

Open-source dataset for ship wake detection in satellite imagery, developed within the UEIKAP project.

## Contents

```text
UEIKAP_dataset/
├── README.md
├── UEIKAP_dataset_optical/
│   ├── AIS_optical_UEIKAP/
│   ├── images/
│   ├── images_tif_TBD/
│   └── labels/
└── UEIKAP_dataset_SAR/
    ├── AIS_interpolated_SAR_UEIKAP/
    ├── images/
    ├── images_tifs/
    ├── labels/
    └── labels_jsons/
```

## Subsets

### Optical
- `images/`: optical image chips used for wake detection
- `labels/`: corresponding annotations
- `AIS_optical_UEIKAP/`: AIS data used for labeling and validation
- `images_tif_TBD/`: reserved for future GeoTIFF release; empty in this version

### SAR
- `images/`: SAR image chips used for wake and ship detection
- `labels/`: corresponding annotations
- `labels_jsons/`: annotation files in JSON format
- `AIS_interpolated_SAR_UEIKAP/`: interpolated AIS data used for labeling and validation
- `images_tifs/`: GeoTIFF SAR images

## Notes

- The dataset contains Sentinel-1 SAR and Sentinel-2 optical data prepared for ship wake detection tasks.
- AIS data are included in this release for labeling and validation purposes.
- The optical and SAR subsets are distributed separately because they differ in imaging physics, preprocessing, and annotation content.
- `images_tif_TBD/` in the optical subset is intentionally empty in this first public version and will be populated in a future update.

## Citation

If you use this dataset, please cite the associated UEIKAP dataset record and paper.

## Acknowledgement

Dataset developed within the UEIKAP project (*Unveil and Explore the In depth Knowledge of Earth Observation Data for Maritime Applications*).
