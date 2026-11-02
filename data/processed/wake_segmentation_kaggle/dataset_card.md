# UEIKAP Authentic Spaceborne SAR Wake Segmentation Benchmark
## Dataset Card & Provenance Documentation

- **Dataset Name:** UEIKAP Sentinel-1 SAR Wake Segmentation Package
- **Source Benchmark:** Universita degli studi di Napoli Federico II / ESA Sentinel-1
- **Permanent DOI:** [10.5281/zenodo.18391027](https://doi.org/10.5281/zenodo.18391027)
- **Sensor:** Sentinel-1 C-Band SAR (5.405 GHz), VV/VH Polarizations
- **Image Size:** 512 x 512 pixels
- **Total Validated Samples:** 514
- **Training Samples:** 401 (48 unique acquisition passes)
- **Validation Samples:** 113 (11 unique acquisition passes)
- **Leakage Prevention:** Grouped strictly by Sentinel-1 satellite pass timestamp (`acquisition_group`). No chip from a validation orbit pass appears in training.
- **Labels:** Binary Wake Masks (0 = Background ocean, 255 = Authentic hydrodynamic ship wake).
