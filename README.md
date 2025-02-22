# Project Data Management and Versioning

## Directory Structure
```
/ESTATEIQ/
│── Data/
│   ├── Raw/         # Raw dataset files (tracked using DVC)
│   ├── Processed/   # Preprocessed datasets
│── Scripts/             # Codebase
│── dvc.yaml         # DVC pipeline configuration
│── .dvc/            # DVC metadata files
│── README.md        # Documentation
```

## Acquiring Data
1. **Download from Source**: Fetch datasets manually from https://data.boston.gov/dataset/property-assessment
2. **DVC Versioning**: Pull the latest tracked dataset:
   ```sh
   dvc pull
   ```

## Handling Data Drift
- Periodically re-fetch and compare dataset statistics.
- Use monitoring tools to detect drift and trigger updates.
- Automate data fetching and model retraining.

## Running Unit Tests
To validate preprocessing and pipeline steps, execute:
```sh
pytest Scripts/test.py
```

## Versioning Data with DVC
1. **Track New Data**:
   ```sh
   dvc add Data/Raw/boston_2025.csv
   git add Data/Raw/boston_2025.csv.dvc
   git commit -m "Versioning dataset"
   git push
   dvc push
   ```
2. **Retrieve Latest Version**:
   ```sh
   dvc pull
   
