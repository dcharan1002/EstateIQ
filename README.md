# EstateIQ: Property Data Processing Pipeline

A robust data preprocessing pipeline for property data with feature engineering, data validation, and automated workflows.

## Project Structure
```
├── dags/               # Airflow DAG definitions
├── data/              # Data directories
│   ├── raw/          # Original, immutable data
│   ├── processed/    # Cleaned and transformed data
│   │   ├── clean/    # Cleaned data
│   │   └── engineered/ # Feature engineered data
│   ├── features/     # Extracted features
│   └── final/        # Final processed datasets
├── src/              # Source code
│   ├── data/        # Data acquisition scripts
│   └── preprocessing/# Data preprocessing modules
│       ├── cleaning.py     # Data cleaning functions
│       ├── core.py        # Core preprocessing utilities
│       ├── features.py    # Feature engineering
│       └── reporting.py   # Data quality reporting
├── tests/            # Test suite
├── docker-compose.yaml    # Docker services configuration
├── Dockerfile            # Docker build instructions
├── dvc.yaml             # DVC pipeline definition
└── requirements.txt      # Python dependencies
```

## Features
- Automated data cleaning and preprocessing
- Feature engineering for property data
- Data validation and quality checks
- Containerized environment
- Airflow workflow automation
- DVC data versioning
- Comprehensive test coverage

## Installation

### Local Setup
1. Clone the repository:
```bash
git clone git@github.com:sshivaditya/EstateIQ.git
cd EstateIQ
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Docker Setup
1. Build and start services:
```bash
docker-compose up --build
```

## Data Pipeline

### 1. Data Preprocessing
The pipeline handles:
- Property condition standardization
- Address parsing and normalization
- Spatial feature generation
- Property age calculations
- Feature engineering

### 2. Feature Engineering
Generated features include:
- Property age categories
- Distance to city center
- Price per square foot
- Land value ratios
- Room statistics

### 3. Quality Control
- Data validation checks
- Missing value handling
- Outlier detection
- Data type consistency

## Development

### Running Tests
```bash
# Run all tests with coverage
pytest

# Run specific test file
pytest tests/test_core.py

# Run specific test
pytest tests/test_core.py::test_identify_column_types_basic
```

## Airflow Integration

### DAG Structure
The main DAG (`dags/main_dag.py`) includes:
1. Data download
2. Data cleaning
3. Feature engineering
4. Quality reporting

### Monitoring
Access Airflow UI at http://localhost:8080:
- Username: airflow2
- Password: airflow2

## Data Version Control

### Track Data Changes
```bash
# Update pipeline
dvc repro

# Push to remote storage
dvc push
```

## Docker Usage

### Running Services
```bash
# Start all services
docker-compose up

# Run specific service
docker-compose up airflow-webserver

# Stop services
docker-compose down
```
## License
MIT License - see LICENSE file for details
