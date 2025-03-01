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

## Key Features
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
```bash
# Build with cache
docker build -t estateiq:latest .

# Force clean build
docker build --no-cache -t estateiq:latest .

# Run with Docker Compose
docker compose build && docker compose up
```

## Pipeline Components

### 1. Data Preprocessing
- Property condition standardization
- Address parsing and normalization
- Spatial feature generation
- Property age calculations

### 2. Feature Engineering
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
- Anomaly detection with configurable thresholds
- Bias detection and pattern preservation

## Development

### Running Tests
```bash
# Run all tests with coverage
python -m pytest
```

## Configuration

### Environment Variables
Create a `.env` file in the project root:
```bash
# Airflow Default User
_AIRFLOW_WWW_USER_USERNAME=airflow2
_AIRFLOW_WWW_USER_PASSWORD=airflow2

# Email Configuration
AIRFLOW__SMTP__SMTP_MAIL_FROM=your_email@gmail.com
AIRFLOW__SMTP__SMTP_PASSWORD=your_generated_app_password
```

### Email Notifications
1. Generate App Password for SMTP:
   - Visit https://support.google.com/accounts/answer/185833
   - Sign in to your Google Account
   - Enable 2-Step Verification if needed
   - Under "App passwords", select "Mail" and your device
   - Use the generated password as AIRFLOW__SMTP__SMTP_PASSWORD

## Workflow Automation

### Airflow DAG Structure
The main DAG (`dags/main_dag.py`) orchestrates:
1. Data download
2. Data cleaning
3. Feature engineering
4. Quality reporting

### Error Handling and Logging
- Comprehensive logging system in logger.py
- Rotating file handlers
- Email notifications for critical errors
- Task failure handling and retries

## Data Version Control (DVC)

### Setup with Google Cloud Storage
1. Install Google Cloud SDK from https://cloud.google.com/sdk/docs/install
2. Configure authentication:
```bash
gcloud auth login
```

### Usage
```bash
# Track and commit changes
dvc commit

# Push data to GCP bucket
dvc push

# Pull data from GCP bucket
dvc pull
```

Note: Ensure you have proper GCP authentication and permissions configured before using DVC with Google Cloud Storage.


## Implementation Status

### 1. Proper Documentation
- Clear README with project structure, features, and setup instructions
- Well-documented code with comprehensive docstrings

### 2. Modular Syntax and Code
- Preprocessing modules separated by functionality (cleaning.py, core.py, features.py)
- Reusable functions and classes with clear interfaces
- Consistent code style and organization

### 3. Pipeline Orchestration (Airflow DAGs)
- Main DAG with clear task dependencies
- Error handling and notifications
- Email alerts for task failures
- Success/failure callbacks implemented

### 4. Tracking and Logging
- Comprehensive logging system in logger.py
- Rotating file handlers for log management
- Console and file logging
- Email notifications for critical errors

### 5. Data Version Control (DVC)
- DVC initialized with Google Cloud Storage remote
- Data versioning configured
- Clear data directory structure
- .gitignore and .dvcignore properly configured

### 6. Pipeline Flow Optimization
- Efficient task dependencies in DAG
- Parallel execution where possible
- Monitoring for bottlenecks
- Task failure handling and retries

### 7. Schema and Statistics Generation
- Feature reporting system
- Data quality monitoring
- Automated statistics generation
- Data validation checks

### 8. Anomalies Detection and Alert Generation
- DataQualityMonitor class for anomaly detection
- Configurable thresholds for outliers
- Missing value detection
- Alert system for data quality issues

### 9. Bias Detection and Mitigation
- Intentionally preserve demographic and location-based patterns
- This retention is necessary for generating accurate market-reflective property values

### 10. Test Modules
- Comprehensive test suite
- Unit tests for all key components
- Test fixtures for common scenarios
- Edge case testing

### 11. Error Handling and Logging
- Robust error handling throughout
- Clear error messages
- Comprehensive logging
- Email notifications for critical errors
