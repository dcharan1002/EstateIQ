# EstateIQ: Property Data Processing Pipeline

A robust data preprocessing pipeline for property data with feature engineering, data validation, and automated workflows.

## Environment Setup Instructions

1. **Prerequisites**
   - Docker and Docker Compose
   - Python 3.x
   - Git
   - DVC

2. **Local Setup**
   ```bash
   # Clone the repository
   git clone git@github.com:sshivaditya/EstateIQ.git
   cd EstateIQ

   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows

   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Docker Setup**
   ```bash
   # Build the Docker image
   docker build -t estateiq:latest .

   # Start services with Docker Compose
   docker compose up -d
   ```

4. **Environment Configuration**
   Create a `.env` file:
   ```bash
   # Airflow credentials
   _AIRFLOW_WWW_USER_USERNAME=airflow2
   _AIRFLOW_WWW_USER_PASSWORD=airflow2

   # Email Configuration (required for alerts)
   AIRFLOW__SMTP__SMTP_MAIL_FROM=your_email@gmail.com
   AIRFLOW__SMTP__SMTP_PASSWORD=your_app_password
   ```

   For the email configuration, you need to generate an app password from your email provider. Here are the instructions for [Gmail](https://support.google.com/mail/answer/185833?hl=en).

   For setting up Google Cloud Storage, you would need to set up a service account and download the JSON key file. You can set the environment variable `GOOGLE_APPLICATION_CREDENTIALS` to the path of the JSON key file. This will work if you have access to the Google Cloud Storage bucket.

   For setting up DVC with Google Cloud Storage, you need to set up the environment variables `GOOGLE_APPLICATION_CREDENTIALS` and `GOOGLE_PROJECT_ID` to the path of the JSON key file and the project ID respectively. And then change the dvc remote config as well.

## Pipeline Execution Steps

1. **Docker Build**
   ```bash
   docker compose build
   ```

2. **Start the container**
   ```bash
   docker compose up
   ```

   The pipeline executes the following steps:
   1. Downloads Boston housing data
   2. Cleans and preprocesses the data
   3. Engineers features
   4. Generates feature reports
   5. Saves final processed datasets

3. **Monitor Pipeline**
   - Access Airflow UI: http://localhost:8080
   - Monitor logs in `logs/` directory
   - Check feature reports in `data/features/`

## Code Structure

```
├── dags/               # Airflow DAG definitions
│   └── main_dag.py    # Main pipeline DAG
├── data/              # Data directories
│   ├── raw/          # Original, immutable data
│   ├── processed/    # Cleaned and transformed data
│   │   ├── clean/    # Cleaned data
│   │   └── engineered/ # Feature engineered data
│   ├── features/     # Extracted features
│   └── final/        # Final processed datasets
├── src/              # Source code
│   ├── data/        
│   │   └── download.py     # Data acquisition
│   ├── monitoring/
│   │   ├── config.json    # Monitoring configuration
│   │   └── logger.py      # Logging setup
│   └── preprocessing/     # Data preprocessing modules
│       ├── cleaning.py    # Data cleaning functions
│       ├── core.py       # Core preprocessing utilities
│       ├── features.py   # Feature engineering
│       └── reporting.py  # Data quality reporting
├── tests/            # Test suite
├── docker-compose.yaml    # Docker services config
├── Dockerfile            # Docker build instructions
├── dvc.yaml             # DVC pipeline definition
└── requirements.txt      # Python dependencies
```

### Key Components

1. **Data Processing Modules**
   - `cleaning.py`: Handles missing values, outliers, duplicates
   - `core.py`: Core preprocessing utilities
   - `features.py`: Feature engineering functions
   - `reporting.py`: Data quality reporting

2. **Pipeline Orchestration**
   - Airflow DAG with email notifications
   - Error handling and monitoring
   - Data quality checks

3. **Monitoring & Logging**
   - Comprehensive logging system
   - Data quality monitoring
   - Email alerts for failures

## Data Version Control (DVC)

1. **Data Pipeline Stages**
   DVC Stores data files and pipeline stages:
   - `data/raw/`: Original data files
   - `data/processed/`: Cleaned and transformed data
   - `data/features/`: Extracted features
   - `data/final/`: Final processed datasets


2. **DVC Commands**
   ```bash
   # Remove the exsisitng DVC Remote
   dvc remote remove myremote

   # Add a new DVC Remote
   dvc remote add -d myremote gs://estateiq-data

   # Run the Workflow

   # Add the data files to DVC
   dvc commit

   # Push the data to the remote
   dvc push

   ```

3. **Reproducibility**
   - All data transformations are tracked
   - Dependencies are captured in `dvc.lock`
   - Data versioning with DVC ensures reproducible results
   - Feature reports provide insights into data transformations

## Implementation Status

### Progress:

#### 1. Proper Documentation
Added a ReadME with the project structure. We have added docstring in all the relevant code blocks.

#### 2. Modular Syntax and Code
We separated the code into different files for tasks like cleaning, core functions, and feature engineering.

#### 3. Pipeline Orchestration (Airflow DAGs)
We built the main DAG with clear task dependencies. We added error handling and notifications, including email alerts for task failures. We also implemented success and failure callbacks.

#### 4. Tracking and Logging
We set up a detailed logging system in logger.py. We made sure the logs are saved to files and displayed on the console. We also added email alerts for critical errors.

#### 5. Data Version Control (DVC)
We initialized DVC with Google Cloud Storage as the remote. We configured data versioning and made sure the data folder structure is similar to one mentioned in the assignment. We also set up the .gitignore and .dvcignore files properly.

#### 6. Pipeline Flow Optimization
We optimized task dependencies in the DAG, ensuring tasks run in parallel when possible. 

#### 7. Schema and Statistics Generation
We implemented a system to generate feature reports and monitor data quality. Automatic statistics generation and data validation has been added, where system creates feature reports for the data and stores them in the Google Cloud Storage.

#### 8. Anomalies Detection and Alert Generation
We created the DataQualityMonitor class to detect anomalies, with configurable thresholds for outliers and missing values. We set up alerts to notify us of any data quality issues.

#### 9. Bias Detection and Mitigation
We made sure to preserve key demographic and location-based patterns so property values are accurate and reflect the market properly.

#### 10. Test Modules
We built a comprehensive set of tests, including unit tests for all key components. Test fixtures for common scenarios have been implemented.

#### 11. Error Handling and Logging
We implemented strong error handling throughout the project and added clear error messages. We set up logging for all errors and email alerts for critical issues.
