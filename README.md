# EstateIQ: Property Data Processing Pipeline

A robust data preprocessing and model development pipeline for property data with feature engineering, data validation, and automated workflows.

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
   Copy `.env.example` to `.env` and configure the following required variables:

   ```bash
   # GCP Configuration (Required)
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/credentials.json
   GOOGLE_CLOUD_PROJECT=estateiqclone
   REGION=us-central1
   ARTIFACT_REGISTRY=estateiq-models
   MODEL_REGISTRY_PATH=models/estate_price_prediction
   SERVICE_ACCOUNT=your-service-account@estateiqclone.iam.gserviceaccount.com

   # Model Serving (Required)
   MODEL_DIR=/app/models
   PORT=8080

   # Email Notifications (Required)
   GMAIL_APP_PASSWORD=your-app-password
   NOTIFICATION_EMAIL=your.email@gmail.com
   GMAIL_USER=your.email@gmail.com

   # MLflow Configuration (Required)
   MLFLOW_TRACKING_URI=sqlite:///mlflow.db
   MLFLOW_EXPERIMENT_NAME=estate_price_prediction

   # DVC Remote (Required)
   DVC_REMOTE_URL=gs://your-project-id-dvc

   # Model Validation Thresholds (Required)
   VALIDATION_R2_THRESHOLD=0.8
   VALIDATION_RMSE_THRESHOLD=1.0
   VALIDATION_MAE_THRESHOLD=0.8

   # Airflow Configuration (Required from docker-compose.yaml)
   AIRFLOW_UID=50000
   _AIRFLOW_WWW_USER_USERNAME=airflow2
   _AIRFLOW_WWW_USER_PASSWORD=airflow2
   ```

   **Important Setup Notes:**
   1. For GCP configuration, you need to:
      - Create a service account and download the credentials JSON
      - Set up an Artifact Registry repository
      - Configure the appropriate IAM permissions

   2. For email notifications:
      - Generate an app password from your Gmail account ([Instructions](https://support.google.com/mail/answer/185833?hl=en))
      - Use the same app password for both GMAIL_APP_PASSWORD and AIRFLOW__SMTP__SMTP_PASSWORD

   3. For DVC:
      - Create a Google Cloud Storage bucket for DVC remote storage
      - Update DVC_REMOTE_URL with your bucket URL

## Required Secrets and Configuration

### GitHub Secrets
```
GCP_SA_KEY: Service account JSON key for GCP authentication
GMAIL_APP_PASSWORD: Gmail app password for notifications
NOTIFICATION_EMAIL: Email address for notifications
```

### Configuration Steps
1. **GCP Setup**
   ```bash
   # Run GCP setup script
   ./scripts/setup_gcp.sh
   
   # This will:
   # - Enable required APIs
   # - Create Artifact Registry repository
   # - Set up service account and permissions
   # - Create necessary storage buckets
   ```

## Utility Scripts

1. **GCP Setup and Deployment**
   ```bash
   # Initial GCP setup
   ./scripts/setup_gcp.sh
   
   # Deploy to Cloud Run
   ./scripts/start_service.sh [options]
   Options:
   --project-id        GCP project ID
   --region           GCP region
   --service-name     Cloud Run service name
   --artifact-registry Artifact registry name
   --model-path       Model registry path
   ```

2. **Local Testing**
   ```bash
   # Build Model
   python -m src.model.train

   # Test model locally
   ./scripts/test_local.sh
   
   # Test specific endpoints
   ./scripts/test_inference.py --local
   ```

3. **Model Training and Deployment**
   ```bash
   # Submit Cloud Build job
   ./scripts/submit_build.sh
   
   # Start model service
   ./scripts/start_service.sh
   
   # Update workflow environment variables
   ./scripts/update_workflow_env.sh   # Updates environment variables in GitHub workflows
   ```

   Note: The update_workflow_env.sh script synchronizes environment variables (Non Secret Public Values) between your .env file and GitHub workflow files (model-training.yml and model-deployment.yml). Run this script after making changes to your environment configuration to ensure your workflows stay up-to-date.

## Project Structure

```
├── dags/                        # Airflow DAG definitions
│   └── main_dag.py             # Main pipeline DAG
├── data/                       # Data directories
│   ├── raw/                   # Original, immutable data
│   ├── processed/             # Cleaned and transformed data
│   │   ├── clean/            # Cleaned data
│   │   └── engineered/       # Feature engineered data
│   ├── features/             # Extracted features
│   └── final/                # Final processed datasets
├── src/                       # Source code
│   ├── data/        
│   │   └── download.py       # Data acquisition
│   ├── deployment/           # Model deployment service
│   │   ├── app.py           # Flask prediction API
│   │   ├── Dockerfile       # Deployment container
│   │   └── load_model.py    # Model loading with retries
│   ├── model/               # Model development
│   │   ├── train.py         # Training orchestration
│   │   ├── models.py        # Model registry & factory
│   │   ├── validate.py      # Validation with metrics
│   │   ├── base.py         # Base model interface
│   │   ├── random_forest.py # RandomForest implementation
│   │   ├── xgboost_model.py # XGBoost implementation
│   │   └── utils/          # ML utilities
│   │       ├── bias_analysis.py    # Bias detection & mitigation
│   │       ├── metrics.py          # Performance metrics
│   │       ├── notifications.py    # Email alerts
│   │       ├── shap_analysis.py    # Feature importance
│   │       └── visualization.py    # Results plotting
│   ├── monitoring/          # System monitoring
│   │   ├── config.json     # Monitoring configuration
│   │   └── logger.py       # Centralized logging
│   └── preprocessing/      # Data preprocessing
│       ├── cleaning.py    # Data cleaning pipeline
│       ├── core.py       # Core utilities
│       ├── features.py   # Feature engineering
│       └── reporting.py  # Quality reporting
├── scripts/               # Utility scripts
│   ├── setup_gcp.sh     # GCP environment setup
│   ├── start_service.sh # Cloud Run deployment
│   ├── test_local.sh   # Local testing
│   └── test_inference.py # Endpoint testing
├── tests/               # Test suite
├── docker-compose.yaml  # Airflow Orchestrator
├── Dockerfile          # Main build Image for Airflow
├── dvc.yaml           # Data versioning pipeline
└── requirements.txt    # Python dependencies
```

<details>
<summary><b>Data Pipeline</b></summary>

## Pipeline Execution Steps

1. **Start Pipeline**
   ```bash
   docker compose up
   ```

   The pipeline executes the following steps:
   1. Downloads Boston housing data
   2. Cleans and preprocesses the data
   3. Engineers features
   4. Generates feature reports
   5. Saves final processed datasets

2. **Monitor Pipeline**
   - Access Airflow UI: http://localhost:8080
   - Monitor logs in `logs/` directory
   - Check feature reports in `data/features/`

### Data Pipeline Components

1. **Data Processing Modules**
   - `cleaning.py`: Handles missing values, outliers, duplicates
   - `core.py`: Core preprocessing utilities
   - `features.py`: Feature engineering functions
   - `reporting.py`: Data quality reporting

2. **Pipeline Orchestration**
   We built Airflow DAGs to run our pipeline. The DAGs send alerts when tasks fail. Quality checks happen after each step.

3. **Monitoring & Logging**
   We set up logs to track all jobs. System sends alerts for any data issues like missing value, outliers and other quality issues..

## Data Version Control (DVC)

1. **Data Pipeline Stages**
   DVC Stores data files and pipeline stages:
   - `data/raw/`: Original data files
   - `data/processed/`: Cleaned and transformed data
   - `data/features/`: Extracted features
   - `data/final/`: Final processed datasets

2. **DVC Commands**
   ```bash
   # Remove the exsisitng DVC Remote (If you want to set up a new remote)
   dvc remote remove myremote

   # Add a new DVC Remote (Google Cloud Storage or any other provider)
   dvc remote add -d myremote gs://estateiq-data

   # Run the Workflow

   # Add the data files to DVC 
   dvc commit

   # Push the data to the remote
   dvc push
   ```

## Implementation Status

### Progress:

#### 1. Proper Documentation
- Added comprehensive README with project structure
- Included docstrings in all data processing modules
- Documented data transformations and pipeline stages

#### 2. Modular Syntax and Code
- Separated preprocessing into distinct modules (cleaning, core, features)
- Created reusable utility functions
- Implemented clear module interfaces

#### 3. Pipeline Orchestration (Airflow DAGs)
- Built main DAG with clear task dependencies
- Added error handling and notifications
- Implemented success and failure callbacks
- Email alerts for task failures

#### 4. Tracking and Logging
- Detailed logging system in logger.py
- File and console logging
- Email alerts for critical errors
- Comprehensive execution tracking

#### 5. Data Version Control (DVC)
- Initialized DVC with Google Cloud Storage
- Configured data versioning
- Structured data folders per requirements
- Set up .gitignore and .dvcignore

#### 6. Pipeline Flow Optimization
- Optimized DAG task dependencies
- Enabled parallel task execution
- Minimized pipeline bottlenecks

#### 7. Schema and Statistics Generation
- Feature report generation system
- Data quality monitoring
- Automatic statistics generation
- Cloud storage integration

#### 8. Anomalies Detection
- DataQualityMonitor implementation
- Configurable outlier thresholds
- Missing value detection
- Automated quality alerts

</details>

<details>
<summary><b>Model Development Pipeline</b></summary>

## Google Cloud SDK (gcloud CLI) Setup

Before implementing the pipeline, you need to install and configure the Google Cloud CLI (gcloud):

### Installing gcloud CLI

1. **MacOS** (using Homebrew):
   ```bash
   brew install --cask google-cloud-sdk
   ```

2. **Windows**:
   - Download the Google Cloud SDK installer from: https://cloud.google.com/sdk/docs/install
   - Run the installer and follow the prompts
   - Restart your terminal/command prompt after installation

3. **Linux** (Debian/Ubuntu):
   ```bash
   # Add the Cloud SDK distribution URI as a package source
   echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

   # Import the Google Cloud public key
   curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

   # Update and install the Cloud SDK
   sudo apt-get update && sudo apt-get install google-cloud-sdk
   ```

### Initial gcloud Setup

After installation:

1. **Initialize gcloud**:
   ```bash
   gcloud init
   ```
   This will:
   - Log you into your Google Cloud account
   - Set up your default project
   - Configure your default region

2. **Verify Installation**:
   ```bash
   gcloud --version
   ```

3. **Authentication**:
   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```

### Script Requirements

The following scripts require gcloud CLI to be installed and configured:

1. `setup_gcp.sh`: Sets up GCP resources and environment
   - Requirements: gcloud CLI with project owner permissions
   - Usage: `./scripts/setup_gcp.sh`

2. `test_gcp.sh`: Tests GCP connectivity and permissions
   - Requirements: gcloud CLI, valid authentication
   - Usage: `./scripts/test_gcp.sh`

3. `start_service.sh`: Deploys to Cloud Run
   - Requirements: gcloud CLI, Cloud Run API enabled
   - Usage: `./scripts/start_service.sh`

4. `submit_build.sh`: Submits Cloud Build job
   - Requirements: gcloud CLI, Cloud Build API enabled
   - Usage: `./scripts/submit_build.sh`

5. `test_local.sh`: Tests model locally
   - Requirements: Python environment, local dependencies
   - Usage: `./scripts/test_local.sh`

## Pipeline Implementation

1. **Docker Containerization**
   Container images are generated using Cloud Build and published to Artifact Registry, then deployed to Cloud Run for scalable, serverless execution.

2. **Data Loading Pipeline**
   Our DVC integration provides versioned data access with automated validation checks. We track all data transformations for reproducibility.

3. **Model Training and Selection**
   We train both RandomForest and XGBoost models using MLflow for experiment tracking. Our selection process uses performance metrics to choose the best model.

4. **Model Validation**
   We validate models using cross-validation and a separate validation dataset, comparing results against defined thresholds.

5. **Bias Detection and Mitigation**
   We check for bias using data slicing and SHAP analysis, applying automated mitigation when needed. This ensures fair model predictions.

6. **Post-Bias Model Selection**
   Our selection process considers both performance and fairness metrics to pick models that balance accuracy with fairness.

7. **Model Registry Integration**
   We store models in GCP's Artifact Registry with versioning and metadata tracking for easy deployment.

## Pipeline Execution Steps

### Local Build
1. **Training and Validation**
   ```bash
   # Start containerized training
   ./scripts/submit_build.sh
   ```

   Pipeline stages:
   1. Load versioned data from DVC
   2. Train multiple model architectures
   3. Validate models on test set
   4. Perform bias analysis
   5. Select best model
   6. Push to Artifact Registry

   Requirements:
   1. GCP project with enabled APIs
   2. DVC data in the Data Folder
   3. .env file with all credentials

2. **Model Deployment**
   ```bash
   # Deploy to Cloud Run
   ./scripts/start_service.sh
   ```

   Deployment includes:
   1. Pull model from Artifact Registry
   2. Start prediction service
   3. Configure monitoring
   4. Enable auto-scaling

    Requirements:
    1. GCP project with enabled APIs
    2. .env file with all credentials

### Cloud Build

1. **Setup Github Workflow**
   - Triggered on push to main branch
   - Builds container image
   - Pushes image to Artifact Registry

   Requirements:
   1. Setup the Secrets in Github
   2. Setup project first with `scripts/setup_gcp.sh`
   3. Run `scripts/update_workflow_env.sh` to update the workflow with environment variables.
   4. Ensure the DVC data is stored in the storage bucket.

**Note:** The above steps are for the local build and cloud build. The cloud build is triggered on the push to the main branch. The cloud build will build the container image and push it to the Artifact Registry. The cloud run service will be updated with the new image.

GCloud cli is required to be installed on the local machine to run the above commands. The GCloud cli can be installed from the instructions before.


</details>
