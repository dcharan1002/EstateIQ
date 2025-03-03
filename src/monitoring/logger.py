import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from email.mime.text import MIMEText
import smtplib
import json

# Configure logging
def setup_logger(name: str) -> logging.Logger:
    """Set up logger with file and console handlers"""
    # Use Airflow's log directory when running in Docker
    log_dir = Path("/opt/airflow/logs")
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_dir / f"{name}.log",
        maxBytes=10485760,  # 10MB
        backupCount=5,
        mode='a+',  # Append mode with creation
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class DataQualityMonitor:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        # Load configuration if exists
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Load monitoring configuration from JSON file"""
        config_path = Path("src/monitoring/config.json")
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        return {
            "missing_threshold": 0.1,  # Max allowed missing value ratio
            "numeric_bounds": {
                "TOTAL_VALUE": {"min": 50000, "max": 5000000},
                "GROSS_AREA": {"min": 100, "max": 20000},
                "LAND_VALUE": {"min": 10000, "max": 2000000},
                "BLDG_VALUE": {"min": 10000, "max": 3000000}
            }
        }
    
    def check_missing_values(self, df: pd.DataFrame) -> List[str]:
        """Check for columns with too many missing values"""
        missing_ratio = df.isnull().mean()
        issues = []
        
        for col, ratio in missing_ratio.items():
            if ratio > self.config["missing_threshold"]:
                msg = f"High missing values in {col}: {ratio:.2%}"
                issues.append(msg)
                self.logger.warning(msg)
        
        return issues

    def check_numeric_bounds(self, df: pd.DataFrame) -> List[str]:
        """Check if numeric values are within expected bounds"""
        issues = []
        
        for col, bounds in self.config["numeric_bounds"].items():
            if col not in df.columns:
                continue
                
            outliers = df[
                (df[col] < bounds["min"]) | 
                (df[col] > bounds["max"])
            ]
            
            if len(outliers) > 0:
                msg = f"Found {len(outliers)} outliers in {col}"
                issues.append(msg)
                self.logger.warning(msg)
                
        return issues

    def check_data_quality(self, df: pd.DataFrame) -> List[str]:
        """Run all data quality checks"""
        issues = []
        issues.extend(self.check_missing_values(df))
        issues.extend(self.check_numeric_bounds(df))
        return issues

class AlertManager:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        
    def send_email_alert(self, subject: str, body: str):
        """Send email alert using Airflow's email system"""
        try:
            from airflow.operators.email import EmailOperator
            from airflow.configuration import conf
            
            # Get recipients from Airflow config
            recipients = conf.get('email', 'email_backend_recipients', 
                                fallback='sshivaditya@gmail.com').split(',')
            
            # Create and execute email operator
            email_op = EmailOperator(
                task_id='send_alert_email',
                to=recipients,
                subject=subject,
                html_content=f"""
                    <h3>{subject}</h3>
                    <p>{body.replace(chr(10), '<br>')}</p>
                """,
                dag=None  # No DAG context needed for direct execution
            )
            email_op.execute(context={})
            self.logger.info(f"Alert email sent: {subject}")
            
        except Exception as e:
            self.logger.warning(
                f"Could not send email alert (will continue processing): {str(e)}\n"
                f"Alert content - Subject: {subject}, Body: {body}"
            )
