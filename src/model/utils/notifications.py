import os
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import smtplib
from pathlib import Path
from datetime import datetime
import traceback
import json

logger = logging.getLogger(__name__)

def create_bias_analysis_table(bias_report):
    """Format bias analysis results into an HTML table."""
    if not bias_report or 'full_results' not in bias_report:
        return ""
    
    html = """
    <h3>Bias Analysis Results</h3>
    <div style="margin: 20px 0;">
        <table style="width: 100%; border-collapse: collapse; border: 1px solid #ddd;">
            <tr style="background-color: #f5f5f5;">
                <th style="padding: 8px; border: 1px solid #ddd;">Feature</th>
                <th style="padding: 8px; border: 1px solid #ddd;">RMSE Ratio</th>
                <th style="padding: 8px; border: 1px solid #ddd;">R² Gap</th>
                <th style="padding: 8px; border: 1px solid #ddd;">Status</th>
            </tr>
    """
    
    for feature, results in bias_report['full_results'].items():
        disparities = results.get('disparities', {})
        rmse_ratio = disparities.get('rmse_ratio', 0)
        r2_gap = disparities.get('r2_gap', 0)
        
        # Determine status based on our thresholds
        has_bias = rmse_ratio > 1.2 or r2_gap > 0.1
        status = "❌" if has_bias else "✅"
        
        html += f"""
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;">{feature}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{rmse_ratio:.3f}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{r2_gap:.3f}</td>
                <td style="padding: 8px; border: 1px solid #ddd; text-align: center;">{status}</td>
            </tr>
        """
    
    html += "</table></div>"
    return html

def create_status_box(title, passed, details):
    """Create a styled status box for a specific check."""
    status_color = "#52c41a" if passed else "#f5222d"
    status_bg = "#f6ffed" if passed else "#fff2f0"
    status_border = "#b7eb8f" if passed else "#ffccc7"
    
    return f"""
        <div style="
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
            background-color: {status_bg};
            border: 1px solid {status_border};
        ">
            <h4 style="margin: 0 0 10px 0; color: {status_color};">
                {title}: {("✅ PASSED" if passed else "❌ FAILED")}
            </h4>
            <p style="margin: 0; color: #666;">{details}</p>
        </div>
    """

def create_training_report(model_name, metrics, plots, run_id, timestamp, validation_thresholds=None, bias_report=None):
    """Create a detailed training report in HTML format with embedded images."""
    # Calculate statuses
    performance_passed = all(
        metrics.get(metric, 0) >= threshold 
        for metric, threshold in (validation_thresholds or {}).items()
    )
    bias_detected = bias_report.get('bias_detected', False) if bias_report else False
    
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 20px; }}
            .status-grid {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
                margin: 20px 0;
            }}
            .metric-table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            .metric-table th, .metric-table td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            .metric-table th {{
                background-color: #f5f5f5;
            }}
            .plot-grid {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
                margin: 20px 0;
            }}
            .plot {{ width: 100%; }}
        </style>
    </head>
    <body>
    <h2>Model Training Report</h2>

    <h3>Evaluation Summary</h3>
    <div class="status-grid">
        {create_status_box(
            "Model Performance",
            performance_passed,
            "Model performance metrics meet defined thresholds" if performance_passed 
            else "One or more metrics below required thresholds"
        )}
        {create_status_box(
            "Fairness Check",
            not bias_detected,
            "No significant bias detected" if not bias_detected 
            else "Socioeconomic bias detected in model predictions"
        )}
    </div>

    <h3>Model Details</h3>
    <ul>
        <li><strong>Model Type:</strong> {model_name}</li>
        <li><strong>Training Time:</strong> {timestamp}</li>
        <li><strong>MLflow Run ID:</strong> {run_id}</li>
    </ul>
    """
    
    # Add metrics table
    if metrics:
        html += """
        <h3>Performance Metrics</h3>
        <table class="metric-table">
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Threshold</th>
                <th>Status</th>
            </tr>
        """
        
        for metric, value in metrics.items():
            threshold = validation_thresholds.get(metric, "N/A") if validation_thresholds else "N/A"
            passes = value >= threshold if isinstance(threshold, (int, float)) else True
            status = "✅" if passes else "❌"
            html += f"""
            <tr>
                <td>{metric}</td>
                <td>{value:.3f}</td>
                <td>{threshold}</td>
                <td>{status}</td>
            </tr>
            """
        html += "</table>"
    
    # Add bias analysis if available
    if bias_report:
        html += create_bias_analysis_table(bias_report)
    
    # Add visualizations
    if plots:
        html += """
        <h3>Model Visualizations</h3>
        <div class="plot-grid">
        """
        
        image_tags = {}
        for plot_name, plot_path in plots.items():
            if Path(plot_path).exists():
                img_id = Path(plot_path).stem
                html += f"""
                <div class="plot">
                    <h4>{plot_name}</h4>
                    <img src="cid:{img_id}" style="width: 100%; height: auto;" />
                </div>
                """
                image_tags[plot_name] = plot_path
        
        html += "</div>"
    
    html += """
    <p style="color: #666; font-size: 0.9em; margin-top: 30px;">
        <em>This is an automated report from EstateIQ Model Pipeline</em>
    </p>
    </body>
    </html>
    """
    
    return html, plots

def send_email_alert(subject, html_content, image_paths=None, success=False):
    """Send HTML email notification with embedded images using Gmail SMTP."""
    try:
        sender_email = os.getenv('GMAIL_USER', 'default@gmail.com')
        sender_password = os.getenv('GMAIL_APP_PASSWORD', '')
        recipients = os.getenv('NOTIFICATION_EMAIL', 'default@gmail.com').split(',')

        if not all([sender_email, sender_password]) or '@gmail.com' not in sender_email:
            logger.warning("Email configuration missing or invalid, skipping notification")
            return

        msg = MIMEMultipart('related')
        msg['From'] = sender_email
        msg['To'] = ', '.join(recipients)
        msg['Subject'] = f"{subject}"
        
        msg_alternative = MIMEMultipart('alternative')
        msg.attach(msg_alternative)
        msg_alternative.attach(MIMEText(html_content, 'html'))

        if image_paths:
            for path in image_paths.values():
                if Path(path).exists():
                    with open(path, 'rb') as f:
                        img = MIMEImage(f.read())
                        img.add_header('Content-ID', f'<{Path(path).stem}>')
                        msg.attach(img)

        try:
            server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipients, msg.as_string())
            server.close()
            logger.info("Email alert sent successfully")
        except smtplib.SMTPAuthenticationError as e:
            logger.warning(f"Email authentication failed: {str(e)}")
        except Exception as e:
            logger.error(f"Email sending failed: {str(e)}")
            
    except Exception as e:
        logger.error(f"Failed to prepare email alert: {e}")
        logger.error(traceback.format_exc())

def notify_training_completion(
    model_name,
    metrics,
    plots,
    run_id,
    validation_thresholds,
    bias_report=None,
    success=True
):
    """Send training completion notification with detailed report and visualizations."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    html_content, image_paths = create_training_report(
        model_name,
        metrics,
        plots,
        run_id,
        timestamp,
        validation_thresholds,
        bias_report
    )
    
    # Create subject with separate performance and bias status
    performance_passed = all(
        metrics.get(metric, 0) >= threshold 
        for metric, threshold in validation_thresholds.items()
    )
    bias_detected = bias_report.get('bias_detected', False) if bias_report else False
    
    status_str = (
        f"{'✅' if performance_passed else '❌'} Performance | "
        f"{'✅' if not bias_detected else '❌'} Fairness"
    )
    subject = f"Model Training Report - {status_str}"
    
    send_email_alert(subject, html_content, image_paths, success=performance_passed)

def notify_error(error, context=None):
    """Send error notification with detailed report."""
    html_content = create_error_report(error, context)
    subject = "❌ Model Training Error"
    send_email_alert(subject, html_content, success=False)

def create_error_report(error, context=None):
    """Create a detailed HTML error report."""
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 20px; }}
            .error-box {{ 
                background-color: #fff1f0;
                border: 1px solid #ffa39e;
                padding: 15px;
                margin: 10px 0;
                border-radius: 4px;
            }}
            .context-box {{
                background-color: #f6ffed;
                border: 1px solid #b7eb8f;
                padding: 15px;
                margin: 10px 0;
                border-radius: 4px;
            }}
            .traceback {{
                background-color: #f5f5f5;
                padding: 15px;
                margin: 10px 0;
                border-radius: 4px;
                font-family: monospace;
                white-space: pre-wrap;
                overflow-x: auto;
            }}
        </style>
    </head>
    <body>
    <h2>❌ Pipeline Error Report</h2>
    
    <div class="error-box">
        <h3>Error Details</h3>
        <p><strong>Type:</strong> {type(error).__name__}</p>
        <p><strong>Message:</strong> {str(error)}</p>
    </div>
    """
    
    if context:
        html += """
        <div class="context-box">
            <h3>Execution Context</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">Parameter</th>
                    <th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">Value</th>
                </tr>
        """
        for key, value in context.items():
            html += f"""
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>{key}</strong></td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;">{value}</td>
                </tr>
            """
        html += "</table></div>"
    
    html += f"""
    <h3>Traceback</h3>
    <div class="traceback">{traceback.format_exc()}</div>
    
    <p style="color: #666; font-size: 0.9em; margin-top: 20px;">
        <em>Error occurred at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em>
    </p>
    </body>
    </html>
    """
    return html
