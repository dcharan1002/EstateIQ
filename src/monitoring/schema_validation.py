import tensorflow_data_validation as tfdv
import os

def validate_schema_and_generate_statistics(raw_data_path: str, schema_path: str, stats_path: str) -> str:
    """
    Validate data schema and generate statistics.

    Args:
    - raw_data_path (str): Path to the raw dataset.
    - schema_path (str): Path to save or load the schema.
    - stats_path (str): Path to save the statistics.

    Returns:
    - str: Success or error message.
    """
    # Make sure directories exist
    os.makedirs(os.path.dirname(schema_path), exist_ok=True)

    # Load dataset and generate statistics
    try:
        data = tfdv.generate_statistics_from_csv(raw_data_path)
        tfdv.write_stats_text(data, stats_path)
        print(f"Statistics saved to {stats_path}")
        
        # Generate schema if not already present
        if not os.path.exists(schema_path):
            schema = tfdv.infer_schema(data)
            tfdv.write_schema_text(schema, schema_path)
            print(f"Schema generated and saved to {schema_path}")
        else:
            print(f"Schema already exists at {schema_path}")
        
        # Load and compare with new dataset for validation
        schema = tfdv.load_schema_text(schema_path)
        anomalies = tfdv.validate_statistics(data, schema)

        if anomalies.anomaly_info:
            print("Data Anomalies Detected:")
            print(anomalies)
            return "Anomalies Detected"
        else:
            print("Data validation passed with no anomalies.")
            return "Data validation passed"
    
    except Exception as e:
        print(f"Error during schema validation or statistics generation: {e}")
        return str(e)


# Usage Example:
raw_data_path = "/data/raw/dataset.csv"  # Replace with actual dataset location
schema_path = "/data/schema/schema.pbtxt"
stats_path = "/data/schema/stats.tfrecord"

validate_schema_and_generate_statistics(raw_data_path, schema_path, stats_path)
