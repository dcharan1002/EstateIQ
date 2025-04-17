-- Create feedback table
CREATE TABLE IF NOT EXISTS feedback (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    predicted_price DECIMAL(15,2) NOT NULL,
    actual_price DECIMAL(15,2) NOT NULL,
    confidence SMALLINT NOT NULL CHECK (confidence >= 1 AND confidence <= 10),
    notes TEXT,
    property_data JSONB NOT NULL,
    deviation DECIMAL(10,4) NOT NULL,
    user_id UUID REFERENCES auth.users(id)
);

-- Create index on deviation for monitoring queries
CREATE INDEX feedback_deviation_idx ON feedback(deviation);

-- Create index on created_at for time-based queries
CREATE INDEX feedback_created_at_idx ON feedback(created_at);

-- Enable Row Level Security
ALTER TABLE feedback ENABLE ROW LEVEL SECURITY;

-- Create policies
CREATE POLICY "Users can insert their own feedback"
ON feedback FOR INSERT
TO authenticated
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can view their own feedback"
ON feedback FOR SELECT
TO authenticated
USING (auth.uid() = user_id);

-- Create function to check if retraining is needed
CREATE OR REPLACE FUNCTION check_retraining_needed()
RETURNS boolean
LANGUAGE plpgsql
AS $$
DECLARE
    rmse DECIMAL;
    mae DECIMAL;
    avg_deviation DECIMAL;
    feedback_count INTEGER;
BEGIN
    -- Calculate metrics from feedback with high confidence
    SELECT 
        SQRT(AVG(POWER(actual_price - predicted_price, 2))) as rmse,
        AVG(ABS(actual_price - predicted_price)) as mae,
        AVG(deviation) as avg_deviation,
        COUNT(*) as count
    INTO rmse, mae, avg_deviation, feedback_count
    FROM feedback
    WHERE confidence >= 7
    AND created_at >= NOW() - INTERVAL '30 days';

    -- Check against thresholds (these can be moved to a config table)
    RETURN feedback_count >= 50 
        AND (rmse > 150000 OR mae > 100000 OR avg_deviation > 0.3);
END;
$$;

-- Create metrics view for monitoring
CREATE VIEW feedback_metrics AS
SELECT 
    DATE_TRUNC('day', created_at) as date,
    COUNT(*) as feedback_count,
    AVG(deviation) as avg_deviation,
    SQRT(AVG(POWER(actual_price - predicted_price, 2))) as rmse,
    AVG(ABS(actual_price - predicted_price)) as mae,
    AVG(confidence) as avg_confidence
FROM feedback
GROUP BY DATE_TRUNC('day', created_at)
ORDER BY date DESC;
