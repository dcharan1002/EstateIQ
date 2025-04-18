-- Drop old RLS policies
DROP POLICY IF EXISTS "Users can insert their own feedback" ON feedback;
DROP POLICY IF EXISTS "Users can view their own feedback" ON feedback;

-- Add clerk_user_id column
ALTER TABLE feedback
    DROP COLUMN IF EXISTS user_id,
    ADD COLUMN IF NOT EXISTS clerk_user_id TEXT NOT NULL DEFAULT '';

-- Create index on clerk_user_id
CREATE INDEX IF NOT EXISTS feedback_clerk_user_idx ON feedback(clerk_user_id);

-- Create new RLS policies for Clerk auth
CREATE POLICY "Anyone can insert feedback with clerk_user_id"
ON feedback FOR INSERT
TO authenticated
WITH CHECK (true);

CREATE POLICY "Users can view feedback"
ON feedback FOR SELECT
TO authenticated
USING (true);

-- Remove the foreign key constraint if it exists
DO $$ 
BEGIN
    IF EXISTS (
        SELECT 1 
        FROM information_schema.table_constraints 
        WHERE constraint_name = 'feedback_user_id_fkey'
    ) THEN
        ALTER TABLE feedback DROP CONSTRAINT feedback_user_id_fkey;
    END IF;
END $$;