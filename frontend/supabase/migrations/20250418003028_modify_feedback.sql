-- Change the table's clerk_user_id column to user_id text not null default auth.jwt()->>'sub'
ALTER TABLE feedback
    DROP COLUMN IF EXISTS clerk_user_id,
    ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL DEFAULT auth.jwt()->>'sub';


-- DROP old polices
DROP POLICY IF EXISTS "Anyone can insert feedback with clerk_user_id" ON feedback;
DROP POLICY IF EXISTS "Users can view feedback" ON feedback;

-- New "Users can insert their own feedback" policy
create policy "Users can insert their own feedback"
ON "public"."feedback"
as 
PERMISSIVE
FOR INSERT
TO authenticated
with check (
((select auth.jwt()->>'sub') = (user_id)::text)
);

-- New "Users can view their own feedback" policy
create policy "Users can view their own feedback"
ON "public"."feedback"
for select
to authenticated
using(
((select auth.jwt()->>'sub') = (user_id)::text)
);