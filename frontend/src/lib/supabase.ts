'use client';

import { createClient } from '@supabase/supabase-js';
import { Session } from '@clerk/nextjs/server';

if (!process.env.NEXT_PUBLIC_SUPABASE_URL) {
  throw new Error('Missing env.NEXT_PUBLIC_SUPABASE_URL');
}
if (!process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY) {
  throw new Error('Missing env.NEXT_PUBLIC_SUPABASE_ANON_KEY');
}


interface FeedbackData {
  predicted_price: number;
  actual_price: number;
  confidence: number;
  notes?: string;
  property_data: Record<string, any>;
  deviation: number;
}

// Submit valuation feedbackk
export async function submitFeedback(feedback: FeedbackData, session: any) {
  
  const supabase = createClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      async accessToken() {
        return session?.getToken() ?? null
      },
    },
  );
  

  try {
    const { data, error } = await supabase
      .from('feedback')
      .insert([feedback]);
    
    if (error) throw error;

    const needsRetraining = await checkRetrainingNeeded(supabase);
    return { data, needsRetraining };
  } catch (error) {
    console.error('Error submitting feedback:', error);
    throw error;
  }
};

// Utility function to check if retraining is needed
export const checkRetrainingNeeded = async (supabase: any) => {
  try {
    const { data, error } = await supabase.rpc('check_retraining_needed');
    if (error) throw error;
    return data;
  } catch (error) {
    console.error('Error checking retraining status:', error);
    return false;
  }
}
