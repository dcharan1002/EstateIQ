import { useRef, useState } from 'react';
import dynamic from 'next/dynamic';
import { downloadPDF } from '@/lib/downloadPDF';
import { useTheme } from 'next-themes';
import { submitFeedback } from '@/lib/supabase';
import { useSession, useUser } from "@clerk/nextjs";
// Dynamically import PropertyMap with SSR disabled
const PropertyMap = dynamic(() => import('./PropertyMap'), { ssr: false });

import { FormData as ValuationReportPropForm } from '@/app/(dashboard)/valuation/page';

interface ValuationReportProps {
  formData: ValuationReportPropForm;
  prediction: {
    prediction: number;
  };
}

interface FeedbackData {
  actualPrice: number;
  confidence: number;
  notes: string;
  propertyData: ValuationReportPropForm;
  predictedPrice: number;
}

// Interface matching Supabase schema
interface SupabaseFeedbackData {
  predicted_price: number;
  actual_price: number;
  confidence: number;
  notes: string;
  property_data: Record<string, any>;
  deviation: number;
}
const ValuationReport: React.FC<ValuationReportProps> = ({ formData, prediction }) => {
  const { user } = useUser();
  const {session} = useSession();
  const reportRef = useRef<HTMLDivElement>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);
  const [feedback, setFeedback] = useState<FeedbackData>({
    actualPrice: 0,
    confidence: 5,
    notes: "",
    propertyData: formData,
    predictedPrice: prediction.prediction
  });
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleFeedbackSubmit = async () => {
    setSubmitError(null);
    setIsSubmitting(true);

    // Validate actual price is within reasonable bounds
    if (feedback.actualPrice < 50000 || feedback.actualPrice > 5000000) {
      setSubmitError("Please enter a valid price between $50,000 and $5,000,000");
      setIsSubmitting(false);
      return;
    }

    // Validate if the reported price is within reasonable deviation from prediction
    const deviation = Math.abs(feedback.actualPrice - feedback.predictedPrice) / feedback.predictedPrice;
    if (deviation > 0.5) { // More than 50% difference
      const confirm = window.confirm(
        "The price you entered differs significantly from our prediction. " +
        "Are you sure this is correct? Click OK to submit anyway, or Cancel to revise."
      );
      if (!confirm) {
        setIsSubmitting(false);
        return;
      }
    }

    if (!user || !session) {
      setSubmitError('You must be logged in to submit feedback');
      setIsSubmitting(false);
      return;
    }

    try {
      // Transform feedback data to match Supabase schema
      const supabaseFeedback: SupabaseFeedbackData = {
        predicted_price: feedback.predictedPrice,
        actual_price: feedback.actualPrice,
        confidence: feedback.confidence,
        notes: feedback.notes,
        property_data: feedback.propertyData,
        deviation: deviation,
      };

      const { needsRetraining } = await submitFeedback(supabaseFeedback, session);

      setFeedbackSubmitted(true);

      if (needsRetraining) {
        console.log("Model retraining triggered due to feedback metrics");
      }
    } catch (err) {
      setSubmitError('Failed to submit feedback. Please try again.');
      console.error('Feedback submission error:', err);
    } finally {
      setIsSubmitting(false);
    }
  };

  const { theme } = useTheme();
  
  const handleDownload = async () => {
    if (!reportRef.current) return;
    setIsGenerating(true);
    setError(null);

    try {
      // Create a clone of the report element for PDF generation
      const clone = reportRef.current.cloneNode(true) as HTMLElement;
      const isDarkMode = theme === 'dark';

      // Remove feedback sections before PDF generation
      const feedbackSections = clone.querySelectorAll('section');
      feedbackSections.forEach(section => {
        if (section.textContent?.includes('Feedback') || section.innerHTML.includes('feedback')) {
          section.remove();
        }
      });
      
      // Apply appropriate theme classes
      if (!isDarkMode) {
        clone.classList.remove('dark');
        clone.querySelectorAll('[class*="dark:"]').forEach(element => {
          element.classList.forEach(className => {
            if (className.startsWith('dark:')) {
              element.classList.remove(className);
            }
          });
        });
      }
      
      document.body.appendChild(clone);

      // Generate PDF from the clone with theme information
      await downloadPDF(clone, 'property-valuation-report.pdf', isDarkMode);

      // Remove the clone
      document.body.removeChild(clone);
    } catch (err) {
      setError('Failed to generate PDF. Please try again.');
      console.error('PDF generation error:', err);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="space-y-6">
      <div ref={reportRef} className="bg-white dark:bg-gray-800 p-8 rounded-lg shadow-sm max-w-3xl mx-auto print:max-w-[800px] print:mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Property Valuation Report</h1>
          <p className="text-gray-500 dark:text-gray-400 mt-2">Generated by EstateIQ</p>
        </div>

        <div className="space-y-6">
          <section>
            <h2 className="text-xl font-semibold mb-4 pb-2 border-b border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white">
              Property Details
            </h2>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-gray-600 dark:text-gray-400">Gross Area</p>
                <p className="font-medium text-gray-900 dark:text-white">{formData.GROSS_AREA} sq ft</p>
              </div>
              <div>
                <p className="text-gray-600 dark:text-gray-400">Living Area</p>
                <p className="font-medium text-gray-900 dark:text-white">{formData.LIVING_AREA} sq ft</p>
              </div>
              <div>
                <p className="text-gray-600 dark:text-gray-400">Land Area</p>
                <p className="font-medium text-gray-900 dark:text-white">{formData.LAND_SF} sq ft</p>
              </div>
              <div>
                <p className="text-gray-600 dark:text-gray-400">Year Built</p>
                <p className="font-medium text-gray-900 dark:text-white">{formData.YR_BUILT}</p>
              </div>
            </div>
          </section>

          <section>
            <h2 className="text-xl font-semibold mb-4 pb-2 border-b border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white">
              Room Information
            </h2>
            <div className="grid grid-cols-3 gap-4">
              <div>
                <p className="text-gray-600 dark:text-gray-400">Bedrooms</p>
                <p className="font-medium text-gray-900 dark:text-white">{formData.BED_RMS}</p>
              </div>
              <div>
                <p className="text-gray-600 dark:text-gray-400">Full Bathrooms</p>
                <p className="font-medium text-gray-900 dark:text-white">{formData.FULL_BTH}</p>
              </div>
              <div>
                <p className="text-gray-600 dark:text-gray-400">Half Bathrooms</p>
                <p className="font-medium text-gray-900 dark:text-white">{formData.HLF_BTH}</p>
              </div>
            </div>
          </section>

          {/* Map Section */}
          {formData.ZIP_CODE && (
            <section>
              <h2 className="text-xl font-semibold mb-4 pb-2 border-b border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white">
                Property Location (Approximate)
              </h2>
              <div className="max-w-2xl mx-auto h-[300px] rounded-lg overflow-hidden mb-4">
                <PropertyMap zipCode={formData.ZIP_CODE} />
              </div>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Map centered on the provided Zip Code.</p>
            </section>
          )}

          <section className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg">
            <h2 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">Estimated Property Value</h2>
            <p className="text-3xl font-bold text-blue-600 dark:text-blue-400">
              ${new Intl.NumberFormat('en-US').format(Math.round(prediction.prediction))}
            </p>
          </section>

          {/* Feedback Section */}
          {!feedbackSubmitted ? (
            <section className="bg-gray-50 dark:bg-gray-800/50 p-6 rounded-lg">
              <h2 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">Property Value Feedback</h2>
              <div className="space-y-4">
                <div>
                  <label htmlFor="actualPrice" className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                    What is the actual value of this property? (if known)
                  </label>
                  <div className="mt-1 relative rounded-md shadow-sm">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                      <span className="text-gray-500 dark:text-gray-400 sm:text-sm">$</span>
                    </div>
                    <input
                      type="number"
                      id="actualPrice"
                      value={feedback.actualPrice || ''}
                      onChange={(e) => setFeedback({...feedback, actualPrice: Number(e.target.value)})}
                      className="block w-full pl-7 pr-12 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:ring-blue-500 focus:border-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                      placeholder="0.00"
                    />
                  </div>
                </div>
                
                <div>
                  <label htmlFor="confidence" className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                    How confident are you in this value? (1-10)
                  </label>
                  <input
                    type="range"
                    id="confidence"
                    min="1"
                    max="10"
                    value={feedback.confidence}
                    onChange={(e) => setFeedback({...feedback, confidence: Number(e.target.value)})}
                    className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer"
                  />
                  <div className="flex justify-between text-xs text-gray-500">
                    <span>Not confident</span>
                    <span>Very confident</span>
                  </div>
                </div>

                <div>
                  <label htmlFor="notes" className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                    Additional Notes (optional)
                  </label>
                  <textarea
                    id="notes"
                    rows={3}
                    value={feedback.notes}
                    onChange={(e) => setFeedback({...feedback, notes: e.target.value})}
                    className="mt-1 block w-full border border-gray-300 dark:border-gray-600 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    placeholder="Any additional information about the property value..."
                  />
                </div>

                {submitError && (
                  <div className="text-red-600 dark:text-red-400 text-sm">
                    {submitError}
                  </div>
                )}

                <button
                  onClick={handleFeedbackSubmit}
                  disabled={isSubmitting}
                  className="w-full bg-green-500 text-white py-2 px-4 rounded-lg hover:bg-green-600 transition-colors disabled:bg-green-300 dark:disabled:bg-green-700"
                >
                  {isSubmitting ? "Submitting..." : "Submit Feedback"}
                </button>
              </div>
            </section>
          ) : (
            <section className="bg-green-50 dark:bg-green-900/20 p-6 rounded-lg">
              <div className="text-center text-green-700 dark:text-green-400">
                <svg className="mx-auto h-12 w-12 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                <h3 className="text-lg font-medium">Thank you for your feedback!</h3>
                <p className="mt-2 text-sm">Your input helps improve our valuation model.</p>
              </div>
            </section>
          )}

          <footer className="text-sm text-gray-500 dark:text-gray-400 mt-8 pt-4 border-t border-gray-200 dark:border-gray-700">
            <p>Report generated on {new Date().toLocaleDateString()}</p>
            <p>This valuation is an estimate based on the provided information and market data.</p>
          </footer>
        </div>
      </div>

      <div className="space-y-2">
        {error && (
          <div className="text-red-600 dark:text-red-400 text-sm text-center bg-red-50 dark:bg-red-900/20 p-2 rounded">
            {error}
          </div>
        )}
        <button
          onClick={handleDownload}
          disabled={isGenerating}
          className="w-full bg-blue-500 text-white py-2 px-4 rounded-lg hover:bg-blue-600 transition-colors disabled:bg-blue-300 dark:disabled:bg-blue-700"
        >
          {isGenerating ? (
            <span className="flex items-center justify-center">
              <svg className="animate-spin -ml-1 mr-3 h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Generating PDF...
            </span>
          ) : (
            "Download PDF Report"
          )}
        </button>
      </div>
    </div>
  );
};

export default ValuationReport;
