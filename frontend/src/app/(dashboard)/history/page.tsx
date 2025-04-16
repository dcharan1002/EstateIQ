"use client";

import { useValuationHistory } from "@/hooks/useValuationHistory";

export default function HistoryPage() {
  const { history, clearHistory } = useValuationHistory();
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Valuation History</h1>
      </div>

      {/* Content */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-8">
        <div className="text-center mb-8">
          <div className="text-4xl mb-4">📅</div>
          <h2 className="text-xl font-semibold text-gray-700 dark:text-gray-200 mb-2">Recent Valuations</h2>
          <p className="text-gray-500 dark:text-gray-400">Track your property valuation history and market trends</p>
        </div>

        <div className="border dark:border-gray-700 rounded-lg overflow-hidden">
          <div className="bg-gray-50 dark:bg-gray-700/50 px-6 py-3 border-b dark:border-gray-700">
            <div className="grid grid-cols-4 gap-4 font-medium text-gray-500 dark:text-gray-400">
              <div>Property</div>
              <div>Date</div>
              <div>Valuation</div>
              <div>Change</div>
            </div>
          </div>
          {history.length === 0 ? (
            <div className="p-8 text-center">
              <p className="text-gray-400 dark:text-gray-500">No valuation history available</p>
              <button
                className="mt-4 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
                onClick={() => window.location.href = '/valuation'}
              >
                Get Your First Valuation
              </button>
            </div>
          ) : (
            <>
              {history.map((record) => (
                <div key={record.id} className="px-6 py-4 border-b dark:border-gray-700 last:border-b-0">
                  <div className="grid grid-cols-4 gap-4">
                    <div className="text-gray-900 dark:text-white">
                      {record.formData.BED_RMS}bd {record.formData.FULL_BTH}ba
                      <span className="block text-sm text-gray-500 dark:text-gray-400">{record.formData.GROSS_AREA} sqft</span>
                    </div>
                    <div className="text-gray-600 dark:text-gray-400">
                      {new Date(record.date).toLocaleDateString()}
                    </div>
                    <div className="font-medium text-gray-900 dark:text-white">
                      ${new Intl.NumberFormat('en-US').format(Math.round(record.prediction.prediction))}
                    </div>
                    <div className="text-gray-600 dark:text-gray-400">
                      -
                    </div>
                  </div>
                </div>
              ))}
              <div className="px-6 py-4 bg-gray-50 dark:bg-gray-700/50">
                <button
                  onClick={clearHistory}
                  className="text-sm text-gray-500 dark:text-gray-400 hover:text-red-500 dark:hover:text-red-400 transition-colors"
                >
                  Clear History
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
