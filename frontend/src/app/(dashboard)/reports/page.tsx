"use client";

import { useValuationHistory } from "@/hooks/useValuationHistory";
import { useState } from "react";
import { useTheme } from "next-themes";

export default function ReportsPage() {
  const { history } = useValuationHistory();
  const [selectedReport, setSelectedReport] = useState<string | null>(null);
  const { theme } = useTheme();

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Property Reports</h1>
      </div>

      {/* Reports Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {history.length === 0 ? (
          <div className="md:col-span-2 lg:col-span-3">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-8 text-center">
              <div className="text-4xl mb-4">📊</div>
              <h2 className="text-xl font-semibold text-gray-700 dark:text-gray-200 mb-2">No Reports Available</h2>
              <p className="text-gray-500 dark:text-gray-400 mb-4">Start by generating your first property valuation report</p>
              <button
                className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
                onClick={() => window.location.href = '/valuation'}
              >
                Generate New Report
              </button>
            </div>
          </div>
        ) : (
          history.map((record) => (
            <div key={record.id} className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
              <div className="flex justify-between items-start mb-4">
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                    {record.formData.BED_RMS}bd {record.formData.FULL_BTH}ba Property
                  </h3>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    {new Date(record.date).toLocaleDateString()}
                  </p>
                </div>
                <div className="text-2xl">🏠</div>
              </div>

              <div className="space-y-3 mb-6">
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div className="text-gray-500 dark:text-gray-400">Area:</div>
                  <div className="text-gray-900 dark:text-white">{record.formData.GROSS_AREA} sqft</div>
                  <div className="text-gray-500 dark:text-gray-400">Year Built:</div>
                  <div className="text-gray-900 dark:text-white">{record.formData.YR_BUILT}</div>
                </div>
              </div>

              <div className="border-t dark:border-gray-700 pt-4">
                <div className="text-center mb-4">
                  <div className="text-sm text-gray-500 dark:text-gray-400">Estimated Value</div>
                  <div className="text-xl font-bold text-blue-500 dark:text-blue-400">
                    ${new Intl.NumberFormat('en-US').format(Math.round(record.prediction.prediction))}
                  </div>
                </div>

                {selectedReport === record.id ? (
                  <button
                    onClick={() => window.print()}
                    className="w-full px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors text-sm"
                  >
                    Print Report
                  </button>
                ) : (
                  <button
                    onClick={() => setSelectedReport(record.id)}
                    className="w-full px-4 py-2 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors text-sm"
                  >
                    View Report
                  </button>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
