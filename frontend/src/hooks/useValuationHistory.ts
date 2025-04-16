import { useState, useEffect } from 'react';

interface ValuationRecord {
  id: string;
  date: string;
  formData: {
    GROSS_AREA: string;
    LIVING_AREA: string;
    LAND_SF: string;
    YR_BUILT: string;
    BED_RMS: string;
    FULL_BTH: string;
    HLF_BTH: string;
  };
  prediction: {
    prediction: number;
  };
}

export function useValuationHistory() {
  const [history, setHistory] = useState<ValuationRecord[]>([]);

  useEffect(() => {
    const stored = localStorage.getItem('valuation_history');
    if (stored) {
      setHistory(JSON.parse(stored));
    }
  }, []);

  const addValuation = (formData: ValuationRecord['formData'], prediction: ValuationRecord['prediction']) => {
    const newValuation: ValuationRecord = {
      id: Date.now().toString(),
      date: new Date().toISOString(),
      formData,
      prediction
    };

    const updatedHistory = [newValuation, ...history].slice(0, 50); // Keep last 50 records
    setHistory(updatedHistory);
    localStorage.setItem('valuation_history', JSON.stringify(updatedHistory));
    return newValuation;
  };

  const clearHistory = () => {
    setHistory([]);
    localStorage.removeItem('valuation_history');
  };

  return {
    history,
    addValuation,
    clearHistory
  };
}
