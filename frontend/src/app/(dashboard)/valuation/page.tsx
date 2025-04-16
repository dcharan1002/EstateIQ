"use client";

import { useState, ChangeEvent, FormEvent } from "react";
import ValuationReport from "@/components/ValuationReport";
import { useValuationHistory } from "@/hooks/useValuationHistory";

// Define types for form data
export interface FormData {
  GROSS_AREA: string;
  LIVING_AREA: string;
  LAND_SF: string;
  YR_BUILT: string;
  YR_REMODEL: string;
  BED_RMS: string;
  FULL_BTH: string;
  HLF_BTH: string;
  NUM_PARKING: string;
  FIREPLACES: string;
  KITCHENS: string;
  TT_RMS: string;
  ZIP_CODE: string;
  structureClass: string;
  intCondition: string;
  overallCondition: string;
  kitchenStyle: string;
  kitchenTypeFullEatIn: boolean;
  acType: string;
  heatType: string;
  propView: string;
  cornerUnit: boolean;
  orientation: string;
  extCondition: string;
  roofCover: string;
}

// Define options for dropdowns (outside the component)
const structureClassOptions = ["", "Brick/Concrete", "Wood/Frame", "Reinforced Concrete"];
const conditionOptions = ["", "Excellent", "Very Good", "Good", "Average", "Fair", "Poor"];
const kitchenStyleOptions = ["", "Modern", "Luxury", "Semi-Modern"];
const acTypeOptions = ["", "None", "Central AC", "Ductless AC"];
const heatTypeOptions = ["", "None", "Forced Hot Air", "Hot Water/Steam"];
const propViewOptions = ["", "Excellent", "Good", "Average/Other"];
const orientationOptions = ["", "None", "End", "Front/Street"];
const extConditionOptions = ["", "Excellent", "Good", "Average/Other"];
const roofCoverOptions = ["", "None", "Slate", "Asphalt Shingle"];

export default function ValuationPage() {
  const { addValuation } = useValuationHistory();
  const [formData, setFormData] = useState<FormData>({
    GROSS_AREA: "",
    LIVING_AREA: "",
    LAND_SF: "",
    YR_BUILT: "",
    YR_REMODEL: "",
    BED_RMS: "",
    FULL_BTH: "",
    HLF_BTH: "",
    NUM_PARKING: "",
    FIREPLACES: "",
    KITCHENS: "",
    TT_RMS: "",
    ZIP_CODE: "",
    structureClass: "",
    intCondition: "",
    overallCondition: "",
    kitchenStyle: "",
    kitchenTypeFullEatIn: false,
    acType: "",
    heatType: "",
    propView: "",
    cornerUnit: false,
    orientation: "",
    extCondition: "",
    roofCover: "",
  });

  const [prediction, setPrediction] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Generic change handler for inputs, selects, and checkboxes
  const handleChange = (e: ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    const checked = (e.target as HTMLInputElement).checked; // Specific check for checkbox type

    setFormData((prevData) => ({
      ...prevData,
      [name]: type === "checkbox" ? checked : value,
    }));
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setPrediction(null);
    setError("");

    // Construct the payload based on backend expectations
    const payload: { [key: string]: number | string } = {
      GROSS_AREA: parseFloat(formData.GROSS_AREA) || 0,
      LIVING_AREA: parseFloat(formData.LIVING_AREA) || 0,
      LAND_SF: parseFloat(formData.LAND_SF) || 0,
      YR_BUILT: parseInt(formData.YR_BUILT) || 0,
      YR_REMODEL: parseInt(formData.YR_REMODEL) || parseInt(formData.YR_BUILT) || 0, // Default remodel year to build year if empty
      BED_RMS: parseInt(formData.BED_RMS) || 0,
      FULL_BTH: parseInt(formData.FULL_BTH) || 0,
      HLF_BTH: parseInt(formData.HLF_BTH) || 0,
      NUM_PARKING: parseInt(formData.NUM_PARKING) || 0,
      FIREPLACES: parseInt(formData.FIREPLACES) || 0,
      KITCHENS: parseInt(formData.KITCHENS) || 0,
      TT_RMS: parseInt(formData.TT_RMS) || 0,
      ZIP_CODE: formData.ZIP_CODE || "00000", // Ensure ZIP code is a string

      // One-hot encode categorical features
      // Structure Class
      "STRUCTURE_CLASS_C - BRICK/CONCR": formData.structureClass === "Brick/Concrete" ? 1 : 0,
      "STRUCTURE_CLASS_D - WOOD/FRAME": formData.structureClass === "Wood/Frame" ? 1 : 0,
      "STRUCTURE_CLASS_B - REINF CONCR": formData.structureClass === "Reinforced Concrete" ? 1 : 0,

      // Interior Condition
      "INT_COND_E - EXCELLENT": formData.intCondition === "Excellent" ? 1 : 0,
      "INT_COND_G - GOOD": formData.intCondition === "Good" ? 1 : 0,
      "INT_COND_A - AVERAGE": formData.intCondition === "Average" ? 1 : 0,
      "INT_COND_F - FAIR": formData.intCondition === "Fair" ? 1 : 0,
      "INT_COND_P - POOR": formData.intCondition === "Poor" ? 1 : 0,

      // Overall Condition
      "OVERALL_COND_E - EXCELLENT": formData.overallCondition === "Excellent" ? 1 : 0,
      "OVERALL_COND_VG - VERY GOOD": formData.overallCondition === "Very Good" ? 1 : 0,
      "OVERALL_COND_G - GOOD": formData.overallCondition === "Good" ? 1 : 0,
      "OVERALL_COND_A - AVERAGE": formData.overallCondition === "Average" ? 1 : 0,
      "OVERALL_COND_F - FAIR": formData.overallCondition === "Fair" ? 1 : 0,
      "OVERALL_COND_P - POOR": formData.overallCondition === "Poor" ? 1 : 0,

      // Kitchen Style
      "KITCHEN_STYLE2_M - MODERN": formData.kitchenStyle === "Modern" ? 1 : 0,
      "KITCHEN_STYLE2_L - LUXURY": formData.kitchenStyle === "Luxury" ? 1 : 0,
      "KITCHEN_STYLE2_S - SEMI-MODERN": formData.kitchenStyle === "Semi-Modern" ? 1 : 0,

      // Kitchen Type (Checkbox)
      "KITCHEN_TYPE_F - FULL EAT IN": formData.kitchenTypeFullEatIn ? 1 : 0,

      // AC Type
      "AC_TYPE_C - CENTRAL AC": formData.acType === "Central AC" ? 1 : 0,
      "AC_TYPE_D - DUCTLESS AC": formData.acType === "Ductless AC" ? 1 : 0,

      // Heat Type
      "HEAT_TYPE_F - FORCED HOT AIR": formData.heatType === "Forced Hot Air" ? 1 : 0,
      "HEAT_TYPE_W - HT WATER/STEAM": formData.heatType === "Hot Water/Steam" ? 1 : 0,

      // Property View
      "PROP_VIEW_E - EXCELLENT": formData.propView === "Excellent" ? 1 : 0,
      "PROP_VIEW_G - GOOD": formData.propView === "Good" ? 1 : 0,
      // Assuming 'Average/Other' maps to 0 for both E and G

      // Corner Unit (Checkbox)
      "CORNER_UNIT_Y - YES": formData.cornerUnit ? 1 : 0,

      // Orientation
      "ORIENTATION_E - END": formData.orientation === "End" ? 1 : 0,
      "ORIENTATION_F - FRONT/STREET": formData.orientation === "Front/Street" ? 1 : 0,

      // Exterior Condition
      "EXT_COND_E - EXCELLENT": formData.extCondition === "Excellent" ? 1 : 0,
      "EXT_COND_G - GOOD": formData.extCondition === "Good" ? 1 : 0,

      // Roof Cover
      "ROOF_COVER_S - SLATE": formData.roofCover === "Slate" ? 1 : 0,
      "ROOF_COVER_A - ASPHALT SHINGL": formData.roofCover === "Asphalt Shingle" ? 1 : 0,
    };

    const apiUrl = process.env.NEXT_PUBLIC_API_URL;
    if (!apiUrl) {
        setError("API URL is not configured. Please check environment variables.");
        setLoading(false);
        return;
    }

    try {
      const res = await fetch(`${apiUrl}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errorBody = await res.text();
        console.error("API Error Response:", errorBody);
        throw new Error(`Failed to fetch prediction: ${res.status} - ${errorBody}`);
      }

      const data = await res.json();
      setPrediction(data);
      // Pass the full FormData object, not just the initial subset
      addValuation(formData, data);
    } catch (err: any) {
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-4">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Property Valuation</h1>
        <p className="mt-2 text-gray-600 dark:text-gray-400">Fill in the property details to get an instant valuation.</p>
      </div>

      {/* Form Card */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6 mb-8">
        <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Numeric Inputs */}
          {[
            { name: "GROSS_AREA", label: "Gross Area (sq ft)", type: "number", required: true },
            { name: "LIVING_AREA", label: "Living Area (sq ft)", type: "number", required: true },
            { name: "LAND_SF", label: "Land Area (sq ft)", type: "number", required: true },
            { name: "YR_BUILT", label: "Year Built", type: "number", required: true },
            { name: "YR_REMODEL", label: "Year Remodeled (optional)", type: "number" },
            { name: "BED_RMS", label: "Bedrooms", type: "number", required: true },
            { name: "FULL_BTH", label: "Full Bathrooms", type: "number", required: true },
            { name: "HLF_BTH", label: "Half Bathrooms", type: "number", required: true },
            { name: "NUM_PARKING", label: "Parking Spots", type: "number", required: true },
            { name: "FIREPLACES", label: "Fireplaces", type: "number", required: true },
            { name: "KITCHENS", label: "Kitchens", type: "number", required: true },
            { name: "TT_RMS", label: "Total Rooms", type: "number", required: true },
            { name: "ZIP_CODE", label: "Zip Code", type: "text", required: true, pattern: "\\d{5}" },
          ].map((field) => (
            <div key={field.name} className="space-y-1">
              <label htmlFor={field.name} className="block text-sm font-medium text-gray-700 dark:text-gray-300">{field.label}</label>
              <input
                type={field.type}
                id={field.name}
                name={field.name}
                value={formData[field.name as keyof Omit<FormData, 'kitchenTypeFullEatIn' | 'cornerUnit'>]} // Exclude boolean keys for value prop
                onChange={handleChange}
                className="w-full p-2 border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder={`Enter ${field.label.toLowerCase()}`}
                required={field.required}
                pattern={field.pattern}
                min={field.type === 'number' ? 0 : undefined} // Basic validation for numbers
              />
            </div>
          ))}

          {/* Dropdowns */}
          {[
            { name: "structureClass", label: "Structure Class", options: structureClassOptions, required: true },
            { name: "intCondition", label: "Interior Condition", options: conditionOptions, required: true },
            { name: "overallCondition", label: "Overall Condition", options: conditionOptions, required: true },
            { name: "kitchenStyle", label: "Kitchen Style", options: kitchenStyleOptions, required: true },
            { name: "acType", label: "AC Type", options: acTypeOptions, required: true },
            { name: "heatType", label: "Heat Type", options: heatTypeOptions, required: true },
            { name: "propView", label: "Property View", options: propViewOptions, required: true },
            { name: "orientation", label: "Orientation", options: orientationOptions, required: true },
            { name: "extCondition", label: "Exterior Condition", options: extConditionOptions, required: true },
            { name: "roofCover", label: "Roof Cover", options: roofCoverOptions, required: true },
          ].map((field) => (
            <div key={field.name} className="space-y-1">
              <label htmlFor={field.name} className="block text-sm font-medium text-gray-700 dark:text-gray-300">{field.label}</label>
              <select
                id={field.name}
                name={field.name}
                value={formData[field.name as keyof Omit<FormData, 'kitchenTypeFullEatIn' | 'cornerUnit'>] as string} // Exclude boolean fields and ensure string type
                onChange={handleChange}
                className="w-full p-2 border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                required={field.required}
              >
                {field.options.map((option: string) => ( // Add type annotation for option
                  <option key={option} value={option}>{option || `Select ${field.label}`}</option>
                ))}
              </select>
            </div>
          ))}

          {/* Checkboxes */}
          <div className="flex items-center space-x-2 md:col-span-1 pt-5"> {/* Added pt-5 for alignment */}
             <input
                type="checkbox"
                id="kitchenTypeFullEatIn"
                name="kitchenTypeFullEatIn"
                checked={formData.kitchenTypeFullEatIn}
                onChange={handleChange}
                className="h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
              />
            <label htmlFor="kitchenTypeFullEatIn" className="text-sm font-medium text-gray-700 dark:text-gray-300">Kitchen: Full Eat-in</label>
          </div>
           <div className="flex items-center space-x-2 md:col-span-1 pt-5"> {/* Added pt-5 for alignment */}
             <input
                type="checkbox"
                id="cornerUnit"
                name="cornerUnit"
                checked={formData.cornerUnit}
                onChange={handleChange}
                className="h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
              />
            <label htmlFor="cornerUnit" className="text-sm font-medium text-gray-700 dark:text-gray-300">Corner Unit</label>
          </div>


          {/* Submit Button */}
          <div className="md:col-span-3">
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-blue-500 text-white py-2 px-4 rounded-lg hover:bg-blue-600 transition-colors disabled:bg-blue-300 dark:disabled:bg-blue-700"
            >
              {loading ? "Calculating..." : "Get Valuation"}
            </button>
          </div>
        </form>
      </div>

      {/* Error Message */}
      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-700 text-red-600 dark:text-red-400 rounded-lg p-4 mb-8">
          {error}
        </div>
      )}

      {/* Results Card */}
      {prediction && (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
          <div className="text-center mb-6">
            <h2 className="text-2xl font-semibold text-gray-900 dark:text-white">Estimated Property Value</h2>
            <p className="text-4xl font-bold text-blue-500 dark:text-blue-400 mt-2">
              ${new Intl.NumberFormat('en-US').format(Math.round(prediction.prediction))}
            </p>
          </div>

          {/* Pass full formData to ValuationReport */}
          <div className="flex justify-center">
            <ValuationReport formData={formData} prediction={prediction} />
          </div>
        </div>
      )}
    </div>
  );
}
