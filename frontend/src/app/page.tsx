"use client";

import { useState } from "react";

export default function Home() {
  const [formData, setFormData] = useState({
    GROSS_AREA: "",
    LIVING_AREA: "",
    LAND_SF: "",
    YR_BUILT: "",
    BED_RMS: "",
    FULL_BTH: "",
    HLF_BTH: "",
  });

  const [prediction, setPrediction] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setPrediction(null);
    setError("");

    try {
      const res = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          GROSS_AREA: parseFloat(formData.GROSS_AREA),
          LIVING_AREA: parseFloat(formData.LIVING_AREA),
          LAND_SF: parseFloat(formData.LAND_SF),
          YR_BUILT: parseInt(formData.YR_BUILT),
          BED_RMS: parseInt(formData.BED_RMS),
          FULL_BTH: parseInt(formData.FULL_BTH),
          HLF_BTH: parseInt(formData.HLF_BTH),
        }),
      });

      if (!res.ok) {
        throw new Error(`Failed to fetch prediction: ${res.status}`);
      }

      const data = await res.json();
      setPrediction(data);
    } catch (err: any) {
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main
      className="min-h-screen flex flex-col items-center justify-center p-6 text-white"
      style={{
        backgroundImage: "url('/images/unslash.jpg')",  // Ensure the image is in the public folder
        backgroundSize: "cover",
        backgroundPosition: "center",
      }}
    >
      {/* EstateIQ Heading */}
      <h1 className="absolute top-4 left-4 text-6xl font-bold text-blue-500 shadow-lg">
        EstateIQ
      </h1>

      {/* Form Section */}
      <div className="flex flex-col items-center justify-center bg-white p-6 rounded-xl shadow-md bg-opacity-70 w-full max-w-md">
        <h2 className="text-2xl font-semibold mb-6 text-gray-900">Find Estimates</h2>

        <form
          onSubmit={handleSubmit}
          className="flex flex-col gap-4 w-full"
        >
          <input
            type="number"
            name="GROSS_AREA"
            placeholder="Gross Area (sqft)"
            value={formData.GROSS_AREA}
            onChange={handleChange}
            className="border p-2 rounded text-gray-700 placeholder-gray-400"
            required
          />
          
          <input
            type="number"
            name="LIVING_AREA"
            placeholder="Living Area (sqft)"
            value={formData.LIVING_AREA}
            onChange={handleChange}
            className="border p-2 rounded text-gray-700 placeholder-gray-400"
            required
          />

          <input
            type="number"
            name="LAND_SF"
            placeholder="Land Area (sqft)"
            value={formData.LAND_SF}
            onChange={handleChange}
            className="border p-2 rounded text-gray-700 placeholder-gray-400"
            required
          />

          <input
            type="number"
            name="YR_BUILT"
            placeholder="Year Built"
            value={formData.YR_BUILT}
            onChange={handleChange}
            className="border p-2 rounded text-gray-700 placeholder-gray-400"
            required
          />

          <input
            type="number"
            name="BED_RMS"
            placeholder="Number of Bedrooms"
            value={formData.BED_RMS}
            onChange={handleChange}
            className="border p-2 rounded text-gray-700 placeholder-gray-400"
            required
          />

          <input
            type="number"
            name="FULL_BTH"
            placeholder="Number of Full Bathrooms"
            value={formData.FULL_BTH}
            onChange={handleChange}
            className="border p-2 rounded text-gray-700 placeholder-gray-400"
            required
          />

          <input
            type="number"
            name="HLF_BTH"
            placeholder="Number of Half Bathrooms"
            value={formData.HLF_BTH}
            onChange={handleChange}
            className="border p-2 rounded text-gray-700 placeholder-gray-400"
            required
          />

          <button
            type="submit"
            className="bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 mt-4"
          >
            {loading ? "Predicting..." : "Get Prediction"}
          </button>
        </form>
      </div>

      {/* Error message */}
      {error && <p className="mt-4 text-red-600">{error}</p>}

      {/* Prediction result */}
      {prediction && (
        <div className="mt-6 bg-white p-4 rounded shadow-md max-w-md w-full">
          <h2 className="text-lg font-semibold mb-2">Estimated Property Price:</h2>
          <p className="text-2xl font-bold text-green-700">
            ₹ {prediction.prediction.toFixed(2)}
          </p>
        </div>
      )}

       
    </main>
  );
}
