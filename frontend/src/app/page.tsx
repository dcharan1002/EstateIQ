"use client";

import Link from "next/link";

export default function LandingPage() {
  return (
    <main className="min-h-screen flex flex-col items-center justify-start p-8 text-white relative">
      <div
        className="absolute inset-0 z-0"
        style={{
          backgroundImage: "url('/images/unslash.jpg')",
          backgroundSize: "cover",
          backgroundPosition: "center",
          backgroundAttachment: "fixed",
        }}
      />
      <div className="absolute inset-0 bg-black bg-opacity-50 backdrop-blur-sm z-1" />
      
      {/* Content */}
      <div className="relative z-10 w-full max-w-6xl mx-auto">
        {/* Header */}
        <div className="flex justify-between items-center mb-12">
          <div>
            <h1 className="text-6xl font-black">
              Estate<span className="text-blue-400">IQ</span>
            </h1>
            <p className="mt-2 text-xl text-gray-300">Intelligent Real Estate Valuation</p>
          </div>
          <div className="flex gap-4">
            <Link 
              href="/sign-in"
              className="px-6 py-2 bg-white/10 hover:bg-white/20 rounded-lg transition-all"
            >
              Sign In
            </Link>
            <Link
              href="/sign-up"
              className="px-6 py-2 bg-blue-500 hover:bg-blue-600 rounded-lg transition-all"
            >
              Get Started
            </Link>
          </div>
        </div>

        {/* Hero Section */}
        <div className="mt-20 text-center">
          <h2 className="text-6xl font-bold mb-8 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
            Transform Property Valuation<br />with AI Intelligence
          </h2>
          <p className="text-2xl text-gray-300 mb-16 max-w-3xl mx-auto leading-relaxed">
            Experience the future of real estate valuation with our advanced AI technology.
          </p>
          <Link
            href="/sign-up"
            className="inline-flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-blue-500 to-blue-600 rounded-lg text-lg font-semibold transition-all transform hover:scale-105 hover:shadow-xl"
          >
            Get Started
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="w-5 h-5">
              <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
            </svg>
          </Link>
        </div>

        {/* Features */}
        <div className="mt-32 grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="bg-white/10 backdrop-blur-md p-8 rounded-xl transform hover:scale-105 transition-all">
            <div className="text-blue-400 text-4xl mb-4">🎯</div>
            <h3 className="text-xl font-semibold mb-3">Precision AI</h3>
            <p className="text-gray-300">Advanced machine learning algorithms for accurate property valuations.</p>
          </div>
          <div className="bg-white/10 backdrop-blur-md p-8 rounded-xl transform hover:scale-105 transition-all">
            <div className="text-blue-400 text-4xl mb-4">⚡️</div>
            <h3 className="text-xl font-semibold mb-3">Instant Results</h3>
            <p className="text-gray-300">Get comprehensive property valuations in seconds.</p>
          </div>
          <div className="bg-white/10 backdrop-blur-md p-8 rounded-xl transform hover:scale-105 transition-all">
            <div className="text-blue-400 text-4xl mb-4">📊</div>
            <h3 className="text-xl font-semibold mb-3">Detailed Analysis</h3>
            <p className="text-gray-300">In-depth reports with key property metrics and insights.</p>
          </div>
        </div>
      </div>
    </main>
  );
}
