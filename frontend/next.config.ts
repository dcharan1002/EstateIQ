import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/predict',
        destination: 'https://estateiq-prediction-5h3cnrngkq-uc.a.run.app/predict'
      }
    ];
  }
};

export default nextConfig;
