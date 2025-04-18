"use client";

import { useEffect, useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import type { LatLngTuple } from 'leaflet';

// Fix for default marker icons in Leaflet with Next.js
const icon = L.icon({
  iconUrl: "https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png",
  shadowUrl: "https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png",
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
});

interface PropertyMapProps {
  zipCode: string;
  initialLat?: number;
  initialLng?: number;
  onLocationChange?: (lat: number, lng: number) => void;
}

async function getCoordinatesFromZipCode(zipCode: string): Promise<LatLngTuple | null> {
  try {
    const response = await fetch(`https://api.zippopotam.us/us/${zipCode}`);
    const data = await response.json();
    if (data && data.places && data.places[0]) {
      return [
        parseFloat(data.places[0].latitude),
        parseFloat(data.places[0].longitude)
      ];
    }
    return null;
  } catch (error) {
    console.error('Error fetching coordinates:', error);
    return null;
  }
}

export default function PropertyMap({ zipCode, initialLat, initialLng, onLocationChange }: PropertyMapProps) {
  // Boston coordinates as default
  const defaultPosition: LatLngTuple = [42.3601, -71.0589];
  const [position, setPosition] = useState<LatLngTuple>(
    initialLat && initialLng ? [initialLat, initialLng] : defaultPosition
  );

  useEffect(() => {
    if (zipCode) {
      getCoordinatesFromZipCode(zipCode).then(coords => {
        if (coords) {
          setPosition(coords);
          onLocationChange?.(coords[0], coords[1]);
        }
      });
    }
  }, [zipCode, onLocationChange]);

  // Using dynamic import for MapContainer to avoid SSR issues
  return (
    <div className="w-full h-full">
      <MapContainer
        key={`${position[0]}-${position[1]}`} // Force re-render when position changes
        center={position}
        zoom={13}
        scrollWheelZoom={false}
        style={{ height: "100%", width: "100%" }}
        className="z-0"
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        />
        <Marker position={position} icon={icon}>
          <Popup>
            Property Location<br />
            ZIP: {zipCode}
          </Popup>
        </Marker>
      </MapContainer>
    </div>
  );
}
