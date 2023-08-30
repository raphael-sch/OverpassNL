import { useEffect, useRef, useState } from "react";
import mapgl from "maplibre-gl";

function useMapboxMap(options: any) {
  const [map, setMap] = useState<any | null>(null);
  const [maploaded, setMaploaded] = useState(false);
  const mapRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const opts = {
      container: mapRef.current,
      preserveDrawingBuffer: true,
      style: {
        version: 8,

        sources: {
          osm: {
            type: "raster",
            tiles: [
              "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png",
              // 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            ],
            tileSize: 256,
        
            maxzoom: 18,
          },
        },
        layers: [
          {
            id: "osm",
            type: "raster",
            source: "osm",
            layout: {
              visibility: "visible",
            },
          },
        ],
      },
      ...options,
    };

    const mapInstance = new mapgl.Map(opts);
    // mapInstance.addControl(
    //   new mapgl.NavigationControl(),
    //   'top-left'
    // );

    mapInstance.on("load", () => {
      setMaploaded(true);
    });

    setMap(mapInstance);
  }, [options]);

  return { map, mapRef, maploaded };
}

export default useMapboxMap;
