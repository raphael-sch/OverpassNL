"use client";

import { Input } from "@/components/ui/input";
import MapContainer from "@/components/Maplibre/MapContainer";
import bbox from '@turf/bbox';

import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";

import useMapboxMap from "@/components/Maplibre/useMaplibreMap";
import React, { useEffect, useMemo, useRef, useState } from "react";
import GeojsonLayer from "@/components/Maplibre/GeojsonLayer";
import Generating from "@/assets/animatingsvg/generating";
import { Textarea } from "@/components/ui/textarea";
import {
  addLineBreaks,
  getAllGeomTypes,
  getRandomDarkColor,
  getboundtext,
} from "@/lib/utils";
import { Button } from "@/components/ui/button";
import RunningSvg from "@/assets/animatingsvg/run";
import { Separator } from "@/components/ui/separator";
import ReorderComponent from "./LayersWithReorder";

import { Toaster, toast } from "sonner";
import { ChevronLeftSquare, ChevronRightSquare, Play } from "lucide-react";
import Popup from "@/components/Maplibre/popup";

var osmtogeojson = require("@/lib/osmtogeojson");

type querystates =
  | "idle"
  | "generating_query"
  | "extracting_from_osm"
  | "extraction_done";

type tabs = "manual" | "askgpt";

type queryresponse = {
  osmquery: string;
  query_name: string;
};

export type dynamicgeojson = {
  [key: string]: any;
};

export interface layer {
  name: string;
  geojson: dynamicgeojson;
  color: string;
  id: string;
  containingGeometries: {
    visibleOnMap: boolean;
    type: string;
  }[];
}

export type ContainingGeometries = layer["containingGeometries"];

const mapopts = {
  center: [-73.5674, 45.5019],
  zoom: 12,
  pitch: 0,
  maxzoom: 25,
};

let queryAbortController: AbortController;
let overpassAbortController: AbortController;

export default function Home() {
  ////const response = await fetch("https://aapi.tech-demos.de/overpassnl/overpassnl", {
  const { map, mapRef, maploaded } = useMapboxMap(mapopts);
  const [queryState, setQueryState] = useState<querystates>("idle");
  const inputRef = useRef<HTMLInputElement>(null);

  const [activeTab, setActiveTab] = useState<tabs>("askgpt");

  const [extractedQuery, setExtractedQuery] = useState<null | queryresponse>(
    null
  );

  const [layers, setLayers] = useState<layer[]>([]);

  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);

    useEffect(() => {
    //fetch("http://127.0.0.1:5000/status", {});
    fetch("https://aapi.tech-demos.de/overpassnl/overpassnl?status", {});
  }, []);

  const extractFeatures = async (userinput?: string) => {
    setQueryState("generating_query");
    if (queryAbortController) queryAbortController.abort();
    if (overpassAbortController) overpassAbortController.abort();
    queryAbortController = new AbortController();
    overpassAbortController = new AbortController();

    let respjson = {
      osmquery: "",
      query_name: "",
      refined: "",
    };
    try {
      setExtractedQuery(null);

      if (activeTab === "askgpt") {
        const response = await fetch("https://aapi.tech-demos.de/overpassnl/overpassnl", {
        //const response = await fetch("http://127.0.0.1:5000/query", {
          signal: queryAbortController.signal,
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ usertext: userinput }),
        });
        const err = new Error();

        if (response.status === 401) {
          err.name = "Invalid Api Key";
          err.cause = 401;
          throw err;
        }
        if (response.status === 429) {
          err.name = "Rate Limited";
          err.cause = 429;
          throw err;
        }

        const respJson = await response.json();
        setExtractedQuery({
          osmquery: addLineBreaks(respJson?.osmquery),
          query_name: respJson?.query_name,
        });
        respjson = {
          osmquery: addLineBreaks(respJson?.osmquery),
          query_name: respJson?.query_name,
          refined: respJson?.refined,
        };
      } else {
        setExtractedQuery({
          osmquery: extractedQuery?.osmquery || "",
          query_name: userinput || "",
        });
        respjson = {
          osmquery: extractedQuery?.osmquery || "",
          query_name: userinput || "",
          refined: "no",
        };
      }
      setQueryState("extracting_from_osm");

      var bounds = map.getBounds();

      var boundtext = getboundtext(bounds);

      let overpassResponse = await fetch(
        "https://overpass-api.de/api/interpreter?",
        {
          signal: overpassAbortController.signal,
          method: "POST",
          body: ` 
          ${respjson?.osmquery
            ?.replaceAll("{{bbox}}", boundtext)
            .replace("[out:xml]", "[out:json]")} 
        `,
        }
      );
      let overpassResponseClone = overpassResponse.clone();
      try{
        const overpassrespjson = await overpassResponse.json();
        const toGeojson = {...osmtogeojson(overpassrespjson)};

        const geomtypes: ContainingGeometries = getAllGeomTypes(toGeojson);

        if (toGeojson?.features.length === 0) {
          toast("Features not found");
        } else {
          setLayers([
            {
              geojson: toGeojson,
              name: respjson.query_name,
              color: getRandomDarkColor([...layers.map((lyr) => lyr.color)]),
              id: `${Date.now()}`,
              containingGeometries: geomtypes,
            },
            ...layers,
          ]);
          if (respjson?.osmquery.indexOf("bbox") === -1) {
            const boundingBox = bbox(toGeojson);
            map.fitBounds(boundingBox, {padding: 20, duration: 2000});
          }

        }
        setQueryState("extraction_done");
      } catch (e: any) {
        if (activeTab === "askgpt" && respjson?.refined === 'no') {
          const overpassrespErrorText = await overpassResponseClone.text();
          extractFeatures(userinput + '[QUERY:]' + respjson?.osmquery + '[NAME:]' + respjson?.query_name + '[ERROR:]' + overpassrespErrorText);
          return;
        }
        console.log(e, e?.message, "error");
        setQueryState("idle");

        if (e.name === "AbortError") return;

        if (e.name === "SyntaxError" && activeTab === "askgpt") {
          return toast(`Syntax error: ${e?.message || "2"}`, {
            action: {
              label: "Continute editing query",
              onClick: () => {
                inputRef.current!.value = respjson?.query_name;
                setActiveTab("manual");
              },
            },
          });
        }
        toast.error(e?.name + ": " + e?.message || "Something went wrong", {
          style: {
            background: "red",
            color: "white",
          },
        });
      }
    } catch (e: any) {
      console.log(e, e?.message, "error");
      setQueryState("idle");

      if (e.name === "AbortError") return;

      if (e.name === "SyntaxError" && activeTab === "askgpt") {
        return toast(`Syntax error: ${e?.message || "2"}`, {
          action: {
            label: "Continute editing query",
            onClick: () => {
              inputRef.current!.value = respjson?.query_name;
              setActiveTab("manual");
            },
          },
        });
      }
      toast.error(e?.name + ": " + e?.message || "Something went wrong", {
        style: {
          background: "red",
          color: "white",
        },
      });
    }
  };

  useEffect(() => {
    if (!map) return;
    JSON.parse(JSON.stringify(layers))
      .reverse()
      .forEach((lyr: layer) => {
        ["Point", "Line", "Polygon"].forEach((a) => {
          map.moveLayer(`${lyr.id}-${a}`);
        });
      });
  }, [layers, map]);

  return (
    <main className="flex min-h-screen flex-row ">
      <Toaster richColors position="top-right" />
      <div className="absolute z-30 p-2">
        <ReorderComponent layers={layers} setLayers={setLayers} />
      </div>
      <div id="map-container" className="w-full md:w-[75%] relative">
        <MapContainer mapRef={mapRef} style={{ width: "100%", height: "100%" }}>
          {layers.map((layer, i) => {
            return (
              <GeojsonLayer
                key={`${layer.id}`}
                id={layer.id}
                geojson={layer?.geojson}
                color={layer.color}
                map={map}
                maploaded={maploaded}
                containingGeometries={layer.containingGeometries}
              />
            );
          })}
          <Popup map={map} />
        </MapContainer>
      </div>

      {mobileSidebarOpen ? (
        <ChevronRightSquare
          onClick={() => {
            setMobileSidebarOpen(!mobileSidebarOpen);
          }}
          className={`absolute bg-secondary top-[50%] ${
            mobileSidebarOpen ? "right-[80%]" : "right-0"
          } md:right-0 md:hidden`}
          size={40}
        />
      ) : (
        <ChevronLeftSquare
          onClick={() => {
            setMobileSidebarOpen(!mobileSidebarOpen);
          }}
          className={`absolute  top-[50%] ${
            mobileSidebarOpen ? "right-[80%]" : "right-0"
          } md:right-0 md:hidden`}
          size={40}
        />
      )}

      <div
        id="querybar"
        className={`absolute right-0 h-full w-[80%] md:w-[30%] bg-secondary rounded-md p-4  flex-col items-center 
        ${mobileSidebarOpen ? "flex" : "hidden"} md:flex
        `}
      >
        <div className="flex-1 flex flex-col items-center justify-start ">
          <h2 className="text-4xl font-bold text-center"> OverpassNL</h2>

          <p className="text-center">
            {/* eslint-disable-next-line */}
            A natural language interface to OpenStreetMap.
          </p>

          <p>
          <a href="https://github.com/raphael-sch/OverpassNL" target="_blank">--click here to learn more--</a>
          </p>
          <p>Disclaimer: Your inputs will be processed by the OpenAI API</p>
        </div>
        {queryState === "generating_query" && (
          <div className="w-full  my-2 flex flex-col items-center justify-center">
            <p>Generating query . . .</p>
            <Generating />
          </div>
        )}

        {(activeTab === "manual" || !["idle"].includes(queryState)) && (
          <div className="w-full my-2 flex flex-col items-center justify-center">
            <div className="w-full my-4 flex items-center justify-center">
              <Textarea
                placeholder="osm query"
                value={extractedQuery?.osmquery || ""}
                onChange={(e) =>
                  setExtractedQuery({
                    query_name: extractedQuery?.osmquery || "",
                    osmquery: e.target.value,
                  })
                }
                rows={10}
                className="bg-slate-600 text-white"
                disabled={activeTab !== "manual"}
              />
            </div>
            {queryState === "extracting_from_osm" && (
              <p className="animate-bounce">Fetching from Overpass . . .</p>
            )}
          </div>
        )}

        {activeTab === "manual" && (
          <label className="text-center self-start p-1 text-sm">
            Name of your query
          </label>
        )}
        <div className="w-full flex items-center justify-center gap-2">
          <Input
            type="email"
            placeholder={
              activeTab === "askgpt" ? "eg: restaurants close to a park" : "query name"
            }
            ref={inputRef}
            onKeyDown={(e) => {
              if (
                e.key === "Enter" &&
                !e.shiftKey &&
                inputRef?.current?.value !== ""
              ) {
                e.preventDefault();
                extractFeatures(inputRef.current?.value || "");
              }
            }}
          />
        </div>
        <Button
          variant={"default"}
          className={`m-2 w-full`}
          onClick={() => {
            if (inputRef?.current?.value !== "") {
              extractFeatures(inputRef.current?.value || "");
            } else {
              inputRef.current?.focus();
            }
          }}
        >
          {
            <RunningSvg
              className={`${
                ["idle", "extraction_done"].includes(queryState)
                  ? "opacity-0 w-[1px]"
                  : "opacity-100 w-[16px]"
              }`}
            />
          }
          RUN
        </Button>

        <Separator className="my-2" />
        <Tabs value={activeTab} className="w-full mt-4">
          <TabsList className="w-full">
            <TabsTrigger
              value="askgpt"
              className={`flex-1 ${
                activeTab === "askgpt" ? "border border-black" : ""
              }`}
              onClick={() => {
                setActiveTab("askgpt");
                inputRef.current!.value = "";
                setQueryState("idle");
                setExtractedQuery({ osmquery: "", query_name: "" });
              }}
            >
              Ask GPT
            </TabsTrigger>
            <TabsTrigger
              value="manual"
              className={`flex-1 ${
                activeTab === "manual" ? "border border-black" : ""
              }`}
              onClick={() => {
                setActiveTab("manual");
                setQueryState("idle");
                inputRef.current!.value = extractedQuery?.query_name || "";
              }}
            >
              Manual Query
            </TabsTrigger>
          </TabsList>
        </Tabs>
      </div>
    </main>
  );
}
