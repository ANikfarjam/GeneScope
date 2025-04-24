"use client";

import Sidebar from "../components/sideBar";
import Chatbot from "../components/ChatBot";
import { useState, useEffect } from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
//import { useState } from "react";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
);

export default function Dashboard() {
  const [showSidebar, setShowSidebar] = useState<boolean>(true);
  //used only so sidebar wont effect the chat interface
  //if window gets smaller
  useEffect(() => {
    const mediaQuery = window.matchMedia("(min-width: 868px)");
    setShowSidebar(mediaQuery.matches);
    const handleResize = (e: MediaQueryListEvent) => {
      setShowSidebar(e.matches);
    };
    mediaQuery.addEventListener("change", handleResize);
    return () => {
      mediaQuery.removeEventListener("change", handleResize);
    };
  }, []);

  return (
    <div className="flex">
      {showSidebar && <Sidebar />}
      <div className="flex-1 min-h-screen bg-gray-100 p-6 ml-16 md:ml-64 transition-all duration-300">
        <h1 className="text-3xl font-bold text-center text-gray-800 mb-8">
          Dashboard
        </h1>
        <div className="mt-10">
          <Chatbot />
        </div>
      </div>
    </div>
  );
}
