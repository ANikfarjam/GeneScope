"use client";

import Sidebar from "../components/sideBar";
import Chatbot from "../components/ChatBot";

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
  //const [chartData] = useState({
  //  labels: ["January", "February", "March", "April", "May"],
  //  datasets: [
  //    {
  //      label: "Sales ($)",
  //      data: [1200, 1900, 3000, 2500, 2800],
  //      backgroundColor: ["rgba(255, 99, 132, 0.5)"],
  //      borderColor: ["rgba(255, 99, 132, 1)"],
  //      borderWidth: 1,
  //    },
  //  ],
  //});

  //// Chart options
  //const options = {
  //  responsive: true,
  //  plugins: {
  //    legend: {
  //      display: true,
  //    },
  //    title: {
  //      display: true,
  //    },
  //  },
  //};

  return (
    <div className="flex">
      {/* Sidebar */}
      <Sidebar />

      {/* Dashboard Content */}
      <div className="flex-1 min-h-screen bg-gray-100 p-6 ml-16 md:ml-64 transition-all duration-300">
        <h1 className="text-3xl font-bold text-center text-gray-800 mb-8">Dashboard</h1>

 

        {/* Chatbot Section */}
        <div className="mt-10">
          <Chatbot />
        </div>
      </div>
    </div>
  );
}
