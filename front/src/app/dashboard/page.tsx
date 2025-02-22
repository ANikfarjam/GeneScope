"use client";

import { Bar, Line, Pie } from "react-chartjs-2";
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
import { useState } from "react";

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
  const [chartData] = useState({
    labels: ["January", "February", "March", "April", "May"],
    datasets: [
      {
        label: "Sales ($)",
        data: [1200, 1900, 3000, 2500, 2800],
        backgroundColor: ["rgba(255, 99, 132, 0.5)"],
        borderColor: ["rgba(255, 99, 132, 1)"],
        borderWidth: 1,
      },
    ],
  });

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

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 max-w-5xl mx-auto">
          {/* Bar Chart */}
          <div className="bg-white shadow-lg rounded-lg p-4">
            <h2 className="text-lg font-semibold text-center mb-2">Sales Data</h2>
            <div className="h-48">
              <Bar data={chartData} />
            </div>
          </div>

          {/* Line Chart */}
          <div className="bg-white shadow-lg rounded-lg p-4">
            <h2 className="text-lg font-semibold text-center mb-2">Sales Trend</h2>
            <div className="h-48">
              <Line data={chartData} />
            </div>
          </div>

          {/* Pie Chart */}
          <div className="bg-white shadow-lg rounded-lg p-4">
            <h2 className="text-lg font-semibold text-center mb-2">Sales Distribution</h2>
            <div className="h-48">
              <Pie
                data={{
                  labels: ["Product A", "Product B", "Product C", "Product D"],
                  datasets: [
                    {
                      label: "Sales Percentage",
                      data: [40, 30, 20, 10],
                      backgroundColor: [
                        "rgba(255, 99, 132, 0.5)",
                        "rgba(54, 162, 235, 0.5)",
                        "rgba(255, 206, 86, 0.5)",
                        "rgba(75, 192, 192, 0.5)",
                      ],
                      borderWidth: 1,
                    },
                  ],
                }}
              />
            </div>
          </div>
        </div>

        {/* Chatbot Section */}
        <div className="mt-10">
          <Chatbot />
        </div>
      </div>
    </div>
  );
}
