"use client";

import { useState, useEffect } from "react";
import Typer from "./Typer";
import { BounceLoader } from "react-spinners";
import { FiSend, FiPlus, FiMoreHorizontal } from "react-icons/fi";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Bar, Line, Pie } from "react-chartjs-2";
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

interface Message {
  role: "user" | "assistant";
  content: string;
  chartType?: "bar" | "line" | "pie";
  chartData?: any;
}

interface User {
  username: string;
  email: string;
}

export default function Chatbot() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    const storedUser = localStorage.getItem("user");
    if (storedUser) {
      setUser(JSON.parse(storedUser));
    }
  }, []);

  const generateChartData = (type: "bar" | "line" | "pie") => {
    const data = {
      labels: ["January", "February", "March", "April", "May"],
      datasets: [
        {
          label: "Sales ($)",
          data: [1200, 1900, 3000, 2500, 2800],
          backgroundColor: [
            "rgba(255, 99, 132, 0.5)",
            "rgba(54, 162, 235, 0.5)",
            "rgba(255, 206, 86, 0.5)",
            "rgba(75, 192, 192, 0.5)",
            "rgba(153, 102, 255, 0.5)",
          ],
          borderColor: [
            "rgba(255, 99, 132, 1)",
            "rgba(54, 162, 235, 1)",
            "rgba(255, 206, 86, 1)",
            "rgba(75, 192, 192, 1)",
            "rgba(153, 102, 255, 1)",
          ],
          borderWidth: 1,
        },
      ],
    };
    return { type, data };
  };

  const sendMessage = async () => {
    if (!input.trim()) return;

    const newMessage: Message = { role: "user", content: input };
    setMessages((prev) => [...prev, newMessage]);
    setInput("");
    setLoading(true);

    try {
      let chartType: "bar" | "line" | "pie" | null = null;

      if (input.toLowerCase().includes("bar chart")) chartType = "bar";
      else if (input.toLowerCase().includes("line chart")) chartType = "line";
      else if (input.toLowerCase().includes("pie chart")) chartType = "pie";

      let assistantMessage: Message;

      if (chartType) {
        const chartData = generateChartData(chartType);
        assistantMessage = {
          role: "assistant",
          content: `Here is your ${chartType} chart:`,
          chartType,
          chartData: chartData.data,
        };
      } else {
        // Normal AI-generated response
        const res = await fetch("/api/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt: input, user: user }),
        });

        const data = await res.json();
        assistantMessage = { role: "assistant", content: data.result };
      }

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Chatbot error:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white shadow-lg rounded-2xl p-6 w-full h-[500px] max-w-[1050px] mx-auto text-black flex flex-col">
      <h2 className="text-2xl font-bold text-gray-800 text-center mb-4">Chatbot</h2>

      {/* Chat Messages Container */}
      <div className="flex-1 overflow-y-auto border rounded-lg p-4 bg-gray-100 space-y-3 shadow-inner">
        {messages.map((msg, index) => (
          <div key={index} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
<div
  className={`px-4 py-2 rounded-xl text-sm max-w-[80%] ${
    msg.role === "user"
      ? "bg-blue-500 text-white"
      : "bg-white text-gray-900 border border-gray-300 "
  }`}
>

              {msg.role === "assistant" ? (
                msg.chartType ? (
                  <div className="h-48 w-80">
                    {msg.chartType === "bar" && <Bar data={msg.chartData} />}
                    {msg.chartType === "line" && <Line data={msg.chartData} />}
                    {msg.chartType === "pie" && <Pie data={msg.chartData} />}
                  </div>
                ) : (
                  <Typer text={msg.content} />
                )
              ) : (
                <ReactMarkdown className="prose max-w-none" remarkPlugins={[remarkGfm]}>
                  {msg.content}
                </ReactMarkdown>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-center mt-2">
            <BounceLoader color="#2563eb" size={30} />
          </div>
        )}
      </div>

      {/* Input Box */}
      <div className="mt-4 flex items-center bg-white border border-gray-300 rounded-full p-2 shadow-md mx-auto ">
        <button className="p-2 text-gray-500 hover:text-gray-700 transition">
          <FiPlus size={20} />
        </button>
        <input
          type="text"
          className="flex-1 p-2 bg-transparent border-none focus:outline-none text-gray-800 placeholder-gray-400"
          placeholder="Type your message..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
        />
        <button className="p-2 text-gray-500 hover:text-gray-700 transition">
          <FiMoreHorizontal size={20} />
        </button>
        <button className="p-2 text-blue-600 hover:text-blue-800 transition" onClick={sendMessage}>
          <FiSend size={22} />
        </button>
      </div>
    </div>
  );
}
