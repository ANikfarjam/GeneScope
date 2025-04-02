"use client";

import { useState, useEffect } from "react";
import Typer from "./Typer";
import { BounceLoader } from "react-spinners";
import { FiSend, FiPlus, FiMoreHorizontal } from "react-icons/fi";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Bar, Line, Pie } from "react-chartjs-2";
import { ChartData } from "chart.js";

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
  chartData?: ChartData<"bar" | "line" | "pie">;
}

//interface User {
//  username: string;
//  email: string;
//}

export default function Chatbot() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const [showCsvPrompt, setShowCsvPrompt] = useState(false);

  useEffect(() => {
    // First message appears immediately
    const timer = setTimeout(() => {
      setShowCsvPrompt(true);
    }, 1900); // Delay long enough for Typer to finish (adjust as needed)

    return () => clearTimeout(timer);
  }, []);
  //const [user, setUser] = useState<User | null>(null);

  //useEffect(() => {
  //  const storedUser = localStorage.getItem("user");
  //  if (storedUser) {
  //    setUser(JSON.parse(storedUser));
  //  }
  //}, []);

  useEffect(() => {
    localStorage.setItem("chatHistory", JSON.stringify(messages));
  }, [messages]);

  //const extractChartData = (input: string) => {
  //  const words = input.split(/\s+/); // Split input by spaces
  //  const labels: string[] = [];
  //  const data: number[] = [];
  //
  //  let lastLabel = "Label"; // Default label
  //  words.forEach((word) => {
  //    const num = parseFloat(word);
  //    if (!isNaN(num)) {
  //      data.push(num);
  //      labels.push(lastLabel);
  //    } else {
  //      lastLabel = word; // Assume a non-number word is a label
  //    }
  //  });
  //
  //  if (data.length === 0) {
  //    return null; // No valid data found
  //  }
  //
  //  return {
  //    labels: labels.length ? labels : ["A", "B", "C", "D"],
  //    datasets: [
  //      {
  //        label: "User Data",
  //        data: data,
  //        backgroundColor: [
  //          "rgba(255, 99, 132, 0.5)",
  //          "rgba(54, 162, 235, 0.5)",
  //          "rgba(255, 206, 86, 0.5)",
  //          "rgba(75, 192, 192, 0.5)",
  //          "rgba(153, 102, 255, 0.5)",
  //        ],
  //        borderColor: [
  //          "rgba(255, 99, 132, 1)",
  //          "rgba(54, 162, 235, 1)",
  //          "rgba(255, 206, 86, 1)",
  //          "rgba(75, 192, 192, 1)",
  //          "rgba(153, 102, 255, 1)",
  //        ],
  //        borderWidth: 1,
  //      },
  //    ],
  //  };
  //};

  const sendMessage = async () => {
    if (!input.trim()) return;

    const newMessage: Message = { role: "user", content: input };
    setMessages((prev) => [...prev, newMessage]);
    setInput("");
    setLoading(true);

    try {
      let assistantMessage: Message;

      const res = await fetch("/api/agent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: input }),
      });

      const data = await res.json();
      let parsed;
      try {
        parsed =
          typeof data.result === "string"
            ? JSON.parse(data.result)
            : data.result;
        console.log("Parsed result:", parsed);

        if (parsed?.type && parsed?.data) {
          assistantMessage = {
            role: "assistant",
            content: `Here is your ${parsed.type} chart:`,
            chartType: parsed.type,
            chartData: parsed.data,
          };
        } else {
          assistantMessage = { role: "assistant", content: data.result };
        }
      } catch {
        assistantMessage = { role: "assistant", content: data.result };
      }

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Chatbot error:", error);
    } finally {
      setLoading(false);
    }
  };
  /*
  these are probably not needed
  !msg.content.includes("http") &&
  !msg.content.includes("![") && <Typer text={msg.content} />}
*/
  return (
    <div className="p-6 w-full h-[800px] max-w-[1050px] mx-auto text-black flex flex-col bg-transparent shadow-none border-none">
      <h2 className="text-2xl font-bold text-gray-800 text-center mb-4 hidden">
        Chatbot
      </h2>

      {/* Chat Messages Container */}
      <div className="flex-1 overflow-y-auto border rounded-lg p-4 bg-gray-100 space-y-3 shadow-inner">
        <div className="flex justify-start">
          <div className="px-4 py-2 rounded-xl rounded-tl-none text-sm max-w-[80%] bg-white text-gray-900 border border-gray-300">
            <Typer text="👋 Hi there! I'm GeneScope. Ask me about breast cancer, genetics, or let me show you a chart!" />
          </div>
        </div>

        {showCsvPrompt && (
          <>
            <div className="flex justify-start">
              <div className="px-4 py-2 rounded-xl rounded-tl-none text-sm max-w-[80%] bg-white text-gray-900 border border-gray-300 space-y-2">
                <Typer text="🧬 Would you like to use our model for prognosis? Please upload your CSV file here and press Continue." />
                <div className="flex items-center space-x-3">
                  {/* Custom File Upload */}
                  <label className="px-3 py-1 bg-gray-200 text-gray-800 rounded cursor-pointer hover:bg-gray-300 transition">
                    Choose CSV
                    <input
                      type="file"
                      accept=".csv"
                      onChange={(e) => {
                        const file = e.target.files?.[0];
                        if (file) {
                          console.log("Uploaded file:", file.name);
                          // Store in state if needed
                        }
                      }}
                      className="hidden"
                    />
                  </label>

                  {/* Continue Button */}
                  <button
                    onClick={() => {
                      console.log("Continue button clicked");
                    }}
                    className="px-4 py-1 bg-blue-600 text-white rounded hover:bg-blue-700 transition"
                  >
                    Continue
                  </button>
                </div>
              </div>
            </div>
          </>
        )}

        {messages.map((msg, index) => (
          <div
            key={index}
            className={`flex ${
              msg.role === "user" ? "justify-end" : "justify-start"
            }`}
          >
            <div
              className={`px-4 py-2 rounded-xl   text-sm max-w-[80%] ${
                msg.role === "user"
                  ? "bg-blue-500 text-white rounded-tr-none"
                  : "bg-white rounded-tl-none text-gray-900 border border-gray-300 "
              }`}
            >
              {msg.role === "assistant" ? (
                <>
                  {msg.content &&
                    !msg.content.includes("http") &&
                    !msg.content.includes("![") && <Typer text={msg.content} />}

                  {msg.chartType && msg.chartData && (
                    <div className="h-48 w-80 mt-2">
                      {msg.chartType === "bar" && (
                        <Bar data={msg.chartData as ChartData<"bar">} />
                      )}
                      {msg.chartType === "line" && (
                        <Line data={msg.chartData as ChartData<"line">} />
                      )}
                      {msg.chartType === "pie" && (
                        <Pie data={msg.chartData as ChartData<"pie">} />
                      )}
                    </div>
                  )}
                </>
              ) : (
                <ReactMarkdown
                  className="prose max-w-none"
                  remarkPlugins={[remarkGfm]}
                >
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
        <button
          className="p-2 text-blue-600 hover:text-blue-800 transition"
          onClick={sendMessage}
        >
          <FiSend size={22} />
        </button>
      </div>
    </div>
  );
}
