"use client";

import { useState, useEffect } from "react";
import Typer from "./Typer";
import { ScaleLoader } from "react-spinners";
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

//for personalizing the chatbot (TODO)
//interface User {
//  username: string;
//  email: string;
//}

export default function Chatbot() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [predictionDone, setPredictionDone] = useState(false);

  const [showCsvPrompt, setShowCsvPrompt] = useState(false);

  useEffect(() => {
    // First message appears immediately
    const timer = setTimeout(() => {
      setShowCsvPrompt(true);
    }, 1900); // Delay long enough for Typer to finish (adjust as needed)

    return () => clearTimeout(timer);
  }, []);

  //personalizing chatbot (TODO)
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

  const handleCsvSubmit = async () => {
    if (!csvFile) return;

    const formData = new FormData();
    formData.append("file", csvFile);

    setLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:5000/api/predict-stage", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("API not connected");
      }

      const data = await response.json();
      const stage = data.predicted_stage;
      const probs: number[] = data.probabilities;
      const clinical = data.clinical_info;

      const stageLabels = [
        "Stage I",
        "Stage IA",
        "Stage IB",
        "Stage II",
        "Stage IIA",
        "Stage IIB",
        "Stage IIIA",
        "Stage IIIB",
        "Stage IIIC",
        "Stage IV",
      ];

      const chartMessage: Message = {
        role: "assistant",
        content: `🧪 Prediction Complete:\n- Stage: ${stage}`,
        chartType: "bar",
        chartData: {
          labels: stageLabels,
          datasets: [
            {
              label: "Prediction Probabilities",
              data: probs,
              backgroundColor: "rgba(54, 162, 235, 0.5)",
              borderColor: "rgba(54, 162, 235, 1)",
              borderWidth: 1,
            },
          ],
        },
      };

      setMessages((prev) => [...prev, chartMessage]);

      // 🧠 Ask LangChain for serious advice based on clinical data
      const explainPrompt = `
  just know that there is a bar chart generated above that displays each stage and their probabilities i dont want to talk about it just know it exist.
  Imagine you just received a patient's clinical data after being predicted that they are at **${stage}**.
  You are a cancer specialist AI.
  Very concisely and seriously advise the patient on what steps they should consider now that they are aware of their stage.
  Make sure to not give any conclusion or overall and encourage to ask more questions, but really make sure it’s a concise response and include age and weight in the conversation.
  Base your suggestions on their **age (${clinical.age_at_index})**, **weight (${clinical.initial_weight})**, and the year of diagnosis (${clinical.year_of_diagnosis}).
      `;

      const res = await fetch("/api/agent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: explainPrompt }),
      });

      const resData = await res.json();

      if (resData.result) {
        const followupMessage: Message = {
          role: "assistant",
          content:
            typeof resData.result === "string"
              ? resData.result
              : JSON.stringify(resData.result),
        };
        setMessages((prev) => [...prev, followupMessage]);
      }

      setTimeout(() => {
        setPredictionDone(true);
      }, 3000);
    } catch (err) {
      console.error("CSV prediction error:", err);
      window.alert(
        "🚨 Could not connect to the model API. Please make sure the Flask server is running."
      );
    } finally {
      setLoading(false);
    }
  };

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
    <div className="px-6 pt-6 w-full h-[90%] min-h-[690px] max-h-[890px] max-w-[1300px] mx-auto text-black flex flex-col bg-transparent shadow-none border-none overflow-hidden">
      <h2 className="text-2xl font-bold text-gray-800 text-center mb-4 hidden">
        Chatbot
      </h2>

      <div className="flex-1 min-h-0 overflow-y-auto border rounded-lg p-4 bg-gray-100 space-y-3 shadow-inner">
        <div className="flex justify-start">
          <div className="px-4 py-2 rounded-xl rounded-tl-none text-sm max-w-[800px] bg-white text-gray-900 border border-gray-300">
            <Typer text="👋 Hi there! I'm GeneScope. Ask me about breast cancer, genetics, or let me show you a chart!" />
          </div>
        </div>

        {showCsvPrompt && (
          <div className="flex justify-start">
            <div className="px-4 py-2 rounded-xl rounded-tl-none text-sm max-w-[800px] bg-white text-gray-900 border border-gray-300 space-y-2">
              <Typer text="🧬 Would you like to use our model for cancer staging prediction? Please upload your CSV file here and press Continue." />
              <div className="flex items-center space-x-3">
                <label className="px-3 py-1 bg-gray-200 text-gray-800 rounded cursor-pointer hover:bg-gray-300 transition">
                  Choose CSV
                  <input
                    type="file"
                    accept=".csv"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) {
                        setCsvFile(file);
                        console.log("Uploaded file:", file.name);
                      }
                    }}
                    className="hidden"
                  />
                </label>

                <button
                  onClick={() => {
                    handleCsvSubmit();
                    setCsvFile(null); // <--- Reset the file after submitting
                    setPredictionDone(false); // <--- Hide the 'Try another' box again
                  }}
                  className="px-4 py-1 bg-pink-600 text-white rounded transition-all duration-300 ease-in-out hover:shadow-xl hover:bg-pink-500 hover:text-black"
                >
                  Continue
                </button>
              </div>
            </div>
          </div>
        )}

        {messages.map((msg, index) => (
          <div
            key={index}
            className={`flex ${
              msg.role === "user" ? "justify-end" : "justify-start"
            }`}
          >
            <div
              className={`px-4 py-2 rounded-xl   text-sm max-w-[800px] ${
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
            <ScaleLoader color="rgba(217, 35, 124, 0.4)" />
          </div>
        )}
        {/* 🔥 New Try Another CSV Button */}
        {predictionDone && (
          <div className="flex justify-start">
            <div className="px-4 py-2 rounded-xl rounded-tl-none text-sm max-w-[800px] bg-white text-gray-900 border border-gray-300 space-y-2">
              <Typer text="🔁 Would you like to try another CSV? Please upload a new one and press Continue." />
              <div className="flex items-center space-x-3">
                <label className="px-3 py-1 bg-gray-200 text-gray-800 rounded cursor-pointer hover:bg-gray-300 transition">
                  Choose CSV
                  <input
                    type="file"
                    accept=".csv"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) {
                        setCsvFile(file);
                        console.log("Uploaded new file:", file.name);
                      }
                    }}
                    className="hidden"
                  />
                </label>

                <button
                  onClick={() => {
                    handleCsvSubmit();
                    setPredictionDone(false);
                  }}
                  className="px-4 py-1 bg-pink-600 text-white rounded transition-all duration-300 ease-in-out hover:shadow-xl hover:bg-pink-500 hover:text-black"
                >
                  Continue
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
      <div className=" w-[60%] min-w-[400px] h-[60px]  mt-4 flex items-center bg-white border border-gray-300 rounded-[22px] p-2 shadow-md mx-auto ">
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
