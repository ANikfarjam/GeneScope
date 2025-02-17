"use client";

import { useState, useEffect } from "react";
import Typer from "./Typer";
import { BounceLoader} from "react-spinners";

interface Message {
  role: "user" | "assistant";
  content: string;
}

export default function Chatbot() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [user, setUser] = useState<{ username: string; email: string } | null>(null);
  useEffect(() => {
    // Retrieve user from localStorage
    const storedUser = localStorage.getItem("user");
    if (storedUser) {
      setUser(JSON.parse(storedUser));
    }
  }, []);

  const sendMessage = async () => {
    if (!input.trim()) return;
    
    const newMessage: Message = { role: "user", content: input };
    setMessages((prev) => [...prev, newMessage]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: input }),
      });

      const data = await res.json();
      
      const assistantMessage: Message = { role: "assistant", content: data.result };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Chatbot error:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white shadow-md rounded-lg p-6 w-[100%] h-[480px] mx-auto">

      <h2 className="text-xl font-semibold text-gray-800 mb-4">Chatbot</h2>
      <div className="h-80 overflow-y-auto border p-3 rounded-md bg-gray-100">
        {messages.map((msg, index) => (
          <div key={index} className={`p-2 ${msg.role === "user" ? "text-right" : "text-left"}`}>
            <p className={`px-3 py-2 rounded-md inline-block ${msg.role === "user" ? "bg-blue-500 text-white" : "bg-gray-300 text-black"}`}>
              {msg.role === "assistant" ? <Typer text={msg.content} /> : msg.content}
            </p>
          </div>
        ))}
        {loading && <p className="text-gray-500"><BounceLoader color="#2563eb" size={35} /></p>}
      </div>
      <div className="mt-4 flex space-x-2">
        <input
          type="text"
          className="flex-1 p-2 border rounded-md"
          placeholder="Ask me something..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
        />
        <button
          className="bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600 transition"
          onClick={sendMessage}
        >
          Send
        </button>
      </div>
    </div>
  );
}
