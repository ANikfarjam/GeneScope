"use client";

import { useState, useEffect } from "react";
//import Link from "next/link";
import { FiMenu, FiX, FiMessageCircle } from "react-icons/fi";

interface Message {
  role: "user" | "assistant";
  content: string;
}

const Sidebar = () => {
  const [isOpen, setIsOpen] = useState(true);
  const [history, setHistory] = useState<Message[]>([]);

  useEffect(() => {
    const storedHistory = localStorage.getItem("chatHistory");
    if (storedHistory) {
      setHistory(JSON.parse(storedHistory));
    }
  }, []);

  return (
    <div className={`h-screen bg-gray-900 text-white flex flex-col ${isOpen ? "w-64" : "w-16"} transition-all duration-300 fixed`}>
      <div className="p-4 flex justify-between items-center">
        {isOpen && <h2 className="text-xl font-bold">History</h2>}
        <button onClick={() => setIsOpen(!isOpen)} className="text-white">
          {isOpen ? <FiX size={24} /> : <FiMenu size={24} />}
        </button>
      </div>

      <nav className="flex flex-col space-y-4 mt-4 p-2">
        {history.length > 0 ? (
          history.map((msg, index) => (
            <div key={index} className="text-sm bg-gray-800 p-2 rounded-md truncate hover:bg-gray-700">
              <FiMessageCircle className="inline mr-2" />
              {msg.content.slice(0, 20)}...
            </div>
          ))
        ) : (
          <p className="text-gray-400 text-sm px-2">No history yet</p>
        )}
      </nav>
    </div>
  );
};

export default Sidebar;
