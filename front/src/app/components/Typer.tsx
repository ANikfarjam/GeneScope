"use client";

import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface TyperProps {
  text: string;
}

const Typer = ({ text }: TyperProps) => {
  const [displayText, setDisplayText] = useState("");

  useEffect(() => {
    let index = 0;
    setDisplayText(""); // Reset text on input change

    const interval = setInterval(() => {
      if (index < text.length) {
        setDisplayText(text.slice(0, index + 1));
        index++;
      } else {
        clearInterval(interval);
      }
    }, 20); // Typing speed

    return () => clearInterval(interval);
  }, [text]);

  return (
    <div className="prose max-w-none prose-lg">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          table: ({ children }) => (
            <div className="overflow-x-auto">
              <table className="w-full border-collapse border border-gray-300 rounded-lg shadow-md">
                {children}
              </table>
            </div>
          ),
          thead: ({ children }) => <thead className="bg-gray-200">{children}</thead>,
          th: ({ children }) => (
            <th className="border border-gray-400 px-4 py-2 text-left font-semibold bg-gray-100">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="border border-gray-300 px-4 py-2">{children}</td>
          ),
          tr: ({ children }) => <tr className="even:bg-gray-50">{children}</tr>,
        }}
      >
        {displayText}
      </ReactMarkdown>
    </div>
  );
};

export default Typer;
