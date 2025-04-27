"use client";

import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface TyperProps {
  text?: string; // Make text optional
  speed?: number; // Optional: milliseconds per character (default 30ms)
}

const Typer = ({ text, speed = 5 }: TyperProps) => {
  const [displayText, setDisplayText] = useState("");

  useEffect(() => {
    if (!text || typeof window === "undefined") return;
    let index = 0;
    let lastTime = performance.now();
    setDisplayText("");

    const step = (now: number) => {
      const elapsed = now - lastTime;

      if (elapsed >= speed) {
        if (index < text.length) {
          setDisplayText(text.slice(0, index + 1));
          index++;
          lastTime = now;
        }
      }

      if (index < text.length) {
        requestAnimationFrame(step);
      }
    };

    const animationId = requestAnimationFrame(step);

    return () => cancelAnimationFrame(animationId);
  }, [text, speed]);

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
          thead: ({ children }) => (
            <thead className="bg-gray-200">{children}</thead>
          ),
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
