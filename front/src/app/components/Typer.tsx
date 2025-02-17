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
    const interval = setInterval(() => {
      if (index < text.length) {
        setDisplayText((prev) => prev + text[index]);
        index++;
      } else {
        clearInterval(interval);
      }
    }, 20); // Typing speed

    return () => clearInterval(interval);
  }, [text]);

  return <ReactMarkdown className="prose max-w-none" remarkPlugins={[remarkGfm]}>{displayText}</ReactMarkdown>;
};

export default Typer;
