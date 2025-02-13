"use client";

import Link from "next/link";
import { useState } from "react";

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <nav className="bg-white shadow-md fixed w-full z-10 top-0 left-0">
      <div className="max-w-6xl mx-auto px-6">
        <div className="flex justify-between items-center py-4">
          {/* Logo */}
          <Link href="/" className="text-2xl font-bold text-pink-600">
            Breast Cancer AI
          </Link>

          {/* Navbar Links (Desktop) */}
          <div className="hidden md:flex space-x-6">
            <Link href="/" className="text-gray-700 hover:text-pink-500">
              Home
            </Link>
            <Link href="/about" className="text-gray-700 hover:text-pink-500">
              About
            </Link>
            <Link href="/visualization" className="text-gray-700 hover:text-pink-500">
              Visualization
            </Link>
            <Link href="/chatbot" className="text-gray-700 hover:text-pink-500">
              ChatBot
            </Link>
            <Link href="/login" className="bg-pink-500 text-white px-4 py-2 rounded-lg hover:bg-pink-600 transition">
              Login
            </Link>
          </div>

          {/* Mobile Menu Button */}
          <button
            className="md:hidden text-gray-700 focus:outline-none"
            onClick={() => setIsOpen(!isOpen)}
          >
            ☰
          </button>
        </div>

        {/* Mobile Menu (Toggle) */}
        {isOpen && (
          <div className="md:hidden flex flex-col space-y-4 pb-4">
            <Link href="/" className="text-gray-700 hover:text-pink-500">
              Home
            </Link>
            <Link href="/about" className="text-gray-700 hover:text-pink-500">
              About
            </Link>
            <Link href="/visualization" className="text-gray-700 hover:text-pink-500">
              Visualization
            </Link>
            <Link href="/chatbot" className="text-gray-700 hover:text-pink-500">
              ChatBot
            </Link>
            <Link href="/login" className="bg-pink-500 text-white px-4 py-2 rounded-lg hover:bg-pink-600 transition">
              Login
            </Link>
          </div>
        )}
      </div>
    </nav>
  );
};

export default Navbar;
