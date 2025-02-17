"use client";

import { useState } from "react";
import Link from "next/link";
import { FiHome, FiBarChart, FiUser, FiMenu, FiX } from "react-icons/fi";

const Sidebar = () => {
  const [isOpen, setIsOpen] = useState(true);

  return (
    <div className={`h-screen bg-gray-900 text-white flex flex-col ${isOpen ? "w-64" : "w-16"} transition-all duration-300 fixed`}>
      {/* Toggle Button */}
      <div className="p-4 flex justify-between items-center">
        {isOpen && <h2 className="text-xl font-bold">Dashboard</h2>}
        <button onClick={() => setIsOpen(!isOpen)} className="text-white">
          {isOpen ? <FiX size={24} /> : <FiMenu size={24} />}
        </button>
      </div>

      {/* Navigation Links */}
      <nav className="flex flex-col space-y-4 mt-4">
        <SidebarLink href="/" icon={<FiHome size={20} />} text="Home" isOpen={isOpen} />
        <SidebarLink href="/dashboard" icon={<FiBarChart size={20} />} text="Dashboard" isOpen={isOpen} />
        <SidebarLink href="/profile" icon={<FiUser size={20} />} text="Profile" isOpen={isOpen} />
      </nav>
    </div>
  );
};

// Reusable Sidebar Link Component
const SidebarLink = ({ href, icon, text, isOpen }: { href: string; icon: React.ReactNode; text: string; isOpen: boolean }) => (
  <Link href={href} className="flex items-center space-x-2 p-3 hover:bg-gray-700 rounded-md transition">
    {icon}
    {isOpen && <span>{text}</span>}
  </Link>
);

export default Sidebar;
