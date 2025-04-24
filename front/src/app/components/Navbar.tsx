"use client";

import Link from "next/link";
import { useState, useEffect, useRef } from "react";
import Image from "next/image";
const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [user, setUser] = useState<{ username: string; email: string } | null>(
    null
  );

  const dropdownRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const storedUser = localStorage.getItem("user");
    if (storedUser) {
      setUser(JSON.parse(storedUser));
    }

    const handleClickOutside = (event: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setDropdownOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleLogout = () => {
    localStorage.removeItem("user");
    setUser(null);
    setDropdownOpen(false);
  };

  return (
    <nav className="bg-white shadow-md fixed w-full z-50 top-0 left-0">
      <div className="max-w-6xl mx-auto px-6">
        <div className="flex justify-between items-center py-4">
          <Link href="/" className="text-2xl font-bold text-pink-600">
            GeneScope
          </Link>

          <div className="hidden md:flex space-x-6 items-center">
            <Link
              href="/analyticalmethod"
              className="text-gray-700 hover:text-pink-500"
            >
              Analytical Methods
            </Link>
            <Link
              href="/majorfindings"
              className="text-gray-700 hover:text-pink-500"
            >
              Major Finding
            </Link>
            <Link
              href="/projectobjective"
              className="text-gray-700 hover:text-pink-500"
            >
              Project Objective
            </Link>

            {user ? (
              <div className="relative" ref={dropdownRef}>
                <Image
                  src="/assets/appIcons/guest_user.png"
                  alt="User Avatar"
                  width={40}
                  height={40}
                  onClick={() => setDropdownOpen((prev) => !prev)}
                  className="w-10 h-10 rounded-full cursor-pointer border-2 border-pink-400"
                />
                {dropdownOpen && (
                  <div className="absolute right-0 mt-2 w-48 bg-white shadow-lg rounded-lg p-4 z-50">
                    <p className="text-sm font-semibold text-gray-700">
                      👤 {user.username}
                    </p>
                    <p className="text-sm text-gray-500 mb-2">{user.email}</p>
                    <button
                      onClick={handleLogout}
                      className="w-full bg-red-500 text-white py-2 rounded hover:bg-red-600 transition"
                    >
                      Logout
                    </button>
                  </div>
                )}
              </div>
            ) : (
              <Link
                href="/login"
                className="bg-pink-500 text-white px-4 py-2 rounded-lg hover:bg-pink-600 transition"
              >
                Login
              </Link>
            )}
          </div>

          <button
            className="md:hidden text-gray-700 focus:outline-none"
            onClick={() => setIsOpen(!isOpen)}
          >
            ☰
          </button>
        </div>

        {/* Mobile Menu */}
        {isOpen && (
          <div className="md:hidden flex flex-col space-y-4 pb-4">
            <Link href="/about" className="text-gray-700 hover:text-pink-500">
              About
            </Link>
            <Link
              href="/analyticalmethod"
              className="text-gray-700 hover:text-pink-500"
            >
              Analytical Methods
            </Link>
            <Link
              href="/majorfindings"
              className="text-gray-700 hover:text-pink-500"
            >
              Major Finding
            </Link>
            <Link
              href="/projectobjective"
              className="text-gray-700 hover:text-pink-500"
            >
              Project Objective
            </Link>

            {user ? (
              <div className="text-center space-y-2">
                <Image
                  src="/assets/appIcons/guest_user.png"
                  alt="User Avatar"
                  width={48}
                  height={48}
                  className="w-12 h-12 rounded-full mx-auto border-2 border-pink-400"
                />
                <p className="text-sm font-semibold text-gray-700">
                  👤 {user.username}
                </p>
                <p className="text-sm text-gray-500">{user.email}</p>
                <button
                  onClick={handleLogout}
                  className="bg-red-500 text-white py-2 rounded hover:bg-red-600 transition"
                >
                  Logout
                </button>
              </div>
            ) : (
              <Link
                href="/login"
                className="bg-pink-500 text-white px-4 py-2 rounded-lg hover:bg-pink-600 transition text-center"
              >
                Login
              </Link>
            )}
          </div>
        )}
      </div>
    </nav>
  );
};

export default Navbar;
