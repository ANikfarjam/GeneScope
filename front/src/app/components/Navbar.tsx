"use client";

import Link from "next/link";
import { useState, useEffect } from "react";

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [user, setUser] = useState<{ username: string; email: string } | null>(
    null
  );

  useEffect(() => {
    const storedUser = localStorage.getItem("user");
    if (storedUser) {
      setUser(JSON.parse(storedUser));
    }
  }, []);

  const handleLogout = () => {
    localStorage.removeItem("user");
    setUser(null);
  };

  //  <Link href="/about" className="text-gray-700 hover:text-pink-500">
  //    About
  //  </Link>
  return (
    <nav className="bg-white shadow-md fixed w-full z-50 top-0 left-0">
      <div className="max-w-6xl mx-auto px-6">
        <div className="flex justify-between items-center py-4">
          <Link href="/" className="text-2xl font-bold text-pink-600">
            GeneScope
          </Link>
          <div className="hidden md:flex space-x-6">
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
              <>
                <span className="text-gray-700">Welcome, {user.username}!</span>
                <button
                  onClick={handleLogout}
                  className="bg-red-500 text-white px-4 py-2 rounded-lg hover:bg-red-600 transition"
                >
                  Logout
                </button>
              </>
            ) : (
              // If no user, show Login button
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
              <>
                <span className="text-gray-700 text-center">
                  Welcome, {user.username}!
                </span>
                <button
                  onClick={handleLogout}
                  className="bg-red-500 text-white px-6 py-2 h-10 rounded-lg hover:bg-red-600 transition"
                >
                  Logout
                </button>
              </>
            ) : (
              <Link
                href="/login"
                className="bg-red-500 text-white px-6 py-3 h-5 rounded-lg hover:bg-red-600 transition"
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
