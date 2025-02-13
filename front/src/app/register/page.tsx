"use client";

import { useState } from "react";
import { signUp } from "../auth";
import Link from "next/link";
import "../styles/register.css"; // Import CSS for styling

export default function RegisterPage() {
  const [email, setEmail] = useState<string>("");
  const [password, setPassword] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    const response = await signUp(email, password);
    if (response.error) {
      setError(response.error);
    } else {
      alert("Account created successfully!");
    }
  };

  return (
    <div className="container" onClick={() => {}}>
      <div className="top"></div>
      <div className="bottom"></div>
      <div className="center">
        <h2>Create an Account</h2>
        {error && <p className="text-red-500">{error}</p>}
        <form onSubmit={handleRegister} className="w-full">
          <input
            type="email"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
          <button type="submit" className="register-btn">
            Register
          </button>
        </form>
        <p className="mt-4 text-gray-600">
          Already have an account?{" "}
          <Link href="/login" className="text-blue-600 hover:underline">
            Login
          </Link>
        </p>
      </div>
    </div>
  );
}
