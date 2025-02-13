"use client";

import { useState } from "react";
import { useRouter } from "next/router";
import { login } from "../auth";
import Link from "next/link";
import "../styles/login.css"; // Import CSS for styling

export default function LoginPage() {
  const [email, setEmail] = useState<string>("");
  const [password, setPassword] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();
  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    const response = await login(email, password);
    if (response.error) {
      setError(response.error);
    } else {
      console.log("Logged in successfully!");
      //for now we redirect to home page
      router.push("/");
    }
  };

  return (
    <div className="container" onClick={() => {}}>
      <div className="top"></div>
      <div className="bottom"></div>
      <div className="center">
        <h2>Please Sign In</h2>
        {error && <p className="text-red-500">{error}</p>}
        <form onSubmit={handleLogin} className="w-full">
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
          <button type="submit" className="login-btn">
            Login
          </button>
        </form>
        <p className="mt-4 text-gray-600">
          Don't have an account?{" "}
          <Link href="/register" className="text-blue-600 hover:underline">
            Register
          </Link>
        </p>
      </div>
    </div>
  );
}
