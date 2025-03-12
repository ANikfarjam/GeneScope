"use client";
import { useEffect, useState } from "react";
import { motion } from "framer-motion";

export default function ParticleBackground() {
  const [particles, setParticles] = useState<{ x: number; y: number }[]>([]);

  useEffect(() => {
    if (typeof window === "undefined") return; // Prevents SSR issues

    const newParticles = Array.from({ length: 20 }).map(() => ({
      x: Math.random() * window.innerWidth,
      y: Math.random() * window.innerHeight,
    }));

    setParticles(newParticles);
  }, []);

  return (
    <div className="fixed top-0 left-0 w-full h-full -z-10 overflow-hidden bg-white">
      {particles.map((particle, i) => (
        <motion.div
          key={i}
          className="absolute w-4 h-4 bg-black rounded-full opacity-70 drop-shadow-lg"
          initial={{ x: particle.x, y: particle.y }}
          animate={{
            y: [
              Math.random() * window.innerHeight,
              Math.random() * window.innerHeight * 0.5,
              Math.random() * window.innerHeight,
            ],
            x: [
              Math.random() * window.innerWidth,
              Math.random() * window.innerWidth * 0.8,
              Math.random() * window.innerWidth,
            ],
            opacity: [0.2, 0.5, 0.8, 0.3],
          }}
          transition={{
            duration: 5 + Math.random() * 5,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />
      ))}
    </div>
  );
}
