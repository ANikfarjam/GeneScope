"use client";

import ParticleBackground from "@/app/components/ParticleBackground";
import { useEffect } from "react";
import AOS from "aos";
import "aos/dist/aos.css";
import Image from "next/image";
import { motion } from "framer-motion";
import Link from "next/link";

export default function About() {
  useEffect(() => {
    AOS.init({ duration: 1000, once: true });
  }, []);

  const team = [
    {
      name: "Yar Moradpour",
      role: "AI Engineer & Full Stack Developer",
      bio: "Yar contributes to multiple aspects of GeneScope, including supporting model experimentation with NLP and KNN techniques. He also played a key role in building the chatbot with LangChain and developing the main website. Yar bridges science and usability through seamless design and intelligent tools.",
      image: "/assets/about/Yar.webp",
    },
    {
      name: "Ashkan",
      role: "Bioinformatics Lead & Data Scientist",
      bio: "Ashkan specializes in interpreting complex genetic data to uncover meaningful insights. He leads the development of the main deep learning model at the core of GeneScope, ensuring the scientific foundation is rigorous, innovative, and impactful. He also contributed extensively to the research content presented on the website, using Marimo to help structure and communicate the science clearly.",
      image: "/assets/about/Ashkan.jpg",
    },
  ];

  return (
    <main className="flex flex-col items-center justify-center min-h-screen text-black p-6 pt-40 relative z-2">
      <ParticleBackground />

      <div className="max-w-4xl text-center">
        <h1 className="text-4xl font-bold mb-4" data-aos="fade-down">
          About Us
        </h1>
        <p className="text-lg mb-8" data-aos="fade-up">
          GeneScope is built by a passionate team committed to combining
          artificial intelligence and bioinformatics to make cancer research
          more personal, accessible, and actionable. We believe in transparent
          science, meaningful visuals, and a deep respect for human variation.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-10">
          {team.map((member, index) => (
            <motion.div
              key={index}
              className="bg-white rounded-lg shadow-lg p-6"
              data-aos="fade-left"
              data-aos-delay={index * 200}
            >
              <div className="flex flex-col items-center">
                <Image
                  src={member.image}
                  alt={member.name}
                  width={180}
                  height={180}
                  className="rounded-full shadow-md object-cover max-h-[180px]"
                />
                <h2 className="text-xl font-bold mt-4">{member.name}</h2>
                <h3 className="text-sm text-gray-500 mb-2">{member.role}</h3>
                <p className="text-sm text-gray-700 text-center max-w-xs">
                  {member.bio}
                </p>
              </div>
            </motion.div>
          ))}
        </div>

        <div className="mt-12" data-aos="zoom-in">
          <Link
            href="/"
            className="bg-black hover:bg-gray-800 text-white font-semibold py-3 px-6 rounded-lg transition"
          >
            Back to Home
          </Link>
        </div>
      </div>
    </main>
  );
}
