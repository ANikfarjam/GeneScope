"use client";

import { useEffect } from "react";
import AOS from "aos";
import "aos/dist/aos.css";

export default function Home() {
  useEffect(() => {
    AOS.init({ duration: 1000, once: true });
  }, []);

  const cardData = [
    {
      title: "What is Breast Cancer?",
      description:
        "Breast cancer is a disease in which cells in the breast grow uncontrollably. Early detection through advanced techniques can help in better treatment.",
    },
    {
      title: "Machine Learning in Detection",
      description:
        "ML models analyze gene expression and histopathology images to classify benign and malignant cases.",
    },
    {
      title: "How It Works",
      description:
        "Our AI model takes in medical imaging or gene expression data, processes it, and provides a classification result.",
    },
    {
      title: "Early Detection Saves Lives",
      description:
        "Identifying breast cancer at an early stage significantly increases treatment success rates.",
    },
  ];

  return (
    <main className="flex flex-col items-center justify-center min-h-screen bg-white text-black p-6">
      <div className="max-w-4xl text-center">
        <h1 className="text-4xl font-bold mb-4" data-aos="fade-right">
          Breast Cancer Classification
        </h1>
        <p className="text-lg" data-aos="fade-right">
          Understanding breast cancer through machine learning and gene expression analysis.
        </p>

        {/* Cards Section with AOS Animation */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-8">
          {cardData.map((card, index) => (
            <div
              key={index}
              className="bg-white-200 p-6 rounded-lg shadow-md"
              data-aos="fade-left"
              data-aos-delay={index * 200} // Staggered effect
            >
              <h2 className="text-2xl font-semibold">{card.title}</h2>
              <p className="mt-2">{card.description}</p>
            </div>
          ))}
        </div>

        {/* Call to Action */}
        <div className="mt-8" data-aos="zoom-in">
          <a
            href="#"
            className="bg-black hover:bg-gray-800 text-white font-semibold py-3 px-6 rounded-lg transition"
          >
            Learn More
          </a>
        </div>
      </div>
    </main>
  );
}
