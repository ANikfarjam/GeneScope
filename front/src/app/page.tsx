"use client";

import { useEffect } from "react";
import AOS from "aos";
import "aos/dist/aos.css";
import { Bar } from "react-chartjs-2";
import { Chart, registerables } from "chart.js";

Chart.register(...registerables);

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
        "ML models analyze medical data, including histopathology images, to classify benign and malignant cases.",
    },
    {
      title: "How It Works",
      description:
        "Our AI model processes medical imaging or gene expression data and provides a classification result.",
    },
    {
      title: "Early Detection Saves Lives",
      description:
        "Identifying breast cancer at an early stage significantly increases treatment success rates.",
    },
  ];

  const casualtiesData = {
    labels: ["2018", "2019", "2020", "2021", "2022", "2023"],
    datasets: [
      {
        label: "Breast Cancer Deaths (Worldwide)",
        data: [626000, 640000, 685000, 670000, 670000, 700000], // Sample Data
        backgroundColor: "rgba(255, 99, 132, 0.6)",
        borderColor: "rgba(255, 99, 132, 1)",
        borderWidth: 1,
      },
    ],
  };

  return (
    <main className="flex flex-col items-center justify-center min-h-screen bg-white text-black p-6 pt-20">
      <div className="max-w-4xl text-center">
        <h1 className="text-4xl font-bold mb-4" data-aos="fade-right">
          Breast Cancer Classification
        </h1>
        <p className="text-lg mb-8" data-aos="fade-right">
          Understanding breast cancer through technology and data analysis. Raising awareness and improving detection can save lives.
        </p>
        <div className="mt-12 shadow-md p-4">
  {/* Title on Top */}
  <h2 className="text-2xl font-bold mb-6 text-center" data-aos="fade-up">
    The Impact of Breast Cancer
  </h2>

  {/* Flex container for Graph & Explanation */}
  <div className="flex flex-col md:flex-row items-center justify-between gap-6">
    {/* Graph on the Left */}
    <div className="w-full md:w-1/2" data-aos="fade-right">
      <Bar data={casualtiesData} options={{ responsive: true }} />
    </div>

    {/* Explanation on the Right */}
    <div className="w-full md:w-1/2 text-left" data-aos="fade-left">
      <p className="text-lg">
        Breast cancer remains one of the leading causes of cancer-related deaths worldwide. 
        The number of casualties has been steadily increasing over the years, with over 
        <span className="font-semibold"> 700,000 deaths in 2023.</span> 
        Early detection and improved treatments are crucial in reducing mortality rates.
      </p>
    </div>
  </div>
</div>

        {/* Card Section */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-8">
          {cardData.map((card, index) => (
            <div
              key={index}
              className="bg-gray-100 p-6 rounded-lg shadow-md"
              data-aos="fade-left"
              data-aos-delay={index * 200} // Staggered effect
            >
              <h2 className="text-2xl font-semibold">{card.title}</h2>
              <p className="mt-2">{card.description}</p>
            </div>
          ))}
        </div>

        {/* Call to Action */}
        <div className="mt-12" data-aos="zoom-in">
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
