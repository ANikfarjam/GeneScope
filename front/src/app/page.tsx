"use client";
import ParticleBackground from "./components/ParticleBackground";
import { useEffect } from "react";
import AOS from "aos";
import "aos/dist/aos.css";
import { Bar } from "react-chartjs-2";
import { Doughnut } from "react-chartjs-2";
import { Chart, registerables } from "chart.js";
import Typer from "./components/Typer";
import { motion } from "framer-motion"; // Import Framer Motion
import Link from "next/link";
import { FaChartBar, FaSearch, FaClipboardList } from "react-icons/fa";

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
    labels: ["All Ages", "<40", "40-49", "50-59", "60-69", "70-79", "80+"],
    datasets: [
      {
        label: "Number of Cases",
        data: [231840, 10500, 35850, 54060, 59990, 42480, 28960],
        backgroundColor: [
          "#2ecc71",
          "#e74c3c",
          "#f1c40f",
          "#3498db",
          "#2ecc71",
          "#e67e22",
          "#f1c40f",
        ],
      },
    ],
  };
  const pieData = {
    labels: ["Localized", "Regional", "Distant", "Unstaged"],
    datasets: [
      {
        data: [66.0, 25.8, 5.8, 2.4],
        backgroundColor: ["#1abc9c", "#2c3e50", "#e74c3c", "#34495e"],
        hoverOffset: 10,
      },
    ],
  };
  const pieOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: "top",
        labels: {
          color: "#000",
          font: {
            size: 14,
            weight: "bold",
          },
        },
      },
      title: {
        display: true,
        text: "Stage Distribution % of New Breast Cancer Cases in the U.S.",
        font: {
          size: 18,
          weight: "bold",
        },
        color: "#e67e22",
      },
    },
  } as const;
  return (
    <main className="flex flex-col items-center justify-center min-h-screen  text-black p-6 pt-40 z-2 ">
      <ParticleBackground />
      <div className="max-w-4xl text-center">
        <h1 className="text-4xl font-bold mb-4" data-aos="fade-right">
          Breast Cancer Classification
        </h1>
        <div
          className="text-lg mb-8 h-32 w-full overflow-hidden relative text-left pt-2"
          data-aos="fade-right"
        >
          <Typer
            text="GeneScope is an AI-powered dashboard that leverages deep learning models to provide prognosis and analytical insights based on patients' gene expression data. 
  By analyzing complex biological patterns, it predicts cancer progression, assesses risk levels, and offers data-driven recommendations. 
  "
          />
        </div>

        <div className="mt-12 shadow-lg p-4 inset-shadow-xs bg-white">
          <h2
            className="text-2xl font-bold mb-6 text-center "
            data-aos="fade-up"
          >
            The Impact of Breast Cancer
          </h2>

          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            <motion.div
              className="w-full md:w-1/2 p-2 "
              data-aos="fade-right"
              whileHover={{ boxShadow: "0px 10px 30px rgba(0, 0, 0, 0.2)" }}
              transition={{ duration: 0.2 }}
            >
              <Bar data={casualtiesData} options={{ responsive: true }} />
            </motion.div>

            <motion.div
              className="w-full md:w-1/2 text-left p-2"
              data-aos="fade-left"
              whileHover={{ boxShadow: "0px 10px 30px rgba(0, 0, 0, 0.2)" }}
              transition={{ duration: 0.2 }}
            >
              <p className="text-lg">
                In 2015, an estimated 231,840 new invasive breast cancer cases
                were reported in the U.S., with the highest incidence in
                individuals aged 50-69. While cases in younger individuals were
                lower, early detection remains critical for improving outcomes.
                Most cases occur in older adults, reinforcing the need for
                routine screening and timely intervention to reduce mortality.
              </p>
            </motion.div>
          </div>
        </div>
        {/* sadddddddddddddddddddddddddddddddd*/}
        <div className="mt-12 shadow-lg p-4 inset-shadow-xs bg-white">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            <motion.div
              className="w-full md:w-1/2 p-2 "
              data-aos="fade-right"
              whileHover={{ boxShadow: "0px 10px 30px rgba(0, 0, 0, 0.2)" }}
              transition={{ duration: 0.2 }}
            >
              <Doughnut data={pieData} options={pieOptions} />
            </motion.div>

            <motion.div
              className="w-full md:w-1/2 text-left p-2"
              data-aos="fade-left"
              whileHover={{ boxShadow: "0px 10px 30px rgba(0, 0, 0, 0.2)" }}
              transition={{ duration: 0.2 }}
            >
              <p className="text-lg">
                Early detection saves lives. In the U.S., 66% of breast cancer
                cases are diagnosed at a localized stage, where survival rates
                exceed 90% with timely treatment. However, nearly 26% of cases
                spread to lymph nodes, and 5.8% reach distant organs, making
                treatment more difficult.
              </p>
            </motion.div>
          </div>
        </div>
        {/* sadddddddddddddddddddddddddddddddd*/}
        {/* Separate Section for Three Navigation Cards */}
        <div className="mt-12 p-4 ">
          <h2
            className="text-2xl font-bold mb-6 text-center"
            data-aos="fade-up"
          >
            Explore More
          </h2>

          <div className="flex flex-col md:flex-row items-center justify-between gap-6 bg-white ">
            {/* Dashboard Card */}
            <Link href="/dashboard" className="w-full md:w-1/3">
              <motion.div
                className="p-6 bg-white rounded-lg shadow-md text-center cursor-pointer  "
                data-aos="fade-right"
                whileHover={{ boxShadow: "0px 10px 30px rgba(0, 0, 0, 0.2)" }}
                transition={{ duration: 0.2 }}
              >
                <FaChartBar className="text-5xl text-blue-600 mb-3" />
                <h2 className="text-lg font-semibold">Dashboard</h2>
                <p className="text-sm text-gray-600 mt-2">
                  View and analyze gene expression trends.
                </p>
              </motion.div>
            </Link>

            {/* Major Findings Card */}
            <Link href="/major-finding" className="w-full md:w-1/3">
              <motion.div
                className="p-6 bg-white rounded-lg shadow-md text-center cursor-pointer "
                data-aos="fade-up"
                whileHover={{ boxShadow: "0px 10px 30px rgba(0, 0, 0, 0.2)" }}
                transition={{ duration: 0.2 }}
              >
                <FaSearch className="text-5xl text-green-600 mb-3" />
                <h2 className="text-lg font-semibold">Major Findings</h2>
                <p className="text-sm text-gray-600 mt-2">
                  Discover key insights from our research.
                </p>
              </motion.div>
            </Link>

            {/* Project Objective Card */}
            <Link href="/project-objective" className="w-full md:w-1/3">
              <motion.div
                className="p-6 bg-white rounded-lg shadow-md text-center cursor-pointer"
                data-aos="fade-left"
                whileHover={{ boxShadow: "0px 10px 30px rgba(0, 0, 0, 0.2)" }}
                transition={{ duration: 0.2 }}
              >
                <FaClipboardList className="text-5xl text-red-600 mb-3" />
                <h2 className="text-lg font-semibold">Project Objective</h2>
                <p className="text-sm text-gray-600 mt-2">
                  Learn about the goals and scope of our study.
                </p>
              </motion.div>
            </Link>
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
