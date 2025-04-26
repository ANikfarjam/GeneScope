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
import { FaSearch, FaClipboardList, FaComment } from "react-icons/fa";
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
          GeneScope at a Glance
        </h1>
        <div
          className="text-lg h-60 w-full overflow-hidden relative text-left pt-2 mar mb-3"
          data-aos="fade-right"
        >
          <Typer
            text="We’re on a mission to make breast cancer analysis smarter, more personal, and easier to understand. GeneScope combines powerful deep learning with real clinical and genetic data to predict cancer stages, spotlight key biomarkers, and uncover hidden patterns in the fight against breast cancer. But here’s the best part you can actually talk to it!
            Our chatbot isn’t just conversational it performs staging predictions too, giving you fast, data-backed answers. Dive into our major findings to see what we’ve uncovered, and learn about our project objectives to understand the bigger picture."
          />
        </div>
        <div className="mt-12 p-4 ">
          <h1 className="text-4xl font-bold mb-4" data-aos="fade-right">
            Our tools{" "}
          </h1>
          <div className="flex flex-col md:flex-row items-center justify-between gap-6 bg-white">
            <Link href="/dashboard" className="w-full md:w-1/3">
              <motion.div
                className="p-6 bg-white rounded-lg shadow-md text-center cursor-pointer"
                data-aos="fade-right"
                whileHover={{ boxShadow: "0px 10px 30px rgba(0, 0, 0, 0.2)" }}
                transition={{ duration: 0.2 }}
              >
                <FaComment className="text-5xl text-blue-300 mb-3 mx-auto" />
                <h2 className="text-lg font-semibold">chatbot</h2>
                <p className="text-sm text-gray-600 mt-2">
                  powered by our staging model for fast, data-backed cancer
                  answers
                </p>
              </motion.div>
            </Link>

            {/* Major Findings Card */}
            <Link href="/majorfindings" className="w-full md:w-1/3">
              <motion.div
                className="p-6 bg-white rounded-lg shadow-md text-center cursor-pointer"
                data-aos="fade-up"
                whileHover={{ boxShadow: "0px 10px 30px rgba(0, 0, 0, 0.2)" }}
                transition={{ duration: 0.2 }}
              >
                <FaSearch className="text-5xl text-green-600 mb-3 mx-auto" />
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
                <FaClipboardList className="text-5xl text-red-600 mb-3 mx-auto" />
                <h2 className="text-lg font-semibold">Project Objective</h2>
                <p className="text-sm text-gray-600 mt-2">
                  Learn about the goals and scope of our study.
                </p>
              </motion.div>
            </Link>
          </div>
        </div>
        <div className="mt-12 shadow-md p-4 inset-shadow-xs bg-white">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            <a
              href="https://breast-cancer.ca/diag-ratngs/"
              target="_blank"
              rel="noopener noreferrer"
              className="w-full md:w-1/2"
            >
              <motion.div
                className="p-2"
                data-aos="fade-right"
                whileHover={{ boxShadow: "0px 10px 30px rgba(0, 0, 0, 0.2)" }}
                transition={{ duration: 0.2 }}
              >
                <Bar data={casualtiesData} options={{ responsive: true }} />
              </motion.div>
            </a>

            <motion.div
              className="w-full md:w-1/2 text-left p-2"
              data-aos="fade-left"
              whileHover={{ boxShadow: "0px 10px 30px rgba(0, 0, 0, 0.2)" }}
              transition={{ duration: 0.2 }}
            >
              <p className="text-lg">
                Breast cancer remains one of the most commonly diagnosed cancers
                worldwide, with the majority of cases found in individuals aged
                50 to 69. While younger cases are less frequent, they often
                present more aggressively making early detection critical at
                every age. Yet, nearly 1 in 3 cases are still diagnosed only
                after the cancer has begun to spread. This underscores the
                urgent need for better screening tools, smarter diagnostics, and
                more personalized insight. That’s where GeneScope steps in.
              </p>
            </motion.div>
          </div>
        </div>
        <div className="mt-12 shadow-md p-4 inset-shadow-xs bg-white">
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

        <motion.div
          className="bg-white shadow-md p-6 md:p-8 my-10 text-left"
          data-aos="fade-up"
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
        >
          <h2 className="text-2xl font-bold mb-6 text-center ">
            Talk to the Science
          </h2>
          <p className="text-lg mb-4">
            This project analyzes both gene expression and clinical data using
            cutting-edge deep learning models to predict cancer stages, identify
            biomarkers, and highlight patterns that may go unnoticed in
            traditional methods. GeneScope isn&apos;t just about machine
            learning. It&apos;s about making that learning accessible,
            explainable, and actionable. Through an intelligent assistant, a
            transparent model, and a deep respect for human variation,
            we&apos;re helping people understand breast cancer on a level never
            seen before. Whether you&apos;re a researcher, a doctor, or just
            curious, GeneScope invites you to ask questions and get answers
            powered by real science. With our intelligent chatbot, built using
            LangChain and OpenAI, we’re turning complex data into accessible
            conversations so that everyone can better understand, explore, and
            take action in the world of cancer care.
          </p>

          <p className="text-lg"></p>
        </motion.div>

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
