"use client";
import ParticleBackground from "./components/ParticleBackground";
import { useEffect } from "react";
import AOS from "aos";
import "aos/dist/aos.css";
import { Bar } from "react-chartjs-2";
import { Doughnut } from "react-chartjs-2";
import { Chart, registerables } from "chart.js";
import Typer from "./components/Typer";
import { motion } from "framer-motion";
import Link from "next/link";
import { FaRegCommentDots, FaSearch, FaChartLine } from "react-icons/fa";

Chart.register(...registerables);

export default function Home() {
  useEffect(() => {
    AOS.init({ duration: 1000, once: true });
  }, []);

  const cardData = [
    {
      title: "Machine Learning Models",
      description:
        "Learn how CatBoost and Cox Proportional Hazards models help predict cancer stages and survival risks.",
      link: "https://catboost.ai/",
    },
    {
      title: "Clinical Risk Factors",
      description:
        "Understand how patient traits like tumor size, lymph node involvement, age, and ethnicity influence cancer outcomes.",
      link: "https://www.cancer.gov/about-cancer/diagnosis-staging/prognosis",
    },
    {
      title: "miRNA Discoveries",
      description:
        "Discover how small RNA molecules (miRNAs) can act as powerful biomarkers for breast cancer progression.",
      link: "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4267179/", // Reliable NCBI article
    },

    {
      title: "Our Modified AHP Method",
      description:
        "Learn about our custom version of the Analytic Hierarchy Process for ranking important genes in breast cancer research.",
      link: "https://en.wikipedia.org/wiki/Analytic_hierarchy_process",
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
          <Typer text="GeneScope at a Glance" />
        </h1>
        <div
          className="text-lg mb-8 min-h-[190px] w-full overflow-hidden relative text-left pt-2 "
          data-aos="fade-right"
        >
          GeneScope is a research platform that combines biology, data science,
          and artificial intelligence to help better understand breast cancer.
          It analyzes patterns in gene activity and clinical information, like
          tumor size and patient age, to find important genes that may be linked
          to cancer development. Using a special method called the Analytic
          Hierarchy Process, GeneScope ranks thousands of genes to spotlight the
          ones that matter most. It also uses machine learning models to predict
          how likely a patient is to be diagnosed at different stages of breast
          cancer. By connecting genetics with clinical factors, GeneScope aims
          to make cancer detection earlier, smarter, and more personalized,
          offering new insights that could improve future treatments and patient
          outcomes.
        </div>
        <div className="mt-9 p-4 ">
          <h2
            className="text-2xl font-bold mb-6 text-center"
            data-aos="fade-up"
          >
            Our Tools
          </h2>

          <div className="flex flex-col md:flex-row items-center justify-between gap-6 bg-white ">
            {/* Dashboard Card */}
            <Link href="/dashboard" className="w-full md:w-1/3">
              <motion.div
                className="p-6 bg-white rounded-lg shadow-md text-center cursor-pointer"
                data-aos="fade-right"
                whileHover={{ boxShadow: "0px 10px 30px rgba(0, 0, 0, 0.2)" }}
                transition={{ duration: 0.2 }}
              >
                <div className="flex flex-col items-center group transition-all duration-2000 ease-in-out">
                  <FaRegCommentDots className="text-5xl text-blue-300 mb-3 group-hover:text-blue-600 transition-colors duration-2000" />
                  <h2 className="text-lg font-semibold">Chatbot</h2>
                  <p className="text-sm text-gray-600 mt-2">
                    powered by our staging model for cancer answers
                  </p>
                </div>
              </motion.div>
            </Link>
            <Link href="/summaryanalysis" className="w-full md:w-1/3">
              <motion.div
                className="p-6 bg-white rounded-lg shadow-md text-center cursor-pointer"
                data-aos="fade-left"
                whileHover={{ boxShadow: "0px 10px 30px rgba(0, 0, 0, 0.2)" }}
                transition={{ duration: 0.2 }}
              >
                <div className="flex flex-col items-center group transition-all duration-2000 ease-in-out">
                  <FaChartLine className="text-5xl text-orange-300 mb-3 group-hover:text-orange-500 transition-colors duration-2000" />
                  <h2 className="text-lg font-semibold">Summary Analysis</h2>
                  <p className="text-sm text-gray-600 mt-2">
                    Learn about the goals and scope of our study.
                  </p>
                </div>
              </motion.div>
            </Link>
            <Link href="/majorfindings" className="w-full md:w-1/3">
              <motion.div
                className="p-6 bg-white rounded-lg shadow-md text-center cursor-pointer"
                data-aos="fade-up"
                whileHover={{ boxShadow: "0px 10px 30px rgba(0, 0, 0, 0.2)" }}
                transition={{ duration: 0.2 }}
              >
                <div className="flex flex-col items-center group transition-all duration-2000 ease-in-out">
                  <FaSearch className="text-5xl text-green-300 mb-3 group-hover:text-green-600 transition-colors duration-2000" />
                  <h2 className="text-lg font-semibold">Major Findings</h2>
                  <p className="text-sm text-gray-600 mt-2">
                    Discover key insights from our research.
                  </p>
                </div>
              </motion.div>
            </Link>
          </div>
        </div>
        <h2
          className="text-2xl font-bold mb-6 text-center mt-12"
          data-aos="fade-up"
        >
          The Impact of Breast Cancer
        </h2>
        <div className=" shadow-lg p-4 inset-shadow-xs bg-white">
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
        <motion.div
          className="bg-white  shadow-md p-4 md:p-4 my-10 text-left"
          data-aos="fade-up"
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
        >
          <h2 className="text-2xl font-bold mb-4 text-black-700 text-center">
            Talk to the Science
          </h2>
          <p className="text-lg mb-4">
            GeneScope analyzes both gene expression and clinical data using
            cutting-edge deep learning models to predict cancer stages, identify
            biomarkers, and highlight patterns that may go unnoticed in
            traditional methods. Unlike traditional studies that rely only on
            basic statistics, GeneScope combines gene-level biological analysis
            with powerful machine learning, including CatBoost, deep learning,
            and a modified Analytic Hierarchy Process (AHP) for more reliable
            predictions. By examining how genes behave differently across ages,
            ethnicities, and tumor characteristics, we ensure that predictions
            are not one-size-fits-all, but personalized for every individual.
            Our models are trained to discover hidden biomarkers, assess patient
            risks, and unlock new insights into breast cancer progression.
            Through an intelligent assistant, a transparent model, and a deep
            respect for human variation, we’re helping people understand breast
            cancer on a level never seen before.
          </p>

          <p className="text-lg"></p>
        </motion.div>

        <h2
          className="text-2xl font-bold mb-6 text-center mt-8"
          data-aos="fade-up"
        >
          More Information
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-5">
          {cardData.map((card, index) => (
            <div
              key={index}
              className="bg-white p-6 rounded-lg shadow-md"
              data-aos="fade-left"
              data-aos-delay={index * 200}
            >
              <h2 className="text-xl font-semibold">{card.title}</h2>
              <p className="mt-2, text-left">{card.description}</p>
              <div className="mt-4">
                <a
                  href={card.link}
                  target="_blank"
                  className="text-blue-500 hover:underline font-semibold"
                >
                  Learn more →
                </a>
              </div>
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
