export default function Home() {
  return (
    <main className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-white p-6">
      <div className="max-w-4xl text-center">
        <h1 className="text-4xl font-bold text-pink-500 mb-4">
          Breast Cancer Classification
        </h1>
        <p className="text-lg text-gray-300">
          Understanding breast cancer through machine learning and gene expression analysis.
        </p>

        {/* Cards Section */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-8">
          {/* Card 1: About Breast Cancer */}
          <div className="bg-gray-800 p-6 rounded-lg shadow-lg hover:shadow-pink-500 transition">
            <h2 className="text-2xl font-semibold text-pink-400">What is Breast Cancer?</h2>
            <p className="mt-2 text-gray-300">
              Breast cancer is a disease in which cells in the breast grow uncontrollably. 
              Early detection through advanced techniques can help in better treatment.
            </p>
          </div>

          {/* Card 2: Machine Learning Role */}
          <div className="bg-gray-800 p-6 rounded-lg shadow-lg hover:shadow-pink-500 transition">
            <h2 className="text-2xl font-semibold text-pink-400">Machine Learning in Detection</h2>
            <p className="mt-2 text-gray-300">
              ML models analyze gene expression and histopathology images to classify benign and malignant cases.
            </p>
          </div>

          {/* Card 3: How the Model Works */}
          <div className="bg-gray-800 p-6 rounded-lg shadow-lg hover:shadow-pink-500 transition">
            <h2 className="text-2xl font-semibold text-pink-400">How It Works</h2>
            <p className="mt-2 text-gray-300">
              Our AI model takes in medical imaging or gene expression data, processes it, and provides a classification result.
            </p>
          </div>

          {/* Card 4: Importance of Early Detection */}
          <div className="bg-gray-800 p-6 rounded-lg shadow-lg hover:shadow-pink-500 transition">
            <h2 className="text-2xl font-semibold text-pink-400">Early Detection Saves Lives</h2>
            <p className="mt-2 text-gray-300">
              Identifying breast cancer at an early stage significantly increases treatment success rates.
            </p>
          </div>
        </div>

        {/* Call to Action */}
        <div className="mt-8">
          <a
            href="#"
            className="bg-pink-500 hover:bg-pink-600 text-white font-semibold py-3 px-6 rounded-lg transition"
          >
            Learn More
          </a>
        </div>
      </div>
    </main>
  );
}
