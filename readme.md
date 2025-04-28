# GeneScope

GeneScope is a full-stack platform for analyzing gene expression patterns and clinical data to predict breast cancer stages, highlight key biomarkers, and support prognosis insights.  
It combines powerful machine learning models, a clean web dashboard, and an intelligent chatbot to make complex biological data understandable and actionable.

## Project Structure

```
GENESCOPE
├── BackEnd/            # Flask server, APIs, models, backend logic
│   ├── Marimo_server/
│   ├── Models/
│   ├── routers/
│   └── GeneScopeServer.py
├── data/               # Gene expression and clinical datasets
├── front/              # Next.js frontend dashboard and chatbot UI
└── project_insites/    # Research notes and project documentation

```

## Features

- Breast Cancer Stage Prediction  
  Predicts cancer stages based on combined gene expression and clinical features.

- Gene Importance Analysis  
  Identifies critical genes using a modified Analytic Hierarchy Process (AHP) and statistical methods.

- Interactive Chatbot  
  Conversational assistant connected to the staging model for real-time, data-backed answers.

- Data Visualization  
  Dynamic charts and graphs to explore trends and findings.

- Full-stack System  
  Backend APIs with Flask and machine learning, and a modern dashboard built in Next.js.

## Getting Started

### Backend (Flask)

```
cd BackEnd
python GeneScopeServer.py
```

### Frontend (Next.js)

#### Environment Variables

Before running the project, create a `.env` file inside the `/front` directory with the following variables:

```env
NEXT_PUBLIC_FIREBASE_API_KEY=your-firebase-api-key
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=your-firebase-auth-domain
NEXT_PUBLIC_FIREBASE_PROJECT_ID=your-firebase-project-id
NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=your-firebase-storage-bucket
NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=your-firebase-messaging-sender-id
NEXT_PUBLIC_FIREBASE_APP_ID=your-firebase-app-id
NEXT_PUBLIC_FIREBASE_MEASUREMENT_ID=your-firebase-measurement-id
NEXT_PUBLIC_STUFF=your-openai-or-other-api-key
OPENAI_API_KEY=your-openai-api-key
PINECONE_API_KEY=your-pinecone-api-key
```

---

Once the `.env` file is set up, install the dependencies and run the development server:

```bash
cd front
npm install
npm run dev
```

The frontend will be available at [http://localhost:3000](http://localhost:3000).

---

## Technologies Used

- Frontend: Next.js, React, TailwindCSS, Chart.js
- Backend: Flask, TensorFlow/Keras, Pandas, Scikit-learn
- Other: Docker-ready for cloud deployment (GCP)

## Acknowledgements

- TCGA (The Cancer Genome Atlas) for public gene expression and clinical datasets
- Open-source libraries and research in biomedical machine learning

---

## Future Enhancements

- Enhanced chatbot capabilities (summarizing research papers)
- Expanded dataset support for multi-cancer analysis
- Cloud-based deployment with automated scaling

---
