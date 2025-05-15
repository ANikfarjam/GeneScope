<p align="center">
  <img src="project_insites/readme_intro.png" alt="Gene Intro" width="800" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Built%20With-Python-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Frontend-TypeScript-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Framework-Next.js-black?logo=next.js&style=for-the-badge" />
  <img src="https://img.shields.io/badge/UI-Marimo-blueviolet?style=for-the-badge" />
  <img src="https://img.shields.io/badge/LangChain-Enabled-success?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Pinecone-Integrated-lightgrey?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Maintained-Yes-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Website-Up-blue?style=for-the-badge" />
</p>

GeneScope is a research platform that combines biology, data science, and artificial intelligence to help better understand breast cancer. It analyzes patterns in gene activity and clinical information, like tumor size and patient age, to find important genes that may be linked to cancer development. Using a special method called the Analytic Hierarchy Process, GeneScope ranks thousands of genes to spotlight the ones that matter most. It also uses machine learning models to predict how likely a patient is to be diagnosed at different stages of breast cancer. By connecting genetics with clinical factors, GeneScope aims to make cancer detection earlier, smarter, and more personalized, offering new insights that could improve future treatments and patient outcomes.

## Project Structure

```
GENESCOPE
├── BackEnd/            # Flask server, APIs, models, backend logic
│   ├── Marimo_server/
|   ├── computation_anlysis/
|   ├── Docker Deployement/
│   ├── Models/
│   ├── routers/
│   └── GeneScopeServer.py
├── data/               # Gene expression and clinical datasets
├── front/              # Next.js frontend dashboard and chatbot UI
└── project_insites/    # Research notes and project documentation
```

## Feature Table


| Feature                   | Description                                                                                       |
|---------------------------|---------------------------------------------------------------------------------------------------|
| **Breast Cancer Stage Prediction** | Predicts cancer stages based on combined gene expression and clinical features.               |
| **Gene Importance Analysis**      | Identifies critical genes using a modified Analytic Hierarchy Process (AHP) and statistical methods. As well as identifying most active miRNA involved in progression of breast cancer. |
|**Cancer Prognosis Analysis**| Analyze how features such as tumore sieze, number of regional lymph nodes and dinstance that tumor have grown utilizing Cox Hazerdus Model. |
| **Interactive Chatbot**           | Conversational assistant connected to the staging model for real-time, data-backed answers.   |
| **Data Visualization**            | Dynamic charts and graphs to explore trends and findings.                                    |
| **Full-stack System**             | Backend APIs with Flask and machine learning, and a website built in Next.js.        |


## Getting Started

### Backend (Flask)

```bash
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

## Technologies Used

- Frontend: Next.js, React, TailwindCSS, Chart.js, OpenAI API, langchain
- Backend: Flask, Marimo, TensorFlow/Keras, Pandas, Scikit-learn, Tensorflow Keras, lifelines.coxPHfitter, carboost.CatBoostClassifier, Docker for deploying marimo server to marimo server, GAN
- Other: Docker-ready for Google cloud deployment (App Engine)
## Analytic Files
BackEnd/computation_analysis
  
---/modAhpFunctions.py ----> This file contains all the fuctions for calculating ttest, entrhopy, wilcoxon and..
  
---/compahp.py ----> calls the functions to calculate each individual secores
  
---/compPWM ----> calculates PWM, Eigen Vectors, Eigen Values
  
---/com_fs ---> uses eigen values of each matrix(AHP ect) to compute final score

BackEnd/Models

---/DMM ----> for both experimental and Main MultiDNN

---/gbst ----> CatBoost Model utilized for calculating patients being diagnosed at any stage

BackEnd/Marimo_server/scripts ---> All the in-marimo analysis such as corrolation analysis, heatmap for comparing tope ranke genes through out different stages and even data collection and clean up

## Acknowledgements

- TCGA (The Cancer Genome Atlas) for public gene expression and clinical datasets
- Open-source libraries and research in biomedical machine learning

## Future Enhancements
- Facinating Multimodel Deep Neural Network optimized to classiffy based on gene-expressions and clinical data
- Enhanced chatbot capabilities (summarizing research papers)
- Expanded dataset support for multi-cancer analysis
- Cloud-based deployment with automated scaling

## Sources

NCBI GEO DataSet Documentation:

https://www.ncbi.nlm.nih.gov/geo/info/

Cancer statistics:

https://pmc.ncbi.nlm.nih.gov/articles/PMC8428369/#:~:text=Breast%20cancer%20is%20the%20most%2


NGS:
https://link.springer.com/article/10.1186/s12864-017-4011-0

Cancer Types:

https://www.cancer.org/cancer/types/breast-cancer/about/types-of-breast-cancer.html#:~:text=Invasive%20(or%20infiltrating)%20breast%20cancer%20has%20spread%20into%20surrounding%20breast,8
0%25%20of%20all%20breast%20cancers

Principal of Cancer Staging:

https://www.facs.org/media/xuxfkbpb/principles_cancer_staging.pdf

AHP Analysis was isnpired by:
https://www.sciencedirect.com/science/article/pii/S0020025515002807?casa_token=GgHMPZ4GQWMAAAAA:kl0woyWwWFWZo5nfQDRqhcvztyB103WUWoYcc733pWUa0jKs_7ypculcinAcZjxHZbA3IzGZL3Q3


MultiDeep Learning Model:

https://ieeexplore.ieee.org/abstract/document/8292801?casa_token=J6Bt__TE05sAAAAA:P7rbhfoJDDM9DSFFDfsc5kUYGSFO-9c0UPnTINRVZkvUjYe9EUufViBbfkZ23pJ_XL2SQMHG1dfN&fbclid=IwY2xjawKSZuRleHRuA2FlbQIxMABicmlkETFMcHNONDN2eXhqdEtNOHRKAR4icRQO8hHxUotWrd-Kpv1tiq4KoLX2pqtAijDZn4HCAk4-z862MB0nWyfr1w_aem_bKQlg5gPM8kEHaUzt9UN3w

Catboost:

https://catboost.ai/

Cox Proportional-Hazards Model:

https://www.sthda.com/english/wiki/cox-proportional-hazards-model

Generative Adversarial Network

https://en.wikipedia.org/wiki/Generative_adversarial_network


Marimo:

https://docs.marimo.io/

Non coding genes miRNA

https://link.springer.com/article/10.1186/S12935-015-0185-1


TNM Staging system:

https://acsjournals.onlinelibrary.wiley.com/doi/full/10.1002/1097-0142%2820010415%2991%3A8%2B%3C1589%3A%3AAID-CNCR1170%3E3.0.CO%3B2-K



