# GeneScope

<img src="https://raw.githubusercontent.com/khmorad/csvStore/refs/heads/main/genescope2.webp" alt="Mood Stabilizer" width="400" />


GeneScope is a research-driven project that classifies **breast cancer subtypes** using **gene expression profiles** and **Hidden Markov Models (HMMs)**. By leveraging advanced statistical methods and machine learning techniques, GeneScope aims to improve early detection and classification of breast cancer, helping guide more effective treatment strategies.

## Project Overview

Breast cancer is one of the most frequently diagnosed cancers worldwide. Early and accurate classification of breast cancer subtypes is critical as different types require specific treatment approaches. GeneScope integrates **gene selection techniques** with **Hidden Markov Models (HMMs)** to identify key genetic markers for classification.

### Methodology

GeneScope employs a multi-step process:
1. **Gene Selection**: Identifying the most informative genes using:
   - t-tests
   - Entropy analysis
   - Receiver Operating Characteristic (ROC) curves
   - Wilcoxon tests
   - Signal-to-noise ratio
2. **Hidden Markov Model (HMM) Classification**:
   - Training individual HMMs for different breast cancer subtypes.
   - Using probabilistic modeling to classify new patient samples.
3. **Data Collection & Processing**:
   - Utilizing gene expression datasets from the **NCBI GEO Database**.
   - Preprocessing data for optimal feature extraction and model training.
4. **Model Training & Evaluation**:
   - Implementing an optimized **HMM-based classifier**.
   - Evaluating performance using cross-validation and statistical metrics.

## Why Gene Expression Matters in Cancer Classification

Every cell in the human body contains the same DNA, but different genes are **expressed (activated)** based on cell type and function. By analyzing gene expression patterns, we can distinguish between normal and cancerous cells and even differentiate between cancer subtypes.

### Gene Selection Techniques Used
- **t-test**: Identifies genes with significant expression differences.
- **Entropy Test**: Measures disorder in gene expression levels.
- **ROC Analysis**: Evaluates genes with strong classification potential.
- **Wilcoxon Test**: Ranks genes based on median expression differences.
- **Signal-to-Noise Ratio (SNR)**: Measures how well a gene distinguishes between classes.

## Breast Cancer Subtypes & Classification

| Subtype | Description |
|---------|-------------|
| **Ductal Carcinoma In Situ (DCIS)** | Non-invasive; abnormal cells remain in milk ducts. |
| **Invasive Ductal Carcinoma (IDC)** | Cancer spreads beyond ducts into surrounding tissue. |
| **Invasive Lobular Carcinoma (ILC)** | Starts in milk-producing lobules and spreads further. |
| **Triple-Negative Breast Cancer** | Lacks estrogen, progesterone, and HER2 receptors, making treatment more challenging. |
| **HER2-Positive Breast Cancer** | Cancer cells overexpress HER2 protein, often more aggressive but treatable with targeted therapy. |

## Technologies Used

- **Front End**: Next.js
- **Machine Learning**: Hidden Markov Models (HMMs)
- **Data Processing**: Python, NumPy, Pandas, Scikit-learn
- **Bioinformatics**: NCBI GEO Database
- **Backend**: Flask, FastAPI (for potential web-based visualization)
- **Cloud Hosting**: AWS (for scalable deployment)

## Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/genescope.git
   ```
2. Navigate to the project directory:
   ```bash
   cd genescope
   ```
3. to be determined later....

## rough Sprint Plan 

| Week Ending | Task |
|-------------|------|
| **Feb 16, 2025** | Data selection and preprocessing |
| **Feb 23, 2025** | Feature extraction and gene ranking implementation |
| **Mar 2, 2025** | Initial HMM model training and evaluation |
| **Mar 9, 2025** | Refine model performance and hyperparameter tuning |
| **Mar 16, 2025** | Implement API endpoints for data retrieval |
| **Mar 23, 2025** | Develop frontend visualization and user interface |
| **Mar 30, 2025** | Integrate frontend with backend services |
| **Apr 6, 2025** | Perform full model validation and testing |
| **Apr 13, 2025** | Optimize API response times and finalize documentation |
| **Apr 20, 2025** | Deploy project on cloud and prepare for final presentation |
| **Apr 27, 2025** | Conduct final tests and improve UI/UX based on feedback |
| **May 4, 2025** | Wrap up project and submit final version |

## Team Members

- **Yar Moradpour**  
  [GitHub](https://github.com/khmorad)  
  [LinkedIn](https://linkedin.com/in/kmoradpour)  

- **[New Team Member 1]**  
  [GitHub](#)  
  [LinkedIn](#)  

- **[New Team Member 2]**  
  [GitHub](#)  
  [LinkedIn](#)  

## References
- [NCBI GEO Database](https://www.ncbi.nlm.nih.gov/geo/)
- [Hidden Markov Models in Cancer Research](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4478970/)
- [Breast Cancer Statistics 2024](https://www.cancer.org/)

## Contact
For questions or collaboration opportunities, reach out via:
- **Email**: khakho.morad@gmail.com
