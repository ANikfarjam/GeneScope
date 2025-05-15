import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import pandas as pd
    import plotly.express as px
    import numpy as np
    from sklearn.metrics import roc_curve, auc
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pickle as pkl
    return go, make_subplots, mo, pd, px


@app.cell(hide_code=True)
def _():
    import plotly.io as pio
    pio.renderers.default = "iframe_connected"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #<span style="color:brown">Breast Cancer Classification usibg Deep Learning Neural Network</span>

     **GeneScope** leverages a deep learning framework to classify breast cancer stages by integrating both clinical and gene expression data. Our dataset comprises over **2,100 patient samples**, each containing structured clinical information—such as age, tumor characteristics (TNM staging), vital status—as well as high-dimensional gene expression profiles.


    ### <span style="color:brown">Intro to he Multimodal Deep Neural Network for Multi-dimensional Data </span>

    The Multimodal Deep Neural Network for Multi-dimensional Data (MDNNMD) is a neural architecture designed to integrate and learn from multiple heterogeneous data sources simultaneously. Instead of feeding all features into a single model, MDNNMD builds independent deep neural networks for each data modality—such as gene expression, clinical recordsEach subnetwork independently learns optimal feature representations, which are then merged using a fusion mechanism to produce the final prediction.

     [Dongdong Sun etal](https://ieeexplore.ieee.org/abstract/document/8292801?casa_token=J6Bt__TE05sAAAAA:P7rbhfoJDDM9DSFFDfsc5kUYGSFO-9c0UPnTINRVZkvUjYe9EUufViBbfkZ23pJ_XL2SQMHG1dfN
    ) pulished an article utilizing this modol on cancer prognosis study. They emphasize the modular and flexible architecture enables MDNNMD to capture complementary information across different data types, allows enrichment of features. Minority class samples may have weak signals in one modality (e.g., clinical), but strong signals in another (e.g., gene expressions). This helps the model see them better.

    ##<span style="color:brown">Common Techinique used for Optimizing and connecting Multimodel Nerworks</span>
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    def multimodal_learning_columns():
        return mo.md("""
        <style>
          .column-container {
            display: flex;
            gap: 20px;
            overflow-x: auto;
            padding: 20px;
            background: #eeeeee;
          }

          .column {
            flex: 0 0 300px;
            height: 460px;
            overflow-y: auto;
            padding: 16px;
            background: #4b3a4d;
            border-radius: 14px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            color: #f2e6ff;
            border: 1px solid #7d5f85;
            transition: transform 0.3s ease;
          }

          .column:hover {
            transform: translateY(-4px);
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.25);
          }

          .column h3 {
            margin-top: 0;
            color: #d6b1ff;
          }

          .column h4 {
            margin-bottom: 4px;
            color: #eac9ff;
            font-size: 1em;
            margin-top: 14px;
          }

          .column p {
            font-size: 0.95em;
            line-height: 1.5;
            color: #e3d6ec;
          }

          .column-container::-webkit-scrollbar {
            height: 8px;
          }

          .column::-webkit-scrollbar {
            width: 8px;
          }

          .column::-webkit-scrollbar-thumb {
            background: #9b7fa2;
            border-radius: 4px;
          }
        </style>

        <div class="column-container">
          <div class="column">
            <h3>🔗 Fusion</h3>
            <p>Fusion refers to combining information from different data modalities—like gene expression and clinical variables—to make a unified prediction. It’s a key strategy in multimodal deep learning, and comes in three main types:</p>

            <h4>Early Fusion</h4>
            <p>All raw features from each modality are concatenated before being input into a single model. This works well if modalities are aligned and comparable in scale, but may suffer from noise or high dimensionality.</p>

            <h4>Intermediate Fusion</h4>
            <p>Each modality is processed by its own neural network, and their intermediate feature representations are combined—usually through concatenation or attention—before passing through shared layers for prediction. This balances independence and interaction.</p>

            <h4>Late Fusion</h4>
            <p>Each modality has a fully independent network, and predictions from each are aggregated (e.g., averaged, weighted sum, or voted) at the decision level. This approach is modular and robust, ideal when modalities are heterogeneous or separately trained.</p>
          </div>

          <div class="column">
            <h3>🧭 Alignment</h3>
            <p>Alignment ensures features from multiple modalities refer to the same subject. For example, a sample’s gene profile must match its clinical record. Techniques like normalization, timestamp syncing, and shared encoding layers help maintain consistent correspondence across views.</p>
          </div>

          <div class="column">
            <h3>🔄 Translation</h3>
            <p>Translation models learn to predict or reconstruct one modality from another, such as estimating gene expression from clinical data. This is useful for imputing missing data, or for discovering latent cross-modal relationships that help generalization.</p>
          </div>

          <div class="column">
            <h3>🤝 Co-Learning</h3>
            <p>Co-learning enables two or more modalities to learn interactively. Rather than training them separately, shared gradients or intermediate feedback are used so that progress in one modality improves representations in the other. This increases synergy and model robustness.</p>
          </div>
        """)

    multimodal_learning_columns()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # <span style="color: brown">Feature Selection for our ML Models</span>
    ##<span style="color: brown">Clinical Data</span>
    ### <span style="color: brown">Utilizing Random Forest</span>
    Our clinical dataset contained over **200 variables** spanning patient demographics, tumor characteristics, and treatment metadata. To streamline the model and reduce noise, we applied a **Random Forest Classifier** to assess feature importance and isolate the most predictive attributes.

    Using feature importance scores generated by the model, we narrowed the input space down to a focused set of approximately **40–50 high-impact clinical features**. These included fields such as tumor staging (AJCC), age at diagnosis, race and other patient demographic data.

    ### <span style="color: brown">Summary of the Selection Process</span>

    **Data Preparation**: Encoded categorical features and filled missing values using column means.

    **Train-Test Split**: Divided the dataset into training and test sets (80/20 split).

    **Random Forest Training**: Trained a baseline model using default parameters to evaluate feature significance.

    **Feature Ranking**: Sorted features based on their importance scores.

    **Selection**: Retained the top ~50 features for downstream modeling and hyperparameter tuning.

    This process not only improved model interpretability but also helped combat overfitting by removing redundant or low-impact variables. It laid a strong foundation for our later multimodal fusion models.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, pd):
    model_summery = mo.md(f"""
    | Class        |     Precision     |     Recall     |     F1-Score     |     Support     |
    |--------------|-------------------|----------------|------------------|-----------------|
    | Stage I | 0.91 | 1.00 | 0.95 | 21 |
    | Stage IA | 1.00 | 0.93 | 0.96 | 29 |
    | Stage IB | 1.00 | 1.00 | 1.00 | 1 |
    | Stage II | 1.00 | 1.00 | 1.00 | 1 |
    | Stage IIA | 1.00 | 0.99 | 1.00 | 106 |
    | Stage IIB | 0.99 | 1.00  | 0.99| 70 |
    | Stage IIIA | 1.00 | 1.00 | 1.00| 38 |
    | Stage IIIB | 0.88 | 1.00 | 0.93| 7  |
    | Stage IIIC | 1.00 | 1.00 | 1.00| 16 |
    | Stage IV | 1.00 | 0.50 | 0.67 | 2 |
    | **Accuracy** | | | **0.99** | **291** |
    | **Macro Avg** | 0.98 | 0.94 | 0.95 | 291 |
    | **Weighted Avg** | 0.99 | 0.99 | 0.99 | 291 |
    """)
    feature_analysis = mo.md(f"""
    ### <span style="color: brown">Feature Analysis of Clinical Data</span>
    With the model trained, we wanted to see which features actually mattered. Random Forests make this easy since they provide feature importance scores out of the box. These scores show how much each variable contributed to reducing uncertainty in the model’s predictions. We sorted the features by importance and created a bar chart of the top 15.
    """)
    feature_ranking = pd.read_csv('./scripts/rd.csv')
    mo.ui.tabs({
        'Model Summery':mo.hstack([model_summery, mo.vstack([feature_analysis,feature_ranking])]),
        'Code':mo.ui.code_editor(f"""
    import pandas as pd
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix

    # Load dataset
    stagedata = pd.read_csv('./AHPresults/fina_Stage_unaugmented.csv')

    # Preprocessing
    clinical_df = stagedata.copy().iloc[:, :-2000]
    clinical_df.dropna(inplace=True)
    target = 'Stage'
    features = clinical_df.columns[1:]

    # Label encode categorical features
    le = LabelEncoder()
    for col in features:
        if clinical_df[col].dtype == 'object':
            clinical_df[col] = le.fit_transform(clinical_df[col].astype(str))

    X = clinical_df[features].copy()
    y = clinical_df[target]
    X.fillna(X.mean(), inplace=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Predict
    y_pred = rf.predict(X_test)

    # Classification Report
    report = classification_report(y_test, y_pred, zero_division=1)
    print(report)
    """)
    })
    return


@app.cell
def _(mo):
    mo.image('./img/top15.png')
    return


@app.cell(hide_code=True)
def _(mo):
    conclussion = f"""After identifying the key features, we narrowed down our list to those that had the most impact on the model. We stored them in a list called features_important. This included clinical observations, demographic info, and a few identifiers that surprisingly had predictive power. Using only these features, we retrained the model — but this time, we wanted to do it right.

    Now, instead of using the default settings, we ran a GridSearchCV to find the best combination of hyperparameters. We defined a grid that tested different numbers of trees (n_estimators), maximum depths, split criteria, and whether to use bootstrapping. The grid search ran 5-fold cross-validation on each parameter combination to get the most reliable estimate of performance.

    This process took longer to run, but it paid off. We ended up with a model that was better tuned to the dataset and less prone to overfitting. Grid search also gave us valuable insight into which settings worked best for this particular problem — whether deeper trees helped or hurt the results, and whether using more estimators led to meaningful improvements.


    The best-performing model used the following parameters:

    - nestima→rs=100n_estimators = 100
    - minmin_samples_split = 5
    - min_samples_leaf = 1min_samples_leaf = 1
    - bootstrap = Falsebootstrap = False"""
    mo.ui.tabs({
        'Conclusion':conclussion,
        'Code':mo.ui.code_editor("""
    features_important = ['paper_pathologic_stage', 'ajcc_pathologic_n', 'ajcc_pathologic_t', 'days_to_collection', 
                'paper_days_to_last_followup', 'year_of_diagnosis', 'treatments', 'initial_weight', 'sample_id', 
                'days_to_birth', 'pathology_report_uuid', 'demographic_id', 'paper_days_to_birth', 'diagnosis_id', 
                'Unnamed: 0', 'age_at_diagnosis', 'sample_submitter_id', 'paper_patient', 'patient', 
                'paper_age_at_initial_pathologic_diagnosis', 'bcr_patient_barcode', 'barcode', 'submitter_id', 
                'sample', 'age_at_index', 'sites_of_involvement', 'paper_PARADIGM.Clusters', 'paper_Mutation.Clusters', 
                'primary_diagnosis', 'paper_CNV.Clusters', 'paper_BRCA_Subtype_PAM50', 'morphology', 'ajcc_pathologic_m', 
                'method_of_diagnosis', 'paper_miRNA.Clusters', 'paper_mRNA.Clusters', 'race', 'paper_DNA.Methylation.Clusters', 
                'laterality', 'ethnicity', 'preservation_method', 'paper_Included_in_previous_marker_papers', 
                'oct_embedded', 'paper_vital_status', 'vital_status', 'prior_malignancy', 'synchronous_malignancy', 
                'age_is_obfuscated', 'prior_treatment', 'tissue_or_organ_of_origin', 'icd_10_code']

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print("best parameters:", grid_search.best_params_)
    print("best score:", grid_search.best_score_)

    y_pred_grid = grid_search.best_estimator_.predict(X_test)
    print(classification_report(y_test, y_pred_grid, zero_division=1))
    """)
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ##<span style="color: brown">Gene Expression</span>
    ### <span style="color: brown">Utilizing AHP</span>

    The Analytic Hierarchy Process (AHP) is a structured decision-making approach that helps prioritize and select the most important criteria in complex problems. It works by breaking a problem into a hierarchy of criteria and sub-criteria, assigning numerical values to their relative importance, and using pairwise comparisons to generate a weighted ranking. Traditional AHP is often qualitative, relying on expert judgment, but in bioinformatics, a modified AHP can integrate statistical methods to enhance objectivity.


    In the context of biomarker analysis for BRCA-related research, AHP can significantly improve the selection of key biomarkers by aggregating multiple statistical gene selection methods. Instead of relying on a single metric, such as a t-test or entropy, the modified AHP integrates multiple ranking criteria (e.g., Wilcoxon test, ROC curves, signal-to-noise ratio) to create a more stable and reliable subset of genes. This method ensures that the chosen biomarkers are not only statistically significant but also robust across different datasets and ranking approaches. By identifying the most influential genes systematically, AHP helps refine the list of biomarkers that could be further analyzed for their role in breast cancer progression, prognosis, or response to treatment.

    <span style="color: brown">Modified AHP:</span>

    * <span style="color: brown">Two-Sample t-Test:</span>

    Purpose: Identifies statistically significant differences in gene expression between two groups (e.g., cancerous vs. healthy cells).

    Method: Compares the means of two independent samples using the t-statistic.

    Output: A t-score and p-value. A small p-value indicates significant differences in expression.

    * <span style="color: brown">Entropy Test:</span>

    Purpose: Measures the disorder in gene expression levels.

    Method: Computes entropy using histogram-based probability distributions.

    Output: Higher entropy values indicate genes with more variability, which are more useful for classification.

    * <span style="color: brown">Wilcoxon Rank-Sum Test:</span>

    Purpose: A non-parametric test used to rank genes based on their median expression differences.

    Method: Compares the ranks of two independent samples instead of their means.

    Output: A Wilcoxon statistic and a p-value. A low p-value suggests significant differences in gene ranks.

    * <span style="color: brown">Signal-to-Noise Ratio (SNR):</span>

    Purpose: Compares the difference in mean expression levels relative to the standard deviation.

    Method: SNR is calculated as the difference between the means of two groups divided by the sum of their standard deviations.

    Output: A higher SNR suggests that the gene has a strong discriminatory power between groups.

    * <span style="color: brown">AHP Weighted Ranking:</span>

    Purpose: Integrates statistical measures into a single weighted ranking system to prioritize significant genes.

    Method: Normalizes scores across all statistical tests and applies predefined weights.

    Output: A final ranking score indicating the importance of each gene in classification.

    * <span style="color: brown"> Eigenvalues and Eigenvectors in Gene Selection (Modified AHP): </span>

    The modified Analytic Hierarchy Process (AHP) used for gene selection involves constructing a pairwise comparison matrix where genes are ranked based on multiple statistical criteria (e.g., t-test, entropy, ROC, Wilcoxon, and SNR).

    The matrix is required to be consistent, meaning that its elements must satisfy certain transitivity properties.
    Eigenvectors are computed from this matrix to obtain ranking scores of genes. These eigenvectors correspond to the principal components that define the most discriminative genes.

    The largest eigenvalue (𝜆𝑚𝑎𝑥) is used to compute the Consistency Index (CI) and Consistency Ratio (CR) to ensure that the ranking process is mathematically sound.
    Eigenvalues and Eigenvectors in HMMs:

    HMMs use transition probability matrices to model state changes in gene expression related to cancer progression.
    The transition probability matrix (A) is a stochastic matrix that describes the likelihood of transitioning from one state to another.

    The stationary distribution of states (long-term probabilities of each state) is found by computing the dominant eigenvector (associated with eigenvalue 1) of this matrix.

    Spectral analysis (using eigenvalues) helps determine the stability and convergence properties of the HMM.
    Why This Matters for Cancer Classification

    Gene Selection (AHP with Eigenvectors) ensures that the most informative genes are chosen based on multiple criteria, improving classification accuracy.

    HMMs with Eigenvalues provide a probabilistic framework to model cancer progression and classify gene expression data efficiently.

    Eigenvectors define important features, helping reduce computational complexity and improving stability.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### <span style="color: brown">Pairwise Comparison Matrix and Eigenvalues in Modified AHP</span>

    The article describes the calculation of the **pairwise comparison matrix** and the **eigenvalues and eigenvectors** in the context of the **modified Analytic Hierarchy Process (AHP)** used for gene selection.

    ## **1. Construction of the Pairwise Comparison Matrix**
    The pairwise comparison matrix $X = (x_{ij})$ is an $n \times n$ matrix, where each element $x_{ij}$ represents the relative importance of gene $i$ compared to gene $j$. The matrix satisfies:

    $$
    x_{ij} = \frac{1}{x_{ji}}, \quad \forall i \neq j
    $$

    $$
    x_{ii} = 1, \quad \forall i
    $$

    The elements of the matrix are computed based on **quantitative criteria**, which include:
    - **t-test**
    - **Entropy**
    - **Receiver Operating Characteristic (ROC) curve**
    - **Wilcoxon test**
    - **Signal-to-Noise Ratio (SNR)**

    The absolute difference between the statistical values of two genes $i$ and $j$ is used to compute the pairwise importance score:

    $$
    d_{ij} = |c_i - c_j|
    $$

    where $c_i$ and $c_j$ are the scores of genes $i$ and $j$ under the given criterion. The final matrix values are scaled within the range **[1,10]** using:

    $$
    c = \frac{d_{ij} - 9}{c_{\max}} + 1
    $$

    $$
    x_{ij} =
    \begin{cases} 
    c, & \text{if } c_i \geq c_j \\
    \frac{1}{c}, & \text{otherwise}
    \end{cases}
    $$

    where $c_{\max}$ is the maximum distance between genes.

    ## **2. Eigenvector Calculation (Ranking Genes)**
    Once the pairwise comparison matrix is constructed, **eigenvectors** are used to determine the ranking of genes. The eigenvector $\lambda$ is computed using:

    $$
    S_j = \sum_{i=1}^{n} x_{ij}
    $$

    $$
    \lambda_i = \frac{1}{n} \sum_{j=1}^{n} \frac{x_{ij}}{S_j}
    $$

    This **normalized eigenvector** represents the ranking of genes.

    ## **3. Eigenvalue Calculation and Consistency Check**
    The largest eigenvalue $\lambda_{\max}$ is estimated as:

    $$
    \lambda_{\max} = \frac{1}{\lambda_i} \sum_{j=1}^{n} x_{ij} \lambda_j
    $$

    The **Consistency Index (CI)** is given by:

    $$
    CI = \frac{\lambda_{\max} - n}{n-1}
    $$

    The **Consistency Ratio (CR)** is then computed using a **Random Index (RI)**:

    $$
    CR = \frac{CI}{RI}
    $$

    A **CR value ≤ 0.1** indicates an acceptable consistency level.

    ## **Summary**
    The pairwise comparison matrix is built using **quantitative ranking** instead of expert judgment
    .
    The eigenvector of the matrix determines the relative ranking of genes.

    The eigenvalue and consistency ratio ensure that the matrix is valid for decision-making.

    This method ensures an **objective** and **stable** way to rank genes, which improves classification accuracy in cancer detection using **Hidden Markov Models (HMMs)**.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(
        "./img/AHPexplnation.png",
        caption="The hierarchy of factors for gene selection by AHP.",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""##<span style="color:brown">Result Of AHP analysis and performance Analysis""")
    return


@app.cell(hide_code=True)
def _(mo, pd):
    ahp_data = pd.read_csv('./AHPresults/final_Mod_ahp_scores.csv')
    ahp_performance = pd.read_csv('./AHPresults/AHP_consistency_results.csv')
    ahp_data.sort_values(by='Scores', inplace=True)
    result_text = mo.md(f"""
    The AHP consistency evaluation demonstrated excellent performance across all assessed metrics. As shown in the results, the Consistency Index (CI) values for t_test, entropy, roc_auc, and snr metrics were extremely close to zero, indicating that the pairwise comparisons were highly consistent. Correspondingly, the Consistency Ratio (CR) values were orders of magnitude lower than the acceptable threshold of 0.1, confirming a very high level of consistency in the constructed AHP matrices. These findings validate that the AHP-based prioritization of the top genes was performed with strong internal consistency, supporting the reliability and robustness of the final gene selection used for downstream cancer classification.
    """)
    mo.ui.tabs({
        'Top Scored 500 Genes':mo.ui.table(ahp_data.head(500)),
        'AHP Consistency Results':mo.vstack([ahp_performance, result_text]),
        "Visual_Analysis": mo.hstack(
                [
                    mo.md(
                        """
                        ### <span style'"color: green"> ROC AUC vs AHP Scores </span>

                        **U-shaped curve:** AHP scores are higher for genes with ROC AUC close to 0 or 1, and lower near 0.5.

                        **Biological meaning:** Genes that strongly distinguish cancer subtypes (either positively or negatively) are prioritized. This includes both upregulated and downregulated biomarkers, suggesting AHP is sensitive to directional changes in gene activity relevant to breast cancer.

                        ### <span style'"color: green"> SNR (Signal-to-Noise Ratio) vs AHP Scores </span>

                        **Clear positive trend:** As SNR increases, so do AHP scores.

                        **Biological meaning:** Genes with strong expression differences and low intra-class variability are ranked higher. These are likely stable and robust breast cancer biomarkers, capable of separating cancerous vs. non-cancerous or different subtypes (e.g., HER2+ vs. triple-negative

                        ### <span style'"color: green"> t-test Statistic vs AHP Scores </span>

                        **Strong linear correlation:** Higher t-values lead to higher AHP scores.

                        **Biological meaning:** AHP emphasizes genes that show statistically significant differential expression between groups. This supports the discovery of potential diagnostic or prognostic genes involved in tumor behavior, estrogen receptor signaling, or aggressiveness.

                        ### <span style'"color: green">  Wilcoxon Statistic and Wilcoxon p values vs AHP Scores </span>

                        **Positive correlation**, especially at higher Wilcoxon values.

                        **Biological meaning:** Even in non-parametric comparisons, AHP is sensitive to expression shifts. This indicates robustness to outliers and validates genes that may not follow normal distribution, which is common in real patient gene expression profiles.

                        **Low p-values (~0) align with high AHP scores**, but score distribution shows banding patterns.

                        **Biological meaning:** Genes with statistically significant differences in expression across conditions (e.g., cancer subtypes or stages) are appropriately prioritized. The banding may reflect discrete thresholds or tied rankings from AHP, often seen when many genes have similar significance

                    """
                    ),
                    mo.vstack(
                        [
                            mo.image("./img/rocVahp.png"),
                            mo.image("./img/snrVahp.png"),
                            mo.image("./img/t_testVahp.png"),
                            mo.image("./img/wilcoxonVahp.png"),
                        ]
                    ),
                ]
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #<span style="color:brown">Model Exploration and Selection</span>
    ###<span style="color:brown">Experimental Design</span>
    Inspired by Sun's paper, for exploration, We began by developing a Multilayer Perceptron (MLP) for clinical variables and another MLP for gene expression data. In multimodal deep learning, especially when integrating different types of data such as gene expression (numeric, high-dimensional) and clinical variables (tabular, categorical), five main fusion strategies are commonly used. As part of our exploratory analysis, we experimented with late fusion, where the outputs of individual models are combined at a later stage to make the final prediction. However, based on our findings, we ultimately decided to pursue a different approach for the final model.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    # Define the full HTML content
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Model Evaluation</title>
        <style>
            body {
                font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 2rem;
                background-color: #f0f0f5;
            }
            .container {
                display: flex;
                flex-wrap: wrap;
                gap: 1.5rem;
                background-color: #2b183c;
                border-radius: 16px;
                padding: 2rem;
                color: #ffffff;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                max-width: 100%;
                margin: auto;
                overflow: auto;
            }
            .column {
                flex: 1;
                padding: 2rem;
                box-sizing: border-box;
            }
            .column.left {
                border-right: 1px solid #444;
            }
            h2, h3, h4 {
                color: #ff6f61;
            }
            p, li {
                font-size: 0.95rem;
                line-height: 1.6;
                color: #ddd;
            }
            .math-block {
                background-color: #eee;
                color: #333;
                font-family: monospace;
                padding: 0.5rem 1rem;
                border-radius: 6px;
                overflow-x: auto;
                margin: 0.5rem 0 1rem;
            }
            ul {
                padding-left: 1.2rem;
            }
        </style>
    </head>
    <body>

    <div class="container">
        <div class="column left">
            <h2>Weight and Bias Forward Propagation</h2>
            <p><strong>Forward computation at Layer <i>k</i></strong></p>
            <p>In gene expression and clinical MLPs, the forward computation uses:</p>
            <div class="math-block">g<sup>k</sup> = W<sup>k</sup> h<sup>k-1</sup> + b<sup>k</sup></div>
            <p>Where:</p>
            <ul>
                <li><strong>g<sup>k</sup></strong> = pre-activation output at layer <i>k</i></li>
                <li><strong>W<sup>k</sup></strong> = weight matrix of layer <i>k</i></li>
                <li><strong>h<sup>k-1</sup></strong> = output from the previous layer (or input if <i>k</i> = 1)</li>
                <li><strong>b<sup>k</sup></strong> = bias vector of layer <i>k</i></li>
            </ul>

            <h3>Fusion Scoring Layer</h3>
            <p>The combined output of each model based on a linear aggregation is calculated as:</p>
            <div class="math-block">o<sub>DNNMD</sub> = α ⋅ o<sub>DNN-Clinical</sub> + β ⋅ o<sub>DNN-Expr</sub></div>
            <div class="math-block">α + β = 1, α ≥ 0, β ≥ 0</div>
            <p>Where α and β are the weights of each model.</p>
        </div>

        <div class="column right">
            <h2>Performance Evaluation</h2>
            <p>We plot the ROC curve and compute the AUC to evaluate sensitivity and 1-specificity. Metrics used:</p>

            <h3>Multiclass Classification Metrics</h3>
            <p>(One-vs-Rest per class <i>i</i>)</p>

            <h4>1. Sensitivity (Recall), Sn<sub>i</sub></h4>
            <div class="math-block">Sn<sub>i</sub> = TP<sub>i</sub> / (TP<sub>i</sub> + FN<sub>i</sub>)</div>
            <p>How well the model correctly identifies class <i>i</i>.</p>

            <h4>2. Precision, Pre<sub>i</sub></h4>
            <div class="math-block">Pre<sub>i</sub> = TP<sub>i</sub> / (TP<sub>i</sub> + FP<sub>i</sub>)</div>
            <p>How many predictions for class <i>i</i> were correct.</p>

            <h4>3. Accuracy (Overall)</h4>
            <div class="math-block">Acc = Total correct predictions / Total samples</div>
            <p>Proportion of correctly classified samples.</p>

            <h4>4. Matthews Correlation Coefficient (MCC)</h4>
            <div class="math-block">
                MCC = (c⋅s − Σ p<sub>k</sub> t<sub>k</sub>) / √((s² − Σ p<sub>k</sub>²)(s² − Σ t<sub>k</sub>²))
            </div>

            <h4>5. Specificity, Sp<sub>i</sub></h4>
            <div class="math-block">Sp<sub>i</sub> = TN<sub>i</sub> / (TN<sub>i</sub> + FP<sub>i</sub>)</div>
        </div>
    </div>

    </body>
    </html>
    """

    performance = mo.Html(html_content)
    return (performance,)


@app.cell(hide_code=True)
def _(go, mo, pd, performance, px):
    eval_df = pd.read_csv("./DNNResults/fusion_model_evaluation.csv")

    # Filter only general metrics
    metrics_of_interest = ["Test Accuracy", "Test Precision", "Test Sensitivity", "Test Specificity", "Test MCC"]
    filtered_df = eval_df[eval_df['Metric'].isin(metrics_of_interest)]

    # Plot a bar chart
    fig = px.bar(
        filtered_df,
        x="Metric",
        y="Score",
        text_auto='.2f',
        title="Exploretory Model Evaluation Metrics",
        labels={"Score": "Metric Score"},
        width=700,
        height=500
    )
    # Create DataFrame
    validation_accuracy_df = pd.DataFrame({
        "Class (Stage)": [
            "Class 0 (Stage I)", "Class 1 (Stage IA)", "Class 2 (Stage IB)",
            "Class 3 (Stage II)", "Class 4 (Stage IIA)", "Class 5 (Stage IIB)",
            "Class 6 (Stage IIIA)", "Class 7 (Stage IIIB)", "Class 8 (Stage IIIC)",
            "Class 9 (Stage IV)"
        ],
        "Best Validation Accuracy": [
            0.8328567343, 0.917870519, 0.8293515358, 1.0,
            0.8888435235, 0.8081109096, 0.8563122924,
            0.8875862069, 0.971434617, 0.8641552511
        ]
    })

    #create a Figure and add Funnel as a trace
    val_fig = go.Figure(go.Funnel(
        y=validation_accuracy_df["Class (Stage)"],
        x=validation_accuracy_df["Best Validation Accuracy"] * 100,  # Scale to %
        textinfo="value+percent initial",
        textposition="inside",
        marker={"color": [
            "#d4edda", "#c3e6cb", "#b1dfbb", "#a0d8ab", "#90d29b",
            "#7fcb8b", "#6fc57b", "#5ebf6b", "#4db95b", "#3cb34b"
        ]}
    ))

    # Update the Figure layout
    val_fig.update_layout(
        title="Validation Accuracy Across Cancer Stages",
        funnelmode="stack",
        yaxis_title="Cancer Stage",
        xaxis_title="Best Validation Accuracy (%)",
        template="simple_white",
        width=800,
        height=600
    )
    exp_visuals = mo.vstack([
        mo.ui.plotly(fig),
        mo.ui.plotly(val_fig),
        mo.ui.table(eval_df)
    ])

    final_tabs = mo.ui.tabs(
        {
            'Performance': exp_visuals,
            'Evaluation Process': performance,
            'Model summery': mo.image('./img/fusion_model_composite.png'),
        }
    ).style({"margin": "auto", "max-width": "1200px"})

    final_tabs
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ###<span style="color:brown">Exploratory Model Evaluation Summary</span>

    The performance of our initial exploratory model reveals a significant imbalance in classification capabilities across cancer stages. While metrics like accuracy (0.73) and specificity (0.97) appear strong, the sensitivity (0.63) highlights a crucial limitation — the model struggles to correctly identify positive cancer cases.

    This sensitivity drop is a common symptom of class imbalance, where underrepresented stages skew the model's learning process. Such imbalanced distributions lead the model to favor majority classes, sacrificing its ability to detect minority class patterns.Even this model has been trained and tuned with Keras Tuner Network Architecture Search, but it seams like the unballence data is heacily effecting the performance of the model.

    Sun's Paper sujest to add Alignemt sub network layers in middle of each MLPs to handle noise and also in a high dimentional data. How ever they are working a more balenced data. Thus another approach is necessury.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # <span style="color:brown">Main Model: CatBoost + MLP + Intermediate Fusion MLP</span>

    To address the performance drop caused by class imbalance—particularly the reduced sensitivity for minority cancer stages—we introduced two architectural innovations in our main model. First, we replaced traditional classifiers for clinical data with CatBoost, a gradient boosting framework optimized for tabular datasets. CatBoost natively handles categorical features and employs built-in class weighting, helping counteract bias toward majority classes and improving detection of underrepresented stages. Second, we adopted an intermediate fusion strategy to combine clinical and gene expression modalities. Clinical data is transformed into embeddings via CatBoost, while gene expression is processed through a dedicated MLP. Their intermediate representations are then merged and passed through a joint MLP for final prediction. This method preserves the strengths of each modality while enabling effective cross-modal learning, outperforming both early and late fusion strategies in flexibility and synergy.

    The MLPs are trained and tuned utilizing Network Architecture search.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    from marimo import Html

    main_model_html = Html(r"""
    <div style="font-family: 'Segoe UI', sans-serif; background: #f9f5f0; padding: 2rem; border-radius: 16px; border: 1px solid #d4bfb5; max-width: 1000px; margin: auto; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
      <h2 style="color: brown; text-align: center;">Network Architecture Search</h2>

      <p style="font-size: 1.05rem; color: #333;">
        NAS is a subfield of automated machine learning that automates all steps in the machine learning pipeline, from data cleaning, to feature engineering and selection, to hyperparameter optimization and architecture search.
      </p>

      <div style="text-align: center; margin: 1.5rem 0;">
        <img src="https://github.com/ANikfarjam/X-ray-_mage_classification_CNNRNN/raw/main/NASdiagram.png" alt="NAS Diagram" style="max-width: 100%; border: 1px solid #ccc; border-radius: 8px;" />
        <p style="font-size: 0.9rem; color: #555;">Illustration of a Network Architecture Search process</p>
      </div>

      <p style="font-size: 1rem; color: #444;">
        To mitigate hyperparameter optimization (HPO) problems, NAS uses a search space, a search strategy, and a performance estimation strategy.
      </p>

      <h3 style="color: #8b4513;">Multimodal Integration</h3>
      <p style="font-size: 1rem; color: #444;">
        A search space is a set of all architectures that NAS can select. A search strategy is a technique used to find the optimal configuration in that space. A performance estimation strategy attempts to predict how well a candidate architecture will perform.
      </p>

      <p style="margin-top: 1.5rem; font-style: italic; color: #555;">
        NAS also trains the model, but with a unique twist: while a typical training process only adjusts the weights of a fixed network architecture, NAS actively trains both the model weights and the network architecture itself, searching for the optimal structure within a defined search space to achieve the best performance on a given task. 
      </p>

      <hr style="margin: 2rem 0; border-top: 1px dashed #aaa;" />

      <h3 style="color: #8b4513;">Mathematical Optimization Behind NAS</h3>
      <p style="font-size: 1rem; color: #444;">
        The mathematical expression for NAS can be summarized as a bilevel optimization, where the goal is to find an architecture that minimizes the validation loss, where optimal weights obtained by minimizing the training loss for that architecture.
      </p>
    </div>
    """)

    main_model_html
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""###<span style="color:brown">Model Performancs:</span>""")
    return


@app.cell(hide_code=True)
def _(make_subplots, mo, pd, px):
    # Load the data
    overall_df = pd.read_csv('./DNNResults/overall_performance_val_test.csv')
    class_df = pd.read_csv('./DNNResults/class_auc_val_test.csv')

    # Reshape overall performance for grouped bar chart
    overall_long = overall_df.melt(
        id_vars='Metric',
        value_vars=['Validation', 'Test'],
        var_name='Dataset',
        value_name='Score'
    )

    # Reshape class performance for grouped bar chart
    class_long = class_df.melt(
        id_vars='Class',
        value_vars=['Validation AUC', 'Test AUC'],
        var_name='Dataset',
        value_name='AUC'
    )

    # Create subplot container
    figs = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Overall Metric Scores", "Per-Class AUC Scores"],
        horizontal_spacing=0.15
    )

    # Plot 1: Overall metric comparison
    bar1 = px.bar(
        overall_long,
        x='Metric',
        y='Score',
        color='Dataset',
        barmode='group',
        text='Score',
        color_discrete_map={'Validation': '#1f77b4', 'Test': '#ff7f0e'}
    )

    # Plot 2: Class-wise AUCs
    bar2 = px.bar(
        class_long,
        x='Class',
        y='AUC',
        color='Dataset',
        barmode='group',
        text='AUC',
        color_discrete_map={'Validation AUC': '#1f77b4', 'Test AUC': '#ff7f0e'}
    )

    # Add traces
    for trace in bar1.data:
        figs.add_trace(trace, row=1, col=1)

    for trace in bar2.data:
        figs.add_trace(trace, row=1, col=2)

    # Final layout tweaks
    figs.update_layout(
        title="Validation vs. Test Performance Summary",
        template="plotly_white",
        height=500,
        width=1100,
        font=dict(size=13),
        showlegend=True,
        legend=dict(x=0.5, y=-0.15, orientation="h", xanchor="center")
    )

    figs.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    mo.vstack(
        [
            mo.image('./img/catMLP.png'),
            mo.ui.plotly(figs)
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""As shown in the performance summary, this architecture delivered significant improvements across most metrics—including AUC and MCC—while sensitivity, though improved, still lags behind, reflecting the lingering effects of data imbalance. GeneScope now provides an interactive API that allows users to upload their own clinical and gene expression data for staging classification. With user consent, we aim to collect and label submitted data to enrich our training dataset. This would enable us to periodically retrain the model, adapt to broader population diversity, and further close the sensitivity gap—especially for rare or underrepresented cancer subtypes.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # <span style="color:brown">Model Performance Comparison</span>

     To evaluate the effectiveness of our **CatBoost + MLP + Intermediate Fusion** architecture, we benchmarked it against several baseline models—including a **Vanilla Neural Network (VNN)**, **K-Nearest Neighbors (KNN)**, and a **dual-MLP model with Late Fusion**. Each of these alternatives was optimized using **SMOTE** to address class imbalance and ensure a fair comparison.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    vnn_md = mo.md("""
    ### Baseline Model: Vanilla Neural Network (VNN)

    Since our dataset includes well-processed gene expression and clinical features, a Vanilla Neural Network (VNN) served as an excellent starting point. It helped identify meaningful patterns without the overhead of complex architectures. We also applied PCA beforehand, which reduced dimensionality and made the data more suitable for a simple neural network to extract key signals.

    While our final model leverages deeper architectures for improved performance, the VNN played a critical role in shaping our modeling strategy. It provided a clean and interpretable baseline, helped us debug more effectively, and made the performance gains of deeper networks more measurable and meaningful.

    ---

    ### How a VNN Works

    At its core, a VNN consists of fully connected layers where each neuron applies a transformation:

    $$
    \mathbf{a}^{(l)} = f\\left( \mathbf{W}^{(l)} \\cdot \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)} \\right)
    $$

    Where:

     \( \mathbf{a}^{(l)} \): activation output of the \( l^\text{th} \) layer  
     \( \mathbf{W}^{(l)} \): weight matrix between layers  
     \( \mathbf{b}^{(l)} \): bias vector  
     \( f \): activation function (e.g., ReLU)  
     \( \mathbf{a}^{(0)} \): input layer (PCA-reduced gene expression + clinical features)

    ---

    ### Activation: ReLU Function

    In our implementation, we used the ReLU (Rectified Linear Unit) activation function:

    $$
    f(x) = \max(0, x)
    $$

    ReLU outputs the input if it's positive, otherwise it returns zero. It's computationally efficient and helps prevent the vanishing gradient problem — a common issue in deep learning.

    ---

    ### Learning: Backpropagation

    Training the network involves minimizing a loss function (e.g., cross-entropy for classification) via gradient descent:

    $$
    \\theta = \\theta - \\eta \\cdot \\nabla_{\\theta} \\mathcal{L}
    $$

    Where:
     \( \\theta \): model parameters (weights and biases)  
     \( \\eta \): learning rate  
     \( \\nabla_{\\theta} \\mathcal{L} \): gradient of the loss function

    ---

    Despite its simplicity, the VNN enabled our model to learn complex relationships between clinical variables and gene expression data. It laid the groundwork for deeper explorations, forming a vital part of our modeling pipeline in **GeneScope**.
    """)
    return (vnn_md,)


@app.cell(hide_code=True)
def _(mo):
    KNN_md = mo.md(f"""
    ### Baseline Model: K-Nearest Neighbors (KNN)

    To better understand the performance of our Deep Neural Network, we also included **K-Nearest Neighbors (KNN)** as a baseline comparison — and it proved to be highly insightful. KNN is a classic, non-parametric algorithm that doesn’t involve any “learning” during training. Instead, it memorizes the training data and makes predictions at inference time based on the closest samples in feature space.

    ---

    ### How KNN Works

    At its core, KNN makes predictions by calculating **distance** — typically **Euclidean distance** — between the new input and all training points:

    $$
    d(\\mathbf{{x}}, \\mathbf{{x}}_i) = \\sqrt{{ \\sum_{{j=1}}^n \\left( x_j - x_{{ij}} \\right)^2 }}
    $$

    Where:

     \( \\mathbf{{x}} \) is the new input vector (the sample to predict)  
     \( \\mathbf{{x}}_i \) is a point in the training set  
     \( x_j \) and \( x_{{ij}} \) are the \( j^\\text{{th}} \) features of those vectors  
     \( d(\\mathbf{{x}}, \\mathbf{{x}}_i) \) is the distance between the two vectors

    After computing distances to all training samples, KNN:

    1.Sorts the training samples by their distance to the new point  
    2.Selects the \( k \) closest neighbors  
    3.Assigns a label based on majority vote (classification) or averages their values (regression)

    ---

    ### Why Use KNN?

    KNN provides a **fundamentally different baseline**. Unlike Deep Neural Networks that learn abstract features during training, KNN operates purely on **local similarity**. This makes it a valuable tool for evaluating how well our **PCA-reduced features** capture meaningful structure in the data — especially in terms of how well cancer stages naturally cluster.

    However, KNN has limitations:

    It scales poorly with large datasets

    It performs poorly in high-dimensional spaces  

    It is sensitive to noise and irrelevant features  

    Because of these challenges, **dimensionality reduction via PCA** was a necessary step before applying KNN to our gene expression data.

    Despite its simplicity, KNN offered valuable insights into the geometry of our data and helped validate the separability of cancer stages — ultimately reinforcing the strengths of the more advanced architectures used in **GeneScope**.
    """)
    return (KNN_md,)


@app.cell(hide_code=True)
def _(KNN_md, mo, vnn_md):
    mo.accordion({
        "VNN Vanila Artifitial Nural Netwrok": mo.vstack([vnn_md, mo.image('./Mls_yar_DELETE/model_performance_vanila_NN.png')]),
        "KNN":mo.vstack([KNN_md,mo.image('./Mls_yar_DELETE/model_performance_KNN.png')])

    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### <span style="color:brown"> Final Comparison results</span>""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image('comparisonMLMTX/performance.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### <span style="color:brown">"Superior Robustness of the Main Model</span>
    While the Vanilla Neural Network (VNN) achieved slightly higher scores in some metrics, this advantage is largely attributed to SMOTE-based data augmentation. In biological contexts—especially with high-dimensional gene expression data—such augmentation can introduce synthetic noise and distort the natural distribution of patient samples, potentially leading to inflated metrics without meaningful biological validity.

    In contrast, the Main Model—which integrates CatBoost for clinical data and an MLP for gene expression using intermediate fusion—achieved consistently strong performance across Accuracy, MCC, and Sensitivity, without relying on synthetic oversampling. The model's ability to learn from true patterns across both modalities, while naturally handling imbalance via CatBoost's built-in class weighting, proves its strength.
    """
    )
    return


if __name__ == "__main__":
    app.run()
