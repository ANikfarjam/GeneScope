import marimo

__generated_with = "0.12.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ###<span style="color: brown">Project Overview</span> 

        We analyze gene expression patterns in breast tissue samples to identify differentially expressed genes associated with early-stage breast cancer and to study cancer progression. Our goal is to develop a predictive model that accurately assesses breast cancer prognosis, improving early detection and treatment planning."
        Lot of early research paper were sujesting using models such as HMM how ever these models are Require pre-selected features (e.g., from AHP-based ranking). May miss complex interactions between genes because they rely on predefined relationships. Thus we decided to use deep learning approach to be able to create a robust model for subtype classificatino and also assist us with prognosis studies. We migh still use ahp analysis for some side edas. as well as corolation analysis on clinical variables helping furthur diganosis.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### <span style="color: brown">Visually see the top 300 Genes in the Healthy samples dataset!

        Now lets see what are the most exressed genes in the BRCA genes. The mean expression for all the genes is selcted and following chart is showing the top 300 genes that had highes mean.

        **Note:** Even the genes has the most variations posses importance but this is not best way to calculate that. Instead we are going to use AHP analysis to rank the genes and analyze that more accurately.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    import pandas as pd
    import plotly.express as px

    # healthy_dataSet = pd.read_csv("../../data/ModelDataSets/helthyExpressions.csv")
    # healthy_dataSet.set_index(healthy_dataSet.columns[0], inplace=True)
    healthy_dataSet = pd.read_csv(
        './data/helthyExpressions.csv', sep=",", index_col=0
    )
    # healthy_dataSet.set_index(healthy_dataSet.columns[0], inplace=True)
    # Extract genes and compute mean expression
    genes = healthy_dataSet.columns
    average = healthy_dataSet.mean(axis=0)  # Compute mean across all samples

    # Create a new DataFrame with gene names and their average expression
    plot_df = pd.DataFrame({"Genes": genes, "avg_expr_level": average})

    # Sort and select the top 300 genes by expression level
    plot_df = plot_df.sort_values(by="avg_expr_level", ascending=False).iloc[:300]

    # Create an interactive bar plot
    fig = mo.ui.plotly(
        px.bar(
            plot_df,
            x="Genes",
            y="avg_expr_level",
            title="Gene Expression Visualization for Top 300 Genes",
            labels={"avg_expr_level": "Mean Expression Level"},
            color="avg_expr_level",  # Optional: color for better visualization
        ).update_layout(xaxis_tickangle=-45)
    )

    # Rotate x-axis labels for better visibility
    mo.ui.tabs({"Visually": fig, "Related_Data": mo.ui.table(healthy_dataSet)})
    return average, fig, genes, healthy_dataSet, pd, plot_df, px


@app.cell(hide_code=True)
def _(mo, pd, plot_df, px):
    # cancer_dataSet = pd.read_csv("../../data/ModelDataSets/cancerExpressions.csv")
    cancer_dataSet = pd.read_csv('./data/cancerExpressions.csv')
    plot_genes = plot_df["Genes"].tolist()
    average2 = cancer_dataSet[plot_genes].mean(axis=0)
    plot2_df = pd.DataFrame({"Genes": plot_genes, "avg_expr_level": average2})
    fig2 = mo.ui.plotly(
        px.bar(
            plot2_df,
            x="Genes",
            y="avg_expr_level",
            title="Gene Expression Visualization for Top 300 Genes",
            labels={"avg_expr_level": "Mean Expression Level"},
            color="avg_expr_level",  # Optional: color for better visualization
        ).update_layout(xaxis_tickangle=-45)
    )

    # Rotate x-axis labels for better visibility
    mo.ui.tabs({"Visualization": fig2, "Relted_Data": mo.ui.table(plot2_df)})
    return average2, cancer_dataSet, fig2, plot2_df, plot_genes


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ##<span style="color: brown">Analysis of biomarkers of breast cancers usting AHP</span>

        **The Cancer Biomarkers** are biological molecules that indicate the presence of cancer or abnormal cell processes. They can be found in blood, urine, tissue, or other bodily fluids. We are studying this by analysing the gene expressions of healthy and cancerous samples. To achive this, we are using <span style="color: green">Analatical Hiracial Process</span>. 

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
        - The pairwise comparison matrix is built using **quantitative ranking** instead of expert judgment.
        - The eigenvector of the matrix determines the relative ranking of genes.
        - The eigenvalue and consistency ratio ensure that the matrix is valid for decision-making.

        This method ensures an **objective** and **stable** way to rank genes, which improves classification accuracy in cancer detection using **Hidden Markov Models (HMMs)**.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(
        "AHPexplnation.png",
        caption="The hierarchy of factors for gene selection by AHP.",
    )
    return


@app.cell
def _(pd):
    ahp_df = pd.read_csv('./AHPresults/final_Mod_ahp_scores.csv')
    return (ahp_df,)


@app.cell(hide_code=True)
def _(ahp_df, mo):
    import altair as alt


    # Sort and take top 500 genes
    ahp_top = ahp_df.sort_values(by="Scores", ascending=False).iloc[:500, :]

    # Ensure "Gene" column is retained and treated as string
    ahp_top_scaled = ahp_top.copy()
    ahp_top_scaled.iloc[:, -1:] *= 1e6  # Apply scaling
    ahp_top_scaled["Gene"] = ahp_top_scaled["Gene"].astype(str)
    ahp_top_scaled["Scores"] = ahp_top_scaled["Scores"].astype(float)
    # Selection for interactive brushing
    brush = alt.selection_interval(encodings=["x", "y"])
    # Scatter Plot (Interactive)
    chart = mo.ui.altair_chart(
        alt.Chart(ahp_top_scaled)
        .mark_circle()
        .encode(
            x="Scores:Q",
            y="t_test:Q",
            color="entropy:Q",
            size="Wilcoxon",
            tooltip=[
                "Gene:N",
                "Scores:Q",
                "t_test:Q",
                "entropy:Q",
                "roc_auc:Q",
                "Wilcoxon",
                "Wilcoxon_p",
                "snr:Q",
            ],
        )
        .add_params(brush)
    )
    return ahp_top, ahp_top_scaled, alt, brush, chart


@app.cell(hide_code=True)
def _(chart, mo):
    # Display chart and dynamically updating table
    mo.vstack([chart, mo.ui.table(chart.value)])
    return


@app.cell
def _(ahp_df, mo):
    mo.ui.data_explorer(ahp_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Our ahp analysis shows that these are top 500 ranked genes associated with breast cancer and their function in the body.""")
    return


@app.cell(hide_code=True)
def _(mo, pd):
    desc_df = pd.read_csv("./data/top500_desc.csv")
    mo.ui.tabs(
        {
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
                            mo.image("rocVahp.png"),
                            mo.image("snrVahp.png"),
                            mo.image("t_testVahp.png"),
                            mo.image("wilcoxonVahp.png"),
                        ]
                    ),
                ]
            ),
            "top 500 genes and thei info": mo.ui.table(desc_df),
        }
    )
    return (desc_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### <span style='color: brown'>Prognosis Analysis</span>

        The prognosis for cancer refers to the expected outcome or course of the disease, including the likelihood of survival and recurrence. It's a doctor's best estimate based on various factors related to the cancer itself and the patient's overall health. Prognosis is influenced by factors like the type and stage of cancer, the patient's age and health, and how the cancer responds to treatment. 

        We were initially planning to use Morkov model or Boosting algorithm to calculate transition probability of cancer stage to another. However since we dont have data that record stage progression we cant predict that. However we can futher analyze the probability of a patient get diagnosed with each stage. Our data can help us to also further study cancer sage with respect to parient's demographic and also study melignant samples by their size, lumpth nodes and their spread on to the other organisms. 


        **<span style='color: brown'>Initial Probability Calculation</span>**

        The initial probability of a stage \( S_i \) is calculated by counting the occurrences of each stage in the dataset and dividing by the total number of samples. However, if the dataset is unbalanced, the probability calculation will be biased. To mitigate this bias, we apply a weight to each stage to balance the contributions:

        \[
        P(S_i) = \frac{w_i \times N_i}{\sum_{j} w_j \times N_j}
        \]

        Where:
        - \( N_i \) is the number of samples in stage \( S_i \).
        - \( w_i = \frac{N_{\text{min}}}{N_i} \) is the weight assigned to stage \( S_i \), where \( N_{\text{min}} \) is the count of the stage with the fewest samples.

        ### <span style='color: brown'>Utilizing Catboost</span>

        CatBoost is a powerful gradient boosting algorithm, and can learn the underlying relationships between clinical and demographic features and cancer stage outcomes directly from the data. By training on features such as age, tumor size, lymph node involvement, metastasis, and patient ethnicity, CatBoost can model the probability of a patient being diagnosed at each cancer stage. It internally handles class imbalance through loss functions and built-in support for weighted datasets, allowing it to estimate class probabilities more accurately even when stage distributions are skewed. Thus, while we cannot trace how a patient moves from one stage to another, CatBoost enables us to assess the likelihood of a patient presenting with a specific stage, based on their biological and demographic profile.

        CatBoost offers built-in support for handling imbalanced classes using the auto_class_weights parameter, which automatically calculates class weights to ensure that minority classes are not underrepresented during training. This is especially useful in cancer prognosis tasks, where certain stages may have far fewer samples. You can specify values like "Balanced" or "SqrtBalanced" to control how the class weights are computed based on either direct ratios or square-root-scaled ratios. These weights are then used internally to modify the loss function, allowing the model to learn equally well from all classes despite imbalance. For example, in the "Balanced" setting, the class weight 

        **Balanced:**

        \[
        CW_k = \frac{\max\limits_{c=1}^{K} \left( \sum_{t_i=c} w_i \right)}{\sum_{t_i=k} w_i}
        \]

        **SqrtBalanced:**

        \[
        CW_k = \sqrt{ \frac{\max\limits_{c=1}^{K} \left( \sum_{t_i=c} w_i \right)}{\sum_{t_i=k} w_i} }
        \]
        """
    )
    return


@app.cell(hide_code=True)
def _(pd):
    """
    loading calculated probabilities from our csvs
    """
    initial_p_df = pd.read_csv('../Models/HMM/probabilitiesResults/weighted_initial_p.csv')
    emmition_p_df = pd.read_csv('../Models/HMM/probabilitiesResults/combined_em_p.csv')
    transition_p_df = pd.read_csv('../Models/HMM/probabilitiesResults/weighted_ts_p.csv')
    return emmition_p_df, initial_p_df, transition_p_df


@app.cell(hide_code=True)
def _(mo, pd, px):
    # Load data
    model_data = pd.read_csv('./AHPresults/fina_Stage_unaugmented.csv')
    stage_p_df = pd.read_csv('../Models/gbst/result/stage_result.csv', index_col=0)
    model_matrix = pd.read_csv('../Models/gbst/result/classification_mtrx.csv', index_col=0)
    gdp_matrix = pd.read_csv('../Models/gbst/result/gdb_p_result.csv')
    manually_calc_prop = pd.read_csv('../Models/HMM/probabilitiesResults/weighted_initial_p.csv')

    # Format 'count' column to 2 decimal places for display
    manually_calc_prop['count'] = manually_calc_prop['count'].round(2)
    manually_calc_prop.rename(columns={'count': 'calculate_p'}, inplace=True)


    # Merge stage probabilities with manually calculated probabilities
    merged_df = stage_p_df.merge(manually_calc_prop, left_on=stage_p_df.index, right_on=manually_calc_prop['Stage'])

    # Drop 'Stage' column if no longer needed
    merged_df.drop(columns='Stage', inplace=True)

    # Plot stage probabilities
    stage_fig = px.bar(
        merged_df,
        x=merged_df.index,
        y=['Estimated Probability', 'calculate_p']
    )

    # Aggregate GDP matrix for True Stage
    columns_to_sum = gdp_matrix.columns.drop('True Stage')
    gdp_agg = gdp_matrix.groupby('True Stage')[columns_to_sum].mean().reset_index()

    # Create image heatmap of GDP aggregated results
    gdg_fig = px.imshow(gdp_agg.iloc[:, 1:])

    # Display in Marimo tabs
    mo.ui.tabs(
        {
            'Stage Diagnosis Probability': mo.vstack([mo.ui.plotly(stage_fig), mo.ui.table(model_data)]),
            'CatBoost Performance': mo.hstack([mo.ui.plotly(gdg_fig), mo.ui.table(model_matrix)])
        }
    )
    return (
        columns_to_sum,
        gdg_fig,
        gdp_agg,
        gdp_matrix,
        manually_calc_prop,
        merged_df,
        model_data,
        model_matrix,
        stage_fig,
        stage_p_df,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### <span style="color: brown">Key Observations:</span>

        The comparison between CatBoost-estimated stage probabilities and the manually weighted probabilities highlights a consistent pattern: early-stage diagnoses (e.g., Stage IIA, IIB) are predicted with higher probability, while late-stage diagnoses (e.g., Stage IV) receive lower predicted probabilities.

        This makes sense within the clinical context of breast cancer, where advanced stages are rare at the point of initial diagnosis. According to the American Cancer Society, the majority of invasive breast cancers are diagnosed in early stages, and Stage IV cases account for a small minority of breast cancer presentations. This supports the biological plausibility of the lower predicted probabilities for late-stage cancers, rather than indicating bias or model weakness.

        ### <span style="color: brown">Conclusion (In the Context of Prognosis and Biomarker Analysis)</span>

        These findings reinforce the reliability of gene expression and clinical features in distinguishing between breast cancer stages, even in the absence of longitudinal data. While we cannot track stage transitions over time, we can use machine learning to assess the likelihood of a patient being diagnosed at a specific stage, based on their molecular and demographic profile.

        In this study, stage probabilities learned from CatBoost serve as a proxy for prognosis analysis, especially when tied back to gene-level biomarker rankings derived from our AHP method. The model’s tendency to assign lower probabilities to later stages mirrors real-world clinical distribution and strengthens the case that our biomarker-based models are capturing true biological signals relevant to diagnosis and prognosis.

        Ultimately, this validates our approach of combining modified AHP for gene ranking with CatBoost classification for prognosis modeling, creating a biologically informed and statistically sound method for supporting early detection and precision diagnostics in breast cancer care.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ###<span style="color:brown">Medical Terminalogies for Cancer Staging</span>

        **Cancer Staging:**

        Stage refers to the extent of your cancer, such as how large the tumor is and if it has spread. 

        **Systems That Describe Stage:**

        There are many staging systems. Some, such as the TNM staging system, are used for many types of cancer. Others are specific to a particular type of cancer. Most staging systems include information about where the tumor is located in the body
        the size of the tumor whether the cancer has spread to nearby lymph nodes
        whether the cancer has spread to a different part of the body

        **The TNM Staging System**

        The TNM system is the most widely used cancer staging system. Most hospitals and medical centers use the TNM system as their main method for cancer reporting. You are likely to see your cancer described by this staging system in your pathology report unless there is a different staging system for your type of cancer. Examples of cancers with different staging systems include brain and spinal cord tumors and blood cancers. 

        In the TNM system:

        * The T refers to the size and extent of the main tumor. The main tumor is usually called the primary tumor.
        * The N refers to the number of nearby lymph nodes that have cancer.
        * The M refers to whether the cancer has metastasized. This means that the cancer has spread from the primary tumor to other parts of the body.
        * When your cancer is described by the TNM system, there will be numbers after each letter that give more details about the cancer—for example, T1N0MX or T3N1M0. The following explains what the letters and numbers mean.

        Primary tumor (T):

        * TX: Main tumor cannot be measured.
        * T0: Main tumor cannot be found.
        * T1, T2, T3, T4: Refers to the size and/or extent of the main tumor. The higher the number after the T, the larger the tumor or the more it has grown into nearby tissues. * * T's may be further divided to provide more detail, such as T3a and T3b.

        Regional lymph nodes (N):

        * NX: Cancer in nearby lymph nodes cannot be measured.
        * N0: There is no cancer in nearby lymph nodes.
        * N1, N2, N3: Refers to the number and location of lymph nodes that contain cancer. The higher the number after the N, the more lymph nodes that contain cancer.

        Distant metastasis (M)

        * MX: Metastasis cannot be measured.
        * M0: Cancer has not spread to other parts of the body.
        * M1: Cancer has spread to other parts of the body.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image('./BRCA_StageGrouping.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.hstack([
        mo.md(f"""
        **<span style="color: brown">Introduction to miRNA Clusters in Breast Cancer Analysis</span>**

    MicroRNAs (miRNAs) are small, non-coding RNAs that regulate gene expression by binding to mRNAs, thereby influencing critical cellular processes such as cell growth, apoptosis, metastasis, and immune response. In breast cancer, dysregulated miRNA expression is closely associated with disease progression, including tumor size, lymph node involvement, distant metastasis, and treatment response.

    miRNA clusters refer to groups of miRNAs that are either transcribed together from a common genomic region or exhibit similar expression patterns across samples. Clustering analysis helps identify patterns associated with distinct clinical features, providing insights into their potential role as biomarkers for prognosis and treatment outcomes.

    The following visualizations illustrate the distribution of miRNA clusters across various tumor sizes, lymph node involvement stages, and distant metastasis status, highlighting their relevance in breast cancer prognosis.
        """),
        mo.image('https://www.mayoclinic.org/-/media/kcms/gbs/patient-consumer/images/2013/08/26/09/58/br00022_im04258_br7_metastatic_breast_cancerthu_jpg.png', width= 500, height=400)
    ])
    return


@app.cell(hide_code=True)
def _(mo, model_data):
    # Creating DataFrames from value_counts() instead of converting to dictionary
    regional_lymph = model_data[['Stage', 'ajcc_pathologic_n','paper_miRNA.Clusters','ethnicity','race', 'age_at_diagnosis', 'vital_status']].value_counts().reset_index(name='value')
    size = model_data[['Stage', 'ajcc_pathologic_t','paper_miRNA.Clusters','ethnicity','race', 'age_at_diagnosis', 'vital_status']].value_counts().reset_index(name='value')
    metastasize = model_data[['Stage', 'ajcc_pathologic_m','paper_miRNA.Clusters','ethnicity','race', 'age_at_diagnosis', 'vital_status']].value_counts().reset_index(name='value')

    # Displaying them in the UI as separate tabs
    data_table = mo.ui.tabs({
        'Tumor Size': mo.ui.table(size.sort_values(by='Stage')),
        'Regional lymph nodes': mo.ui.table(regional_lymph.sort_values(by='Stage')),
        'Distant metastasis': mo.ui.table(metastasize.sort_values(by='Stage'))
    })
    return data_table, metastasize, regional_lymph, size


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### <span style='color:brown'>Cox Proportional Hazards Model in Our Study</span>

        For our breast cancer prognosis study, we use the Cox Proportional Hazards Model to understand how clinical and molecular factors influence a patient's risk of mortality over time. While our dataset lacks exact time-to-death, we approximate follow-up duration based on diagnosis year to estimate survival outcomes. By modeling variables like tumor characteristics (T/N/M), age at diagnosis, ethnicity, race, and miRNA expression clusters, we identify which features are associated with higher or lower hazard (risk) of death. This helps us interpret the prognostic value of key biomarkers and demographic factors in breast cancer outcomes.
        """
    )
    return


@app.cell(hide_code=True)
def _(model_data):
    alternate_table = model_data[['Stage',  'year_of_diagnosis','ajcc_pathologic_t', 'ajcc_pathologic_n','ajcc_pathologic_m','paper_miRNA.Clusters','ethnicity','race', 'age_at_diagnosis', 'vital_status']]
    #convert age from days to year
    alternate_table['age_at_diagnosis']=alternate_table['age_at_diagnosis'].apply(lambda age: float(age/365))
    """

    0–4 years Infants/Toddlers
    5–14 years Childhood
    15–19 years Adolescents
    20-29 Young Adults
    30-49 Adults
    50–64 Middle-Aged Adults
    65 -  Seniors

    """
    def age_group(age):
        if age <= 4:
            return 'Infants/Toddlers'
        elif 4 < age <= 14:
            return 'Childhood'
        elif 14 < age <= 19:
            return 'Adolescents'
        elif 19 < age <= 29:
            return 'Young Adults'
        elif 29 < age <= 49:
            return 'Adults'
        elif 49 < age <= 64:
            return 'Middle-Aged Adults'
        else:
            return 'Seniors'
    alternate_table['age_at_diagnosis'] = alternate_table['age_at_diagnosis'].apply(age_group)

    return age_group, alternate_table


@app.cell(hide_code=True)
def _(mo, pd, px):
    #import data
    cox_hazerdus_p = pd.read_csv('../Models/CoxPHFitter/result/cox_comparison_metrics.csv')
    cox_hazerdus_p.drop(columns='cmp to', inplace=True)
    cox_summry = pd.read_csv('../Models/CoxPHFitter/result/cox_model_summary.csv',index_col=0)
    cox_summry = cox_summry.T
    # Replace any values > 10 with NaN (or clip at 10 if you prefer)
    cox_summry_numeric = cox_summry.select_dtypes(include='number')
    cox_summry[cox_summry_numeric.columns] = cox_summry_numeric.where(cox_summry_numeric <= 10)
    hazard_fig = px.bar(cox_hazerdus_p, x='covariate', y='p', color='z')
    # cox_summry.set_index('covariate',inplace=True)
    summery_fig = px.imshow(cox_summry)

    insight = mo.md(f"""
    ### <span style='color:brown'>Key Findings</span>

    **<span style='color:brown'>Elevated Risk Among Asian Populations:</span>**

    * The model indicates that individuals identified as Asian have a higher hazard ratio compared to the reference group (White individuals).  
    * This aligns with epidemiological data showing that cancer is the leading cause of death among Asian Americans, with higher incidences of liver, stomach, and nasopharyngeal cancers.  
      [Source: PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5283572/)

    **<span style='color:brown'>Age-Related Risks:</span>**

    * Seniors exhibit a significantly higher hazard ratio, suggesting increased mortality risk with advancing age.  
    * Conversely, young adults show a lower hazard ratio, indicating a reduced risk relative to the reference group.

    **<span style='color:brown'>Impact of Cancer Stage:</span>**

    * Advanced cancer stages, particularly Stage IV, are associated with markedly higher hazard ratios, underscoring the critical importance of early detection and intervention.

    **<span style='color:brown'>Genetic and Biological Factors:</span>**

    * Certain genetic polymorphisms prevalent in Asian populations, such as ALDH2 deficiency, contribute to increased susceptibility to cancers like esophageal cancer.  
      [Source: Wikipedia - Alcohol Flush Reaction](https://en.wikipedia.org/wiki/Alcohol_flush_reaction)

    * Additionally, variations in the CYP2D6 gene affect drug metabolism, potentially influencing treatment efficacy and outcomes.  
      [Source: Wikipedia - CYP2D6](https://en.wikipedia.org/wiki/CYP2D6)
    """)


    mo.vstack([
        mo.ui.tabs({
            'Hazerdus Probabilities': hazard_fig,
            'Model Summery': summery_fig
        }),
        insight
    ])


    return (
        cox_hazerdus_p,
        cox_summry,
        cox_summry_numeric,
        hazard_fig,
        insight,
        summery_fig,
    )


if __name__ == "__main__":
    app.run()
