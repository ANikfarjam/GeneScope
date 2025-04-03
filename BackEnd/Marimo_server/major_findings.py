import marimo

__generated_with = "0.11.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ###<span style="color: green">Project Overview</span> 

        We analyze gene expression patterns in breast tissue samples to identify differentially expressed genes associated with early-stage breast cancer and to study cancer progression. Our goal is to develop a predictive model that accurately assesses breast cancer prognosis, improving early detection and treatment planning."
        Lot of early research paper were sujesting using models such as HMM how ever these models are Require pre-selected features (e.g., from AHP-based ranking). May miss complex interactions between genes because they rely on predefined relationships. Thus we decided to use deep learning approach to be able to create a robust model for subtype classificatino and also assist us with prognosis studies. We migh still use ahp analysis for some side edas. as well as corolation analysis on clinical variables helping furthur diganosis.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### <span style="color: green">Visually see the top 300 Genes in the Healthy samples dataset!

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

        <span style="color: green">Modified AHP:</span>

        * <span style="color: green">Two-Sample t-Test:</span>

        Purpose: Identifies statistically significant differences in gene expression between two groups (e.g., cancerous vs. healthy cells).

        Method: Compares the means of two independent samples using the t-statistic.

        Output: A t-score and p-value. A small p-value indicates significant differences in expression.

        * <span style="color: green">Entropy Test:</span>

        Purpose: Measures the disorder in gene expression levels.

        Method: Computes entropy using histogram-based probability distributions.

        Output: Higher entropy values indicate genes with more variability, which are more useful for classification.

        * <span style="color: green">Wilcoxon Rank-Sum Test:</span>

        Purpose: A non-parametric test used to rank genes based on their median expression differences.

        Method: Compares the ranks of two independent samples instead of their means.

        Output: A Wilcoxon statistic and a p-value. A low p-value suggests significant differences in gene ranks.

        * <span style="color: green">Signal-to-Noise Ratio (SNR):</span>

        Purpose: Compares the difference in mean expression levels relative to the standard deviation.

        Method: SNR is calculated as the difference between the means of two groups divided by the sum of their standard deviations.

        Output: A higher SNR suggests that the gene has a strong discriminatory power between groups.

        * <span style="color: green">AHP Weighted Ranking:</span>

        Purpose: Integrates statistical measures into a single weighted ranking system to prioritize significant genes.

        Method: Normalizes scores across all statistical tests and applies predefined weights.

        Output: A final ranking score indicating the importance of each gene in classification.

        * <span style="color: green"> Eigenvalues and Eigenvectors in Gene Selection (Modified AHP): </span>

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


        <span style="color: brown">Please run [computeAhp.py](https://github.com/ANikfarjam/GeneScope) to calculate the ahp scores!</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # **Pairwise Comparison Matrix and Eigenvalues in Modified AHP**

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


@app.cell
def _(mo):
    mo.md(
        r"""
        ### <span style='color: brown'>Prognosis Analysis</span>

        Cancer prognosis analysis involves estimating the likely course and outcome of a cancer, including survival rates and recurrence probabilities, by examining factors like tumor stage, size, and spread, as well as patient-specific characteristics. 
        """
    )
    return


@app.cell
def _(pd):
    stage_df = pd.read_csv('./AHPresults/stage_dataSet.csv')
    stage_df.iloc[:,:5]
    return (stage_df,)


@app.cell
def _(stage_df):
    stage_df.iloc[:,:].unique()
    return


if __name__ == "__main__":
    app.run()
