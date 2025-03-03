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
        r"""
        <span style="color: green">Author: Ashkan Nikfarjam</span>

        Now that we have our data we are going to do AHP analysis that act as feature selection

        ahp.py has all the functions to calculate AHP for us.

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
        """
    )
    return


@app.cell
def _():
    """
    Implorting all the libraries
    """
    import numpy as np
    import pandas as pd
    # import tqdm
    # import multiprocessing as mp
    # import modin.pandas as mpd
    return np, pd


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
    mo.md(r"""now we use functions above to rank our genes""")
    return


@app.cell
def _(pd):
    #load dataset
    DCIS_dataset = pd.read_csv("DCIS_renamed_data.csv")
    DCIS_dataset=DCIS_dataset.set_index(DCIS_dataset['Gene'].to_numpy())
    DCIS_dataset.drop(columns=DCIS_dataset.columns[0],inplace=True)
    print(DCIS_dataset.head())
    return (DCIS_dataset,)


@app.cell
def _(DCIS_dataset, np, pd):
    import modAhpFunctions

    final_scores, priority_vectors, pairwise_matrices, consistency_ratios = modAhpFunctions.compute_ahp_scores(DCIS_dataset.iloc[:, :7].values, DCIS_dataset.iloc[:, 7:].values)

    results_df = pd.DataFrame({"Gene": DCIS_dataset.index.tolist(), **priority_vectors})
    print('Resulting DF:\n', results_df.head())
    results_df.to_csv("ahpResults/RCIS_Mod_new_scores.csv",index=False)
    np.save("ahpResults/pairwise_matrices.npy",pairwise_matrices)
    np.save("ahpResults/consistency_ratios_matrices.npy",consistency_ratios)

    return (
        consistency_ratios,
        final_scores,
        modAhpFunctions,
        pairwise_matrices,
        priority_vectors,
        results_df,
    )


if __name__ == "__main__":
    app.run()
