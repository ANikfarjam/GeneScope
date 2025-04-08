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

        For studying prognosis, since we already planned to create an HMM model for data augmentation, we decided to analyze transmission, emission, and initial probabilities. This will give us valuable insight into the probability of cancer advancing to the next stage, as well as the probability of a malignant tissue falling into a particular stage of cancer.

        **<span style='color: brown'>Initial Probability Calculation</span>**

        The initial probability of a stage \( S_i \) is calculated by counting the occurrences of each stage in the dataset and dividing by the total number of samples. However, if the dataset is unbalanced, the probability calculation will be biased. To mitigate this bias, we apply a weight to each stage to balance the contributions:

        \[
        P(S_i) = \frac{w_i \times N_i}{\sum_{j} w_j \times N_j}
        \]

        Where:
        - \( N_i \) is the number of samples in stage \( S_i \).
        - \( w_i = \frac{N_{\text{min}}}{N_i} \) is the weight assigned to stage \( S_i \), where \( N_{\text{min}} \) is the count of the stage with the fewest samples.

        **<span style='color: brown'>Transition Probability Calculation</span>**

        The transition probability between two stages \( S_i \) and \( S_j \) is calculated using:

        \[
        P(S_j | S_i) = \frac{w_i \times T_{ij}}{\sum_k w_i \times T_{ik}}
        \]

        Where:
        - \( T_{ij} \) is the number of transitions from stage \( S_i \) to stage \( S_j \).
        - \( w_i \) is the weight calculated as above.

        This ensures that stages with fewer samples are appropriately considered during the transition probability calculation.

        **<span style='color: brown'>Emission Probability Calculation</span>**

        Emission probabilities are calculated separately for numerical and categorical variables.

        - **Numerical Data:**
          - The mean of each feature per stage is calculated.
        - **Categorical Data:**
          - The probability of each category within a column is calculated per stage.

        Combining the numerical and categorical probabilities provides a comprehensive emission probability matrix.
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
def _(emmition_p_df, initial_p_df, mo, pd, transition_p_df):
    import plotly.graph_objects as go
    import networkx as nx

    # Prepare the adjacency matrix by removing the 'Stage' column and converting to numeric values
    transition_p_dfs = transition_p_df.set_index('Stage')
    transition_p_dfs = transition_p_dfs.apply(pd.to_numeric, errors='coerce')

    # Set the index of initial_p_df to the 'Stage' column for easy lookup
    initial_p_dfs = initial_p_df.set_index('Stage')

    # Extracting stages from the dataframe
    stages = transition_p_dfs.index.tolist()
    columns = transition_p_dfs.columns.tolist()

    # Creating a directed graph
    G = nx.DiGraph()

    # Adding nodes
    G.add_nodes_from(stages)

    # Adding weighted edges based on transition probabilities, skipping self-loops
    for from_stage in stages:
        for to_stage in columns:
            if from_stage != to_stage:  # Ignore self-loops
                prob = transition_p_dfs.loc[from_stage, to_stage]  # Use transition_p_dfs here
                if prob > 0:
                    G.add_edge(from_stage, to_stage, weight=prob)

    # Define node positions for top-down hierarchical layout
    pos = {}
    layer_height = 200
    layer_width = 300
    y_position = 0

    # Define stages' positions manually to make them structured
    for i, stage in enumerate(stages):
        pos[stage] = (i * layer_width, y_position)
        if i % 2 == 1:  # Move to the next layer every two stages for clarity
            y_position -= layer_height

    # Creating edge traces
    edge_x = []
    edge_y = []
    edge_weights = []

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        edge_weights.append(edge[2]['weight'])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Extract initial probabilities for color mapping
    node_colors = [initial_p_dfs.loc[stage, 'count'] if stage in initial_p_dfs.index else 0 for stage in stages]

    # Creating node traces
    node_x = []
    node_y = []
    node_text = []
    for stage in stages:
        x, y = pos[stage]
        node_x.append(x)
        node_y.append(y)
        node_text.append(stage)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition='top center',
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            cmin=min(node_colors),
            cmax=max(node_colors),
            color=node_colors,
            colorbar=dict(
                title='Initial Probability',
                thickness=15,
                xanchor='left',
                titleside='right'
            ),
            size=15,
            line_width=2
        )
    )

    # Creating annotations for edge weights (probabilities)
    annotations = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2]['weight']
        annotations.append(
            dict(
                x=(x0 + x1) / 2,
                y=(y0 + y1) / 2 - 15,  # Moving text slightly below the edges
                text=f'{weight:.4f}',
                showarrow=False,
                font=dict(color="blue", size=10)  # Making weight text blue
            )
        )

    # Creating the figure and naming it 'transition_fig'
    transition_fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Breast Cancer Stage Transition Probability Tree',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        annotations=annotations,
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='rgba(240, 240, 240, 0.8)'
                    )
                   )

    # Display the Plotly figure
    mo.vstack([
        mo.ui.tabs({
            'Weighted':mo.ui.plotly(transition_fig),
            'Unwieghted':mo.image('./unwighted.png')
                   }),
        mo.ui.tabs({
        'transition':mo.ui.table(transition_p_df),
        'initial':mo.ui.table(initial_p_df),
        'emmition':mo.ui.table(emmition_p_df),

        })])
    return (
        G,
        annotations,
        columns,
        edge,
        edge_trace,
        edge_weights,
        edge_x,
        edge_y,
        from_stage,
        go,
        i,
        initial_p_dfs,
        layer_height,
        layer_width,
        node_colors,
        node_text,
        node_trace,
        node_x,
        node_y,
        nx,
        pos,
        prob,
        stage,
        stages,
        to_stage,
        transition_fig,
        transition_p_dfs,
        weight,
        x,
        x0,
        x1,
        y,
        y0,
        y1,
        y_position,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### <span style="color: brown">Key Observations:</span>

        **<span style="color: brown">Initial Probabilities (Color Intensity):**

        * The color bar indicates initial probabilities for each stage, with darker shades representing lower probabilities and brighter shades representing higher probabilities.

        * Stage IIA remains the most frequent starting point, shown by its brighter shade compared to other stages. This indicates a higher weighted probability of diagnosis at this stage, even after adjusting for sample size disparities.

        * Stage IIIA has a darker shade, implying a much lower initial probability compared to others, likely due to its relative rarity or reduced representation in the dataset.

        **<span style="color: brown">Transition Probabilities (Edge Labels):</span>**

        * Transitions between stages are indicated by edge labels with probability values. These probabilities now account for sample size differences.

        * The highest transition probabilities are Stage IB to Stage II (0.1000) and Stage II to Stage IIA (0.1250). This suggests that disease progression from Stage IB to Stage II and from Stage II to Stage IIA remains frequent, even after weighting adjustments.

        * Transition probabilities are significantly lower between later stages, such as Stage IIIC to Stage IV (0.0119). This suggests that progression from Stage IIIC to Stage IV is still rare, emphasizing the importance of early-stage interventions.

        **<span style="color: brown">Progression Pattern:</span>**

        * The overall progression pattern remains consistent, with higher transition probabilities occurring between earlier stages and lower probabilities as the disease advances.

        * While there is a natural flow from early stages to advanced stages, the likelihood of progression reduces significantly as the stages progress, particularly beyond Stage IIA.


        **<span style="color: brown">Potential Implication:</span>**

        * Early Detection Matters: The visualization highlights that earlier stages are more likely to progress to slightly advanced stages, but the probability of reaching late-stage Stage IV is quite low.

        * Improved Accuracy Through Weighting: By factoring in sample size differences, the model offers a more balanced perspective of transition probabilities, ensuring that rarer stages are not overshadowed by more common ones.
        """
    )
    return


@app.cell(hide_code=True)
def _(pd):
    clinical = pd.read_csv('./AHPresults/fina_Stage_unaugmented.csv')
    clinical
    return (clinical,)


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


@app.cell
def _(clinical, mo):
    # Creating DataFrames from value_counts() instead of converting to dictionary
    regional_lymph = clinical[['Stage', 'ajcc_pathologic_n','paper_miRNA.Clusters','ethnicity','race', 'age_at_diagnosis', 'vital_status']].value_counts().reset_index(name='value')
    size = clinical[['Stage', 'ajcc_pathologic_t','paper_miRNA.Clusters','ethnicity','race', 'age_at_diagnosis', 'vital_status']].value_counts().reset_index(name='value')
    metastasize = clinical[['Stage', 'ajcc_pathologic_m','paper_miRNA.Clusters','ethnicity','race', 'age_at_diagnosis', 'vital_status']].value_counts().reset_index(name='value')

    # Displaying them in the UI as separate tabs
    data_table = mo.ui.tabs({
        'Tumor Size': mo.ui.table(size.sort_values(by='Stage')),
        'Regional lymph nodes': mo.ui.table(regional_lymph.sort_values(by='Stage')),
        'Distant metastasis': mo.ui.table(metastasize.sort_values(by='Stage'))
    })

    data_table
    return data_table, metastasize, regional_lymph, size


@app.cell
def _(clinical):
    alternate_table = clinical[['Stage',  'ajcc_pathologic_t', 'ajcc_pathologic_n','ajcc_pathologic_m','paper_miRNA.Clusters','ethnicity','race', 'age_at_diagnosis', 'vital_status']]
    #convert age from days to year
    alternate_table['age_at_diagnosis']=alternate_table['age_at_diagnosis'].apply(lambda age: float(age/365))
    alternate_table
    return (alternate_table,)


app._unparsable_cell(
    r"""
    \"\"\"

    0–4 years Infants/Toddlers
    5–14 years Childhood
    15–19 years Adolescents
    20-29 Young Adults
    30-49 Adults
    50–64 Middle-Aged Adults
    65 -  Seniors

    \"\"\"
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

    prognosis_fig = px.scatter(
        alternate_table,
        x=
    )
    """,
    name="_"
)


@app.cell
def _(clinical):
    clinical['race'].value_counts()
    return


if __name__ == "__main__":
    app.run()
