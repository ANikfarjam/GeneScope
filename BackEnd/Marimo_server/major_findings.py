import marimo

__generated_with = "0.12.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import plotly.io as pio
    pio.renderers.default = "iframe_connected"
    return (pio,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ###<span style="color: brown">Project Overview</span> 

        This project analyzes gene expression patterns in breast tissue samples, combined with clinical data, to identify key biomarkers associated with breast cancer and gain insight into patient prognosis. We apply statistical methods and machine learning algorithms to uncover genes that show significant differences in expression levels, highlighting those that may play a role in mutation and cancer development.

        Our prognosis analysis models the probability of a sample being diagnosed at each cancer stage based on its gene expression and clinical features. We also explore patterns across stages and evaluate how factors like tumor size and lymph node involvement contribute to disease progression and mortality risk.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### <span style="color: brown">Visually see the top 300 Genes in the Healthy samples dataset!

        We started by hypothesizing the most exressed genes in our healthy sample could be the ones that could play the main rolle in breast cells. Thus we decided to compare the mean expression of 300 most expressed genes in healthy samples and compare them to their mean expression in melignant samples. How ever we are going to use a modified Analytic Hierarchy Process, a decision-making method that uses pairwise comparisons to rank important genese in breast cancer for more acurate anlysis.
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
    # cancer_dataSet = pd.read_csv('./data/cancerExpressions.csv')
    # plot_genes = plot_df["Genes"].tolist()
    # average2 = cancer_dataSet[plot_genes].mean(axis=0)
    # plot2_df = pd.DataFrame({"Genes": plot_genes, "avg_expr_level": average2})
    plot2_df = pd.read_csv('./AHPresults/cancer_mean_exp.csv')
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
    mo.md("""Even without utilizing any statistical models, it is noticable that the same most expressed genes display significant differance between healthy and malignant samples.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ##<span style="color: brown">Analysis of biomarkers of breast cancers usting AHP</span>

        **The Cancer Biomarkers** are biological molecules that indicate the presence of cancer or abnormal cell processes. They can be found in blood, urine, tissue, or other bodily fluids. We are studying this by analysing the gene expressions of healthy and cancerous samples. To achive this, we are using <span style="color: green">Analatical Hiracial Process</span>.
        """
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
def _():
    """
    top 50 genes in each stage
    """
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Our ahp analysis shows that these are top 500 ranked genes associated with breast cancer and their function in the body.""")
    return


@app.cell(hide_code=True)
def _(mo, pd):
    desc_df = pd.read_csv("./AHPresults/gene_descriptions.csv")
    mo.ui.tabs(
        {
            "top 500 genes and thei info": mo.ui.table(desc_df),
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
    return (desc_df,)


@app.cell(hide_code=True)
def _(ahp_top, mo, pd, px):
    import numpy as np
    # Read biomarker file
    biomarker = pd.read_csv('./AHPresults/fina_Stage_unaugmented.csv')

    # Load your top genes list
    top_20_gene = ahp_top['Gene'].to_list()[:20]

    # Filter to only those genes that exist in the biomarker DataFrame
    existing_genes = [gene for gene in top_20_gene if gene in biomarker.columns]

    # Optionally warn about missing genes
    missing_genes = [gene for gene in top_20_gene if gene not in biomarker.columns]
    if missing_genes:
        print("Warning: The following genes are missing from the biomarker dataset:", missing_genes)

    # Select only the available columns
    biomarker = biomarker.loc[:, ['Stage'] + existing_genes]
    biomarker = biomarker.groupby(by='Stage').mean().reset_index()
    biomarker = biomarker.set_index('Stage')

    # bio_graph = px.imshow(
    #     biomarker,
    #     text_auto=True,  # this is the correct Plotly arg for annotations
    #     aspect="auto",
    #     color_continuous_scale="Cividis",
    #     title="Mean Expression of Top Biomarkers by Cancer Stage"
    # )

    # bio_graph.update_layout(
    #     xaxis_title="Genes",
    #     yaxis_title="Cancer Stage",
    #     font=dict(size=12)
    # )

    # bio_graph.show()

    # Log transform for visualization only
    biomarker_viz = np.log1p(biomarker)

    # Convert real values to strings for annotation
    text_labels = biomarker.round(2).astype(str)

    # Plot
    bio_graph = px.imshow(
        biomarker_viz,
        text_auto=False,
        color_continuous_scale="blues",
        aspect="auto",
        title="Log-Scaled Mean Expression of Top Biomarkers by Cancer Stage"
    )

    bio_graph.update_layout(
        xaxis_title="Genes",
        yaxis_title="Cancer Stage",
        font=dict(size=12)
    )

    mo.ui.plotly(bio_graph)
    return (
        bio_graph,
        biomarker,
        biomarker_viz,
        existing_genes,
        missing_genes,
        np,
        text_labels,
        top_20_gene,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ###<span style='color:brown'>Key Insights from Top 20 Breast Cancer Biomarkers</span>

        In our study, we looked at thousands of genes and ranked the top 20 most important ones based on how strongly they relate to breast cancer stages. The heatmap shows how active (or “expressed”) each of these genes is at different stages of cancer—from early (Stage 0) to more advanced (Stage IV and X). Here’s what we found in simple terms:

        1. <span style='color:brown'>Some Genes Stay Loud the Whole Time </span>

        SPARCL1 and KIF4B are like background music that gets louder as the cancer gets worse.

        SPARCL1 helps cells move and stick to each other. High levels may mean that cancer cells are preparing to spread.

        KIF4B is involved when cells divide. High activity suggests that cancer cells are multiplying quickly.

        These two genes stay highly active across almost all cancer stages, which might make them reliable “watchdogs” to track the disease.

        2. <span style='color:brown'>Some Genes Shout Early, Then Quiet Down</span>

        MIR497HG and CYYR1 are more active in early stages like Stage 0 or I.

        MIR497HG is related to stopping cell growth and possibly helps kill damaged cells. High early levels could be the body’s way of trying to stop cancer early on.

        CYYR1 isn’t fully understood, but it might help with cell-to-cell communication, especially when things first start going wrong.

        These genes could be helpful for catching cancer before it spreads.

        3. <span style='color:brown'> Some Genes Build Up Over Time</span>

        Genes like NEK2 and SPC25 start off quiet but get more active in later stages (like Stage II and III).

        These genes help cells divide properly. In cancer, they may become overactive, allowing cells to grow out of control.

        That gradual increase might be a sign that the cancer is getting more aggressive.

        4. <span style='color:brown'> Communication Genes Show Up Later</span>

        IL11RA and JAM2 are more active in later stages, and they help with cell communication and possibly inflammation.

        Think of them like messengers or cell “doorbells”—when cancer becomes more serious, these messengers are used more often, possibly helping tumors grow or move to new places.
        """
    )
    return


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
    mo.image('./img/BRCA_StageGrouping.png')
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
def _(mo):
    mo.md(
        r"""
        ### Interpreting the Cox Proportional Hazards Model

        - A **hazard ratio > 1** means the variable **increases the risk** (a negative effect).
        - A **hazard ratio < 1** means the variable **decreases the risk** (a protective effect).
        - A **hazard ratio = 1** means the variable has **no effect** on the risk.

        ---

        ### Hypothesis Testing in the Cox Proportional Hazards Model

        **Null Hypothesis (H0):**  
        The covariate has no effect on the hazard (i.e., coef=0coef = 0, or HazardRatio=1Hazard Ratio = 1).

        ---

        - **p-value < 0.05** -> **Reject H0**  
          - **Statistically significant**  
          - The covariate **does affect** the hazard.

        - **p-value ≥ 0.05** -> **Fail to reject H0**  
          - **Not statistically significant**  
          - The covariate **does not have a significant effect** on the hazard.
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
            'Model Summery': mo.vstack([summery_fig, mo.ui.table(pd.read_csv('../Models/CoxPHFitter/result/cox_model_summary.csv',index_col=0).T)])
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        [Marion T Weigel etal](https://erc.bioscientifica.com/view/journals/erc/17/4/R245.xml?body=pdf-62565) published and article on the **<span style='color: brown'>Breast Cancer Prognosis** and as the result of their research the could massure the size of the tumorous tissue as well as number of cancerous serounding LymphNodes: 

        | Category | Tumor Size (cm) | Tumor Size (inches) | Description |
        |----------|-----------------|---------------------|------------|
        TX | — | — | Primary tumor cannot be assessed
        T0 | 0 | 0 | No evidence of primary tumor
        Tis | — | — | Carcinoma in situ
        T1mi | ≤ 0.1 | ≤ 0.04 | Microinvasion
        T1a | 0.1 – 0.5 | 0.04 – 0.2 | 
        T1b | 0.5 – 1 | 0.2 – 0.4 | 
        T1c | 1 – 2 | 0.4 – 0.8 | 
        T2 | 2 – 5 | 0.8 – 2.0 | 
        T3 | 5 | 2.0 | 
        T4 | Any size | Any size | Tumor of any size with direct extension to chest wall and/or skin


        | Category | Min Number of Lymph Node | Maximim Number of Lymph Node | Description |
        |----------|-----------------|---------------------|------------|
        | NX | | | Regional lymph nodes cannot be assessed (e.g. previously removed) |
        | N0 | 0 | 0 | No regional lymph node metastasis |
        | N1 | 1 | 3 | Metastasis to movable ipsilateral axillary lymph node(s) |
        | N2 | 4 | 6 | Metastasis to ipsilateral axillary lymph node(s) fixed to each other or to other structures |
        | N3a | 7 | 15 | Metastasis to ipsilateral internal mammary lymph node(s) |
        | N3b | 16 | | |
        """
    )
    return


@app.cell(hide_code=True)
def _(alternate_table, mo):
    mo.ui.table(alternate_table)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""For further prognosis analysis after calculationg risk factor, we decided to agregate our data further to group each stage with the Cancer Size, number Lymph node and age to further asses the charactristics of cancer stages.""")
    return


@app.cell(hide_code=True)
def _(alternate_table, pd):
    # Define the T-stage to tumor size mapping in centimeters
    t_stage_to_size = {
        'Tis': (0, 0),
        'T1mi': (0.1, 0.1),
        'T1a': (0.1, 0.5),
        'T1b': (0.5, 1),
        'T1c': (1, 2),
        'T1': (0.1, 2),     # fallback if T1 subtype isn't available
        'T2': (2, 5),
        'T3': (5, 7),
        'T4': (7, 10),
        'TX': (None, None)  # unassessable
    }

    # Define the N-stage to number of lymph nodes involved
    n_stage_to_count = {
        'N0': (0, 0),
        'N1': (1, 3),
        'N2': (4, 6),
        'N3a': (7, 15),
        'N3b': (16, float('inf')),
        'NX': (None, None)  # unassessable
    }

    # Function to convert cm to inches
    def cm_to_inch(cm_range):
        if cm_range[0] is None:
            return (None, None)
        return tuple(round(cm / 2.54, 2) for cm in cm_range)

    # Make a copy of your table
    df = alternate_table.copy()

    # Map tumor size range (in cm)
    df['tumor_size_cm'] = df['ajcc_pathologic_t'].map(lambda t: t_stage_to_size.get(t, (None, None)))
    df[['tumor_size_min_cm', 'tumor_size_max_cm']] = pd.DataFrame(df['tumor_size_cm'].tolist(), index=df.index)
    df[['tumor_size_min_in', 'tumor_size_max_in']] = df['tumor_size_cm'].apply(lambda cm: cm_to_inch(cm)).apply(pd.Series)
    df.drop(columns='tumor_size_cm', inplace=True)

    # Map regional lymph node count range
    df['lymph_nodes_range'] = df['ajcc_pathologic_n'].map(lambda n: n_stage_to_count.get(n, (None, None)))
    df[['lymph_nodes_min', 'lymph_nodes_max']] = pd.DataFrame(df['lymph_nodes_range'].tolist(), index=df.index)
    df.drop(columns='lymph_nodes_range', inplace=True)

    # Aggregate the data (example aggregation on tumor_size_min_cm, adjust as needed)
    agg_df = df.groupby(by=['Stage', 'ethnicity', 'race', 'age_at_diagnosis', 'vital_status']).agg({
        'tumor_size_min_cm': 'mean',
        'tumor_size_max_cm': 'mean',
        'lymph_nodes_min': 'mean',
        'lymph_nodes_max': 'mean'
    }).reset_index()
    return agg_df, cm_to_inch, df, n_stage_to_count, t_stage_to_size


@app.cell
def _(agg_df):
    agg_df
    return


@app.cell(hide_code=True)
def _(agg_df, mo):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    # If it's already categorized, just rename the column
    agg_df.rename(columns={'age_at_diagnosis': 'age_group'}, inplace=True)

    # Then melt the data for plotting
    df_melted = agg_df.melt(
        id_vars=['Stage', 'age_group', 'race'],
        value_vars=['tumor_size_min_cm', 'lymph_nodes_min'],
        var_name='Measurement',
        value_name='Value'
    )
    # Grouped mean for age_group bars
    bar_data = df_melted.groupby(['Stage', 'age_group', 'Measurement']).agg(
        mean_value=('Value', 'mean')
    ).reset_index()

    # Grouped mean for race lines
    race_data = df_melted.groupby(['Stage', 'race', 'Measurement']).agg(
        mean_value=('Value', 'mean')
    ).reset_index()

    # Create subplots
    unique_measurements = bar_data['Measurement'].unique()
    prog_fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=[f"{m}" for m in unique_measurements],
        vertical_spacing=0.15
    )

    # Define custom colors
    age_colors = {
        'Young Adults': 'rgba(148, 103, 189, 0.7)',
        'Adults': 'rgba(44, 160, 44, 0.7)',
        'Middle-Aged Adults': 'rgba(31, 119, 180, 0.7)',
        'Seniors': 'rgba(255, 127, 14, 0.7)'
    }
    line_styles = ['solid', 'dash', 'dot', 'dashdot']
    line_color_map = {}

    # Add bar plots (age_group)
    for i, measurement in enumerate(unique_measurements, start=1):
        m_df = bar_data[bar_data['Measurement'] == measurement]
        for age_groups in m_df['age_group'].unique():  
            subset = m_df[m_df['age_group'] == age_groups]
            prog_fig.add_trace(go.Bar(
                x=subset['Stage'],
                y=subset['mean_value'],
                name=age_groups,  
                marker_color=age_colors.get(age_groups, 'gray'),
                showlegend=(i == 1)
            ), row=i, col=1)


    # Add line plots (race)
    for i, measurement in enumerate(unique_measurements, start=1):
        r_df = race_data[race_data['Measurement'] == measurement]
        for j, race in enumerate(r_df['race'].unique()):
            subset = r_df[r_df['race'] == race]
            if race not in line_color_map:
                line_color_map[race] = f"rgba({50+j*30}, {100+j*20}, {150+j*10}, 1)"
            prog_fig.add_trace(go.Scatter(
                x=subset['Stage'],
                y=subset['mean_value'],
                mode='lines+markers',
                name=race,
                line=dict(dash=line_styles[j % len(line_styles)],
                          color=line_color_map[race]),
                showlegend=(i == 1)
            ), row=i, col=1)

    # Final touches
    prog_fig.update_layout(
        height=850,
        title='Mean Tumor Size and Lymph Node Count by Stage\nAge (Bars) and Race (Lines)',
        barmode='group',
        font=dict(size=13),
        legend_title_text='Group',
        margin=dict(t=100)
    )
    prog_fig.update_yaxes(title_text="Mean Value (cm / node count)", row=1)
    prog_fig.update_yaxes(title_text="Mean Value (cm / node count)", row=2)
    prog_fig.update_xaxes(title_text="Cancer Stage", tickangle=45)

    mo.ui.plotly(prog_fig)
    return (
        age_colors,
        age_groups,
        bar_data,
        df_melted,
        go,
        i,
        j,
        line_color_map,
        line_styles,
        m_df,
        make_subplots,
        measurement,
        prog_fig,
        r_df,
        race,
        race_data,
        subset,
        unique_measurements,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ###<span style='color:brown'>Key Findings from Tumor and Lymph Node Analysis</span>

        From our analysis of breast cancer progression across different stages, several patterns emerge when comparing tumor size and lymph node involvement by age group (bars) and race (lines):

        * Tumor Growth Peaks at Stage IIIA:


        Across all age groups, tumor sizes show a significant increase by Stage IIIA, with values reaching 6–7 cm on average. Young Adults, suprisingly, show notably large tumor sizes at this stage, but smaller number of lymph nodes compare to middle age group. Supporting our risk detection calculatiion. Thus we can make conclussion on the Young adults even are less prone to be diagnosed with breast cancer but they are more at risk at early stages of breast cancer. So an emmidiate profetional treatment are very crucial to this group.

        * Lymph Node Involvement Mirrors Tumor Size Trends:


        Lymph node counts also peak around Stage IIIA, especially among Middle-Aged Adults and Seniors, indicating a strong correlation between tumor size and lymphatic spread during mid-to-late cancer progression.

        * Notable Variation Across Racial Groups in Advanced Stages:

        By Stage IIIA and beyond, Black or African American patients consistently show higher tumor sizes and lymph node involvement compared to other races. This may reflect disparities in access to care, diagnosis delays, or biological factors and requires deeper investigation. Asian patients closely show high toumor size and lymph node, specially on stage III.

        * Age-Related Trends in Early Detection:

        Early-stage tumors (Stages 0–II) are generally smaller and associated with fewer lymph nodes, particularly among Adults and Seniors. This could suggest more consistent screening in these age groups compared to Young Adults.

        * Variability in Stage IV and Stage X:

        Stage IV and unclassified Stage X cases show fluctuating tumor and lymph node metrics across all groups, likely reflecting the heterogeneous nature of advanced and unstageable cancers.
        """
    )
    return


if __name__ == "__main__":
    app.run()
