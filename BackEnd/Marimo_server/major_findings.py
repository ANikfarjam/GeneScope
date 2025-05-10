import marimo

__generated_with = "0.12.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    from itertools import product
    import pickle as pkl
    import pandas as pd
    import plotly.express as px
    return go, make_subplots, mo, pd, pkl, product, px


@app.cell(hide_code=True)
def _():
    import plotly.io as pio
    pio.renderers.default = "iframe_connected"
    return (pio,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ###<span style="color: brown">Overview</span> 

        This project analyzes gene expression patterns in breast tissue samples, combined with clinical data, to identify key biomarkers associated with breast cancer and gain insight into patient prognosis. We apply statistical methods and machine learning algorithms to uncover genes that show significant differences in expression levels, highlighting those that may play a role in mutation and cancer development.

        Our prognosis analysis models the probability of a sample being diagnosed at each cancer stage based on its gene expression and clinical features. We also explore patterns across stages and evaluate how factors like tumor size and lymph node involvement contribute to disease progression and mortality risk.

        Through our analysis we found intresting findings that would like to share with you.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # <span style="color: brown">Biomarkers</span>
        ### <span style="color: brown">Most Active Genes in Healthy Tissue<span>

        We started by identifying the 300 most active genes in healthy breast tissue, as these likely play key roles in normal cell function. When we compared their activity levels to those in cancerous tissue, we immediately saw big differences, even before using advanced statistical tools. Our analysis doesn’t just focus on the highly active genes; it also considers genes with low or unusual activity. That’s because both overactivity and underactivity can disrupt how cells work. Too much gene activity can lead to uncontrolled cell growth or mutations, while too little can weaken the body’s ability to stop abnormal cells, including those that may become cancer.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, pkl):
    # import pandas as pd
    # import plotly.express as px

    # # healthy_dataSet = pd.read_csv("../../data/ModelDataSets/helthyExpressions.csv")
    # # healthy_dataSet.set_index(healthy_dataSet.columns[0], inplace=True)
    # healthy_dataSet = pd.read_csv(
    #     './data/helthyExpressions.csv', sep=",", index_col=0
    # )
    # # healthy_dataSet.set_index(healthy_dataSet.columns[0], inplace=True)
    # # Extract genes and compute mean expression
    # genes = healthy_dataSet.columns
    # average = healthy_dataSet.mean(axis=0)  # Compute mean across all samples

    # # Create a new DataFrame with gene names and their average expression
    # plot_df = pd.DataFrame({"Genes": genes, "avg_expr_level": average})

    # # Sort and select the top 300 genes by expression level
    # plot_df = plot_df.sort_values(by="avg_expr_level", ascending=False).iloc[:300]

    # # Create an interactive bar plot
    # fig = mo.ui.plotly(
    #     px.bar(
    #         plot_df,
    #         x="Genes",
    #         y="avg_expr_level",
    #         title="Gene Expression Visualization for Top 300 Genes",
    #         labels={"avg_expr_level": "Mean Expression Level"},
    #         color="avg_expr_level",  # Optional: color for better visualization
    #     ).update_layout(xaxis_tickangle=-45)
    # )

    # # Load gene descriptions
    # polished_df = pd.read_csv('./comparisonMLMTX/description_genes_healthy.csv')
    # html_table = polished_df.to_html(classes='healthy-table', index=False, escape=False)

    with open('./scripts/pkl_files/healthy_fig.pkl','br') as f:
        _fig = pkl.load(f)
    with open('./scripts/pkl_files/healthy_table.pkl','br') as f:
        _html_table = pkl.load(f)
    # Styled scrollable table (green tones for healthy)
    styled_table = f"""
    <style>
    .healthy-table td {{
        white-space: normal;
        word-wrap: break-word;
        max-width: 400px;
        vertical-align: top;
    }}
    .healthy-table th {{
        background-color: #f0f0f0;
        padding: 8px;
    }}
    .healthy-table td, .healthy-table th {{
        border: 1px solid #b2dfdb;
        padding: 8px;
        text-align: left;
        font-family: Arial, sans-serif;
        font-size: 14px;
    }}
    .scroll-box {{
        max-height: 500px;
        overflow-y: auto;
        padding-right: 10px;
    }}
    </style>
    <div class="scroll-box">
    {_html_table}
    </div>
    """

    # Styled summary panel
    styled_summary = """
    <div style="
        font-family: Arial, sans-serif;
        font-size: 15px;
        line-height: 1.6;
        background-color: #ebdada;
        padding: 20px;
        border-left: 5px solid #491e1e;
        max-width: 500px;
        height: 500px;
        overflow-y: auto;
    ">
    <b>Key Insight:</b><br><br>
    In healthy breast ductal tissue, the most highly expressed genes are largely responsible for maintaining cellular stability, supporting growth, and defending against stress. These genes play critical roles in essential processes like protein production, tissue integrity, and immune surveillance. For example, several top genes are involved in building ribosomes and managing protein synthesis—an indication of high cellular activity. Others regulate iron storage and prevent oxidative damage, helping cells stay healthy in a dynamic tissue environment. Some genes even contribute to immune functions that monitor for abnormal changes. Altogether, these active genes suggest that healthy breast ductal cells are highly engaged in growth, repair, and protection—working constantly to maintain a balanced and resilient tissue environment.
    </div>
    """




    # Display side-by-side using hstack

    # Rotate x-axis labels for better visibility
    mo.ui.tabs({
        "Visually": _fig, 
        "Key Insight": mo.hstack([
            mo.Html(styled_summary),
            mo.Html(styled_table)
        ])
    })
    return f, styled_summary, styled_table


@app.cell(hide_code=True)
def _(mo, pd, pkl):
    # cancer_dataSet = pd.read_csv("../../data/ModelDataSets/cancerExpressions.csv")
    # cancer_dataSet = pd.read_csv('./data/cancerExpressions.csv')
    # plot_genes = plot_df["Genes"].tolist()
    # average2 = cancer_dataSet[plot_genes].mean(axis=0)
    # plot2_df = pd.DataFrame({"Genes": plot_genes, "avg_expr_level": average2})
    with open('./scripts/pkl_files/unhealthy_df.pkl', 'br') as _f:
        plot2_df = pd.read_pickle(_f)
    with open('./scripts/pkl_files/unhealthy_fig.pkl', 'br') as _f:
        fig2 = pkl.load(_f)

    # Rotate x-axis labels for better visibility
    mo.ui.tabs({"Visualization": fig2, "Relted_Data": mo.ui.table(plot2_df.reset_index().drop(['index', 'Unnamed: 0'], axis=1))})
    return fig2, plot2_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## <span style="color: brown">Using AHP to Identify Cancer Biomarkers</span>

        To identify genes most closely associated with breast cancer, we applied the Analytic Hierarchy Process (AHP), a structured ranking method that considers multiple factors at once. This method by utilizing ROC AUC, Signal-to-Noise Ratio, t-test & Wilcoxon tests, to measure how well a gene distinguishes between healthy and cancerous tissue.
        To capture expression consistency across samples and to detect statistically significant expression differences.

        This approach allowed us to prioritize genes likely to be functionally important, mutation-prone, or biologically disruptive, making them strong candidates for further research in cancer diagnostics and treatment.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, pkl):
    import altair as alt
    with open('scripts//pkl_files/ahp_top_scaled.pkl', 'rb') as _f:
        ahp_top_scaled = pkl.load(_f)

    # Reconstruct chart as before
    brush = alt.selection_interval(encodings=["x", "y"])
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

    return ahp_top_scaled, alt, brush, chart


@app.cell(hide_code=True)
def _(chart, mo):
    alt_plot = mo.vstack([chart, mo.ui.table(chart.value)])
    alt_plot
    return (alt_plot,)


@app.cell(hide_code=True)
def _(mo, pd):
    desc_df = pd.read_csv("./AHPresults/gene_descriptions.csv")
    # Create HTML table
    html_table_ahp = desc_df.to_html(classes='diff-table', index=False, escape=False)

    # CSS and scrollable container for the table
    styled_table_ahp = f"""
    <style>
    .diff-table td {{
        white-space: normal;
        word-wrap: break-word;
        max-width: 400px;
        vertical-align: top;
    }}
    .diff-table th {{
        background-color: #f0f0f0;
        padding: 8px;
    }}
    .diff-table td, .diff-table th {{
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
        font-family: Arial, sans-serif;
        font-size: 14px;
    }}
    .scroll-container {{
        max-height: 500px;
        overflow-y: auto;
        padding-right: 10px;
    }}
    </style>

    <div class="scroll-container">
    {html_table_ahp}
    </div>
    """

    # Styled summary panel
    styled_summary_ahp = """
    <div style="
        font-family: Arial, sans-serif;
        font-size: 15px;
        line-height: 1.6;
        background-color: #fefefe;
        padding: 20px;
        border-left: 5px solid #1d71b5;
        max-width: 500px;
        height: 500px;
        overflow-y: auto;
    ">
    <b>Key Insight:</b><br><br>
    After ranking the most important genes, we found that these genes show big differences in activity between healthy and cancerous breast tissue. Many of them help control how breast cells grow, stay healthy, or communicate with each other. Some of these genes don’t make proteins directly but instead act like switches that control other genes. These include long non-coding RNAs and microRNA host genes, which can be harmful when they’re not working properly—they can mess up how proteins are made or disrupt the normal function of breast cells. Other genes we found are linked to how cells move signals or maintain their outer structure. When these processes are thrown off, it can lead to cancer. Because of their roles, these genes could be useful for detecting breast cancer early or even for designing new treatments.
    </div>
    """

    # Display them side by side
    mo.hstack([
        mo.Html(styled_table_ahp),
        mo.Html(styled_summary_ahp)
    ])
    return desc_df, html_table_ahp, styled_summary_ahp, styled_table_ahp


@app.cell(hide_code=True)
def _(mo):
    mo.md("""We have stablished that genes most prone to mutations are those responsible for regulating the growth of cancerous cells and supplying nutrients to breast ductal cells. Now we are able to shift our focus to analyze how these genes are expressed across different stages of breast cancer. By doing so, we aim to uncover which genes are active at each stage and how they may work together to suppress tumor progression. This insight can enhance our understanding of the gene networks involved in cancer development and resistance visually and statistically.""")
    return


@app.cell(hide_code=True)
def _(mo, pkl):
    with open('scripts/pkl_files/ahp_hitmap.pkl', 'rb') as _f:
        bio_graph = pkl.load(_f)

    mo.ui.plotly(bio_graph)
    return (bio_graph,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ###<span style='color:brown'>Key Insights from Top 20 Breast Cancer Biomarkers</span>

        Based on the heatmap visualization and the functional roles of the top biomarkers, we observe that genes involved in cell proliferation, nutrient transport, and tumor suppression show distinct expression patterns across different stages of breast cancer. For instance, KIF4B and NEK2, both crucial for mitotic progression and chromosomal stability, display heightened expression across nearly all stages, suggesting a continuous requirement for cell division machinery during tumor growth. SPARCL1, known for its role in inhibiting tumor invasion and promoting cell adhesion, shows peak expression early on (notably Stage 0), potentially reflecting an early attempt to suppress tumor spread. Conversely, genes like PPAPDC1A and LOC283914, with more specialized or less-characterized functions, remain lowly expressed throughout, implying a more stage-specific or passive role. This expression trend reinforces the hypothesis that the most mutation-prone genes—those driving growth or metabolic support—are also those most dynamically regulated as the cancer progresses. These patterns provide insight into how gene networks may cooperate or become dysregulated during cancer evolution and could serve as stage-specific therapeutic targets or diagnostic markers.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # <span style='color: brown'>Prognosis Analysis</span>

        Cancer prognosis refers to the prediction of the likely course and outcome of the disease—specifically, the chances of survival, recurrence, and progression. For breast cancer, prognosis is heavily influenced by clinical stage at diagnosis, tumor size, lymph node involvement, distant that tumor has been grown, patient age, race, and genetic background.

        In our study, we leveraged clinical and gene expression data to estimate the probability of a patient being diagnosed at each cancer stage. However, since our data is not time seried and doesnt contains the prograssion data, we coudlnd calculate the probability of stage developement. Furhter more, we utilized Cox Hazerdous Regression model to identify how these clinical and biological factors can effect the patients well being and deepen our understanding of the Breast Cancer pathways.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        **<span style='color: brown'>Uteliazing Catboost for Initial Stage Probability Calculation</span>**

        The initial probability of a stage \( S_i \) is calculated by counting the occurrences of each stage in the dataset and dividing by the total number of samples. However, if the dataset is unbalanced, the probability calculation will be biased. To mitigate this bias, we apply a weight to each stage to balance the contributions:

        \[
        P(S_i) = \frac{w_i \times N_i}{\sum_{j} w_j \times N_j}
        \]

        Where:
        - \( N_i \) is the number of samples in stage \( S_i \).
        - \( w_i = \frac{N_{\text{min}}}{N_i} \) is the weight assigned to stage \( S_i \), where \( N_{\text{min}} \) is the count of the stage with the fewest samples.

        ### <span style='color: brown'>Utilizing Catboost</span>

        CatBoost is a powerful gradient boosting algorithm, and can learn the underlying relationships between clinical and demographic features and cancer stage outcomes directly from the data. By training on features such as age, tumor size, lymph node involvement, metastasis, and patient ethnicity, CatBoost can model the probability of a patient being diagnosed at each cancer stage. It internally handles class imbalance through loss functions and built-in support for weighted datasets, allowing it to estimate class probabilities more accurately even when stage distributions are skewed. Thus, while we cannot trace how a patient moves from one stage to another, CatBoost enables us to assess the likelihood of a patient presenting with a specific stage, based on their biological and demographic profile.

        CatBoost offers built-in support for handling imbalanced classes using the auto_class_weights parameter, which automatically calculates class weights to ensure that minority classes are not underrepresented during training. This is especially useful in cancer prognosis tasks, where certain stages may have far fewer samples. We can specify values like "Balanced" or "SqrtBalanced" to control how the class weights are computed based on either direct ratios or square-root-scaled ratios. These weights are then used internally to modify the loss function, allowing the model to learn equally well from all classes despite imbalance. For example, in the "Balanced" setting, the class weight 

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
    model_data = pd.read_csv('./AHPresults/fina_Stage_unaugmented2.csv', low_memory=False)
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

        ---

        ### <span style="color: brown">Medical Terminology</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Cancer Staging Overview": mo.md("""
            **Cancer Staging** refers to how much cancer is in the body and where it’s located. It helps doctors decide on treatment and predict outcomes.
        """),
        "Staging Systems": mo.vstack([
            mo.md("""
            There are several staging systems. The most common is the **TNM system**, which classifies cancer based on:
            - **T** (Tumor): Size and extent of the main tumor.
            - **N** (Nodes): Whether nearby lymph nodes are involved. Lymph nodes are small bean-shaped structures that are part of the body's immune system.
            - **M** (Metastasis): Whether the cancer has spread to distant organs.

            Understanding these numbers is very critical, because cancerous lymph nodes, tumor size, and how far the cancer has spread not only reflect disease progression but can also cause localized pain. As cancer grows, it may spread to other organs, posing serious danger.

            The following images illustrate lymph node involvement by stage and how breast cancer can spread to organs like the liver and lungs, requiring immediate attention.
            """),
            mo.hstack([
                mo.image('./img/Breast-cancer-stages-lymph.jpg'),
                mo.image('https://www.mayoclinic.org/-/media/kcms/gbs/patient-consumer/images/2013/08/26/09/58/br00022_im04258_br7_metastatic_breast_cancerthu_jpg.png', width=500, height=400)
            ])
        ]),
        "TNM Breakdown": mo.ui.tabs({
            "T (Tumor)": mo.md("""
                - **TX**: Cannot be measured.
                - **T0**: No tumor found.
                - **T1–T4**: Increasing size and/or local spread.
            """),
            "N (Nodes)": mo.md("""
                - **NX**: Lymph nodes can't be evaluated.
                - **N0**: No lymph node involvement.
                - **N1–N3**: Increasing number or fixation of involved lymph nodes.
            """),
            "M (Metastasis)": mo.md("""
                - **M0**: No distant spread.
                - **M1**: Cancer has spread to other parts of the body.
            """),
            "Breast Cancer TNM related Information": mo.image('./img/BRCA_StageGrouping.png')
        })
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ###<span style="color: brown">Identifying miRNAs in Breast Cancer Using BioMart</span>

        MicroRNAs (miRNAs) are a special type of gene—they do not produce proteins. Instead, they regulate other genes by turning them “on” or “off,” which can affect critical processes like cell growth, immune response, and cancer development. Because of this regulatory power, many miRNAs are considered biomarkers—biological indicators that can help detect or track disease.

        In our project, we did not initially know which of our top genes (ranked using AHP) were miRNAs or other non-coding RNAs. To solve this, we used Ensembl BioMart, a widely trusted biological database. BioMart allows researchers to input a list of gene names or Ensembl IDs and get back annotated information about each gene—such as whether it's protein-coding, miRNA, lncRNA, or another type.

        🔍 Why this matters:
        Identifying which genes are miRNAs helps us:

        Understand how gene expression is regulated in breast cancer.

        Discover non-coding RNAs that may play roles in tumor growth, metastasis, or treatment response.

        Prioritize genes for deeper study or potential clinical testing.

        🧬 What we did:

        We took the top 2,000 ranked genes from our AHP analysis.

        Used their gene names or Ensembl IDs to query BioMart.

        Retrieved structured annotations like gene type (e.g., miRNA, protein_coding), chromosomal location, and other metadata.

        **Our goal is to find out activity level of each of these non coding genes in each stage.**
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, pkl):
    with open('./alternative_prognosis data/miRna_visuals.pkl', 'rb') as file:
        charts = pkl.load(file)
    mo.ui.plotly(charts)
    return charts, file


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ###<span style='color:brown'>Key Findings</span>
        One of the most striking discoveries in our analysis was the consistently high expression of a particular microRNA, MIR4508, across all breast cancer stages—especially in advanced stages like Stage IV. This pattern suggests that MIR4508 may play a significant role in cancer progression. MicroRNAs like MIR4508 are small regulatory molecules that can influence the activity of other genes, particularly those involved in controlling cell growth and survival. In many cancers, overactive microRNAs are known to disrupt the balance of gene expression, potentially silencing tumor-suppressing genes or enhancing the effects of oncogenes. Based on its expression profile, MIR4508 may contribute to key cancer-related processes such as uncontrolled cell division, resistance to cell death, changes in cell structure that allow cancer to spread (a process known as epithelial-to-mesenchymal transition), and even resistance to treatment.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### <span style='color:brown'>Cox Proportional Hazards Model in Our Study</span>

        For our breast cancer prognosis study, we use the Cox Proportional Hazards Regression Model to understand how clinical and molecular factors influence a patient's risk of mortality over time. While our dataset lacks exact time-to-death, we approximate follow-up duration based on diagnosis year to estimate survival outcomes. By modeling variables like tumor characteristics (T/N/M), age at diagnosis, ethnicity, race, and miRNA expression clusters, we identify which features are associated with higher or lower hazard (risk) of death. This helps us interpret the prognostic value of key biomarkers and demographic factors in breast cancer outcomes.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Hypothesis Testing in the Cox Proportional Hazards Model and Interpretation

        **Null Hypothesis (H₀):**  
        The covariate has no effect on the hazard (i.e., coef = 0 or Hazard Ratio = 1).

        ---

        **Hazard Ratio (HR)**  
        - **HR > 1**: Increases death risk  
        - **HR < 1**: Protective effect  
        - **HR = 1**: No effect  

        ---

        **Coefficient (log hazard ratio)**  
        Indicates both the direction and strength of the effect:
        - **Positive coef** → Feature increases risk  
        - **Negative coef** → Feature decreases risk  

        ---

        - **p-value < 0.05** → **Reject H₀**  
          - Statistically significant  
          - The covariate **does affect** the hazard.

        - **p-value ≥ 0.05** → **Fail to reject H₀**  
          - Not statistically significant  
          - The covariate **does not have a significant effect** on the hazard.

        ---

        ***Our goal is to fine the features that fail the null hypothesis. After this, we are going to compare the HR and Coefficients.***
        """
    )
    return


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

    # cox_summry.set_index('covariate',inplace=True)
    summery_fig = px.imshow(cox_summry)

    # Filter features where p-value < 0.05 (i.e., reject the null hypothesis)
    significant_features_df = cox_summry.T
    significant_features_df = significant_features_df[significant_features_df["p"] < 0.05][["coef", "exp(coef)", "p"]]

    # Sort by p-value for better readability
    significant_features_df = significant_features_df.sort_values(by="p")

    hazard_fig = px.bar(significant_features_df, x=significant_features_df.index, y='coef', color='p')

    features_md = mo.md(f"""
    We set the null hypothesis for each feature to zero. This means the feature has no association with the hazard (risk of death in this context). In simpler terms: Rejecting the null hypothesis (p-value < 0.05).

    Features That Accept the Null Hypothesis
    Based on the contents of your cox_model_summary.csv, here are the features that fail to show statistical significance (p-value ≥ 0.05), meaning they accept the null hypothesis and are not significantly associated with survival risk:

    """)

    insight = mo.md(f"""

    Below is a table of features rejcting the null hypothesis. Aming to extract the the fignificant importance by analyzing ther HR and Log hazerdus ratio.


    {
    mo.ui.table(significant_features_df)
    }


    ℹ️ Why Do We Use `coef` (log scale) in the Model?

    The Cox model is based on the following formula:

    $$
    h(t \mid X) = h_0(t) \cdot \exp(\eta_1 X_1 + \eta_2 X_2 + \dots + \eta_n X_n)
    $$

    - `coef` is the **β (beta coefficient)** learned by the model for each variable.
    - It represents the **logarithmic effect** of the variable on the hazard (risk of event).
    - We use the log scale because:
      - It makes the math additive for multiple variables.
      - It ensures the final hazard ratio is always **positive** (since risk can't be negative).
    - To interpret the effect in real terms, we take the exponential:  
      **`exp(coef)` = hazard ratio (HR)**

    #### 🔁 Interpretation Examples:
    - `coef = 1.0` → `exp(coef) = 2.72` → **2.72× higher risk**
    - `coef = -0.5` → `exp(coef) = 0.61` → **39% lower risk**
    """)

    conclussion = mo.md(
    """
    <span style="color:brown">Conclusion</span>

    * Based on the features we extracted it seams like when cancer is at stage that its mutated LympthNodes are at category N3B, in this category cancer cells in lymph nodes are in the armpit and lymph nodes behind the breastbone the number of mutated lymphnodes are more than 10. With coef of 9.64, patients diagnosed with this catgory, are at really high mortality risk.

    * Genes miRNA that are part of cluster C5 are posing more danger for matients mortality. We already stablished that the noncoding gene that was highly expressed was the one that regulates emune system.

    * Finally the cinier patient obviously are more at risk. 
    """
    )

    mo.vstack([
        mo.ui.tabs({
            'Hazerdus Probabilities': hazard_fig,
            'Model Summery': mo.vstack([summery_fig, mo.ui.table(pd.read_csv('../Models/CoxPHFitter/result/cox_model_summary.csv',index_col=0).T, max_columns=15,

                                                                )])
        }),
        insight, conclussion
    ])
    return (
        conclussion,
        cox_hazerdus_p,
        cox_summry,
        cox_summry_numeric,
        features_md,
        hazard_fig,
        insight,
        significant_features_df,
        summery_fig,
    )


if __name__ == "__main__":
    app.run()
