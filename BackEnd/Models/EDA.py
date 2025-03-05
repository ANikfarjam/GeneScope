import marimo

__generated_with = "0.11.7"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Author: Ashkan Nikfarjam""")
    return


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
        r"""
        # <span style="color:green">Data Source</span>

        The Gene Expression dataset has been extracted from the [NCBI Gene Expression Omnibus GSEGSE62944](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE62944). The dataset consists of 9264 tumor samples and 741 normal samples across 24 cancer types from The Cancer Genome Atlas. This data set is created from all the TCGA samples that have an expression of 23,000 genes!
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(src="TCGAcancerTypes.png",caption="The Camer Types available in TCGA database. This image is from the TCGA officcial website!")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""According to [NCBI]("") 20 years after the original publication of the human genome, the number of protein-coding genes is stabilizing around 19,500 (Figure 2), although the number of isoforms of these genes is still a subject of intensive study and discussion.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## <span style="color: green">Expression Collection Methods: 

        Next-Generation Sequencing (NGS) is a high-throughput method for sequencing DNA by following a series of steps. First, in library preparation, the genome is fragmented into smaller pieces, and short adapter sequences are attached to both ends, forming a DNA library. Next, in bridge amplification, these fragments are hybridized to a solid surface and undergo multiple amplification cycles, creating clusters of identical DNA fragments. In sequencing, fluorescently labeled nucleotides are incorporated into the DNA strands, and a camera captures the emitted signals to determine the sequence of each fragment through multiple sequencing cycles. Finally, in alignment and data analysis, the short DNA reads are aligned to a reference genome, overlapping regions are assembled into contigs, and the final sequence is reconstructed. This efficient and scalable technique is widely used in genomic research, cancer studies, transcriptomics, and personalized medicine.
        """
    )
    return


@app.cell
def _(mo):
    mo.image(src="NGS.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In this data the gene expression is normalized using FPKM and TPM where value of matrix represent the ratio of read per killobase or million. 

        \[
        FPKM = \frac{\text{Reads per gene}}{\text{Gene length (kb)} \times \text{Total mapped reads (millions)}}
        \]

        and

        \[
        \text{RPK} = \frac{\text{Reads per gene}}{\text{Gene length (kb)}}
        \]

        and finaly 

        \[
        TPM = \frac{\text{RPK}}{\sum \text{RPK}} \times 10^6
        \]

        Our research sujest TPM (Transcripts Per Million) over FPKM (Fragments Per Kilobase per Million reads) because it ensures consistency across samples by normalizing for gene length before sequencing depth. Unlike FPKM, TPM guarantees that the total expression across all genes sums to one million, making values directly comparable across different datasets. This prevents biases from varying sequencing depths and avoids inflation of expression values in smaller samples. Additionally, TPM corrects for discrepancies in lowly expressed genes by proportionally scaling read counts, leading to more accurate cross-sample comparisons. This means this data can be used to classify any gene expression samples as long as they are within the 24 cancer type that this dataset represent including breast cancer.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### <span style="color: green">Data Files in GSE62944

        These are the available suplementory file for this geo dataset that we can use for labaling and creating our data set in csv format and furthermore creating Test/Train/Val data: 

        | Supplementary file | Discription |File type/resource |
        |--------------------|------|--------------------|
        | GSE62944_01_27_15_TCGA_20_420_Clinical_Variables_7706_Samples.txt.gz | 548 clinical variables for each sample are provided in the | TXT |
        | GSE62944_01_27_15_TCGA_20_CancerType_Samples.txt.gz | list of mutated samples | TXT |
        | GSE62944_06_01_15_TCGA_24_548_Clinical_Variables_9264_Samples.txt.gz | 548 clinical variables for each sample are provided in the | TXT |
        | GSE62944_06_01_15_TCGA_24_CancerType_Samples.txt.gz | describes 24 cancers types by sample | TXT |
        | GSE62944_06_01_15_TCGA_24_Normal_CancerType_Samples.txt.gz | list each sample normal samples respectively | TXT |
        | GSE62944_RAW.tar | 5.9 Gb of raw data | TAR (of TXT) |
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""<span style="color: brown">1. EDA for 24 cancer type file</span>""")
    return


@app.cell
def _():
    import pandas as pd
    def create_supl_df(source, cancerType):
        """
            this function creates data frame from a suplementart file
            inout: suplementery file path, cancerType of interes
            returns -> a data_frame

        """

        #read suplementory file    
        with open(source, 'r') as file:
            data=[line.split('\t') for line in file.readlines()]

        data_dic = {"Samples":[], "CancerType":[]}
        for line in data:
            data_dic['Samples'].append(line[0])
            data_dic['CancerType'].append(line[1])
        cancer_type_df = pd.DataFrame(data_dic)
        cancer_type_df['CancerType']=cancer_type_df['CancerType'].str.strip()
        cType_count = cancer_type_df.CancerType.value_counts()
        cancer_type_df = cancer_type_df[cancer_type_df['CancerType']==cancerType]
        return cancer_type_df, data, cType_count
    return create_supl_df, pd


@app.cell
def _(create_supl_df):
    cancerBRCA_type_df, data, cType_count= create_supl_df('../../data/SuplementoryFiles/TCGA_24_CancerType_Samples.txt', 'BRCA')
    return cType_count, cancerBRCA_type_df, data


@app.cell(hide_code=True)
def _(mo):
    mo.md("""* These are the unique cancer types in dataset""")
    return


@app.cell
def _(cType_count):
    cType_count
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        and ther discription: 

        | Cander Type | Description |
        |-------------|-------------|
        | "PRAD" | "Prostate Adenocarcinoma" |
        | "LGG" | "Low-Grade Glioma" |
        | "OV" | "Ovarian Serous Cystadenocarcinoma" |
        | "BRCA"| "Breast Invasive Carcinoma" |
        | "SKCM"| "Skin Cutaneous Melanoma" |
        | "LUAD"| "Lung Adenocarcinoma" |
        | "LUSC"| "Lung Squamous Cell Carcinoma" |
        | "THCA"| "Thyroid Carcinoma" |
        | "UCEC"| "Uterine Corpus Endometrial Carcinoma" |
        | "KIRP"| "Kidney Renal Papillary Cell Carcinoma" |
        | "HNSC"| "Head and Neck Squamous Cell Carcinoma" |
        | "CESC"| "Cervical Squamous Cell Carcinoma and Endocervical Adenocarcinoma" |
        | "COAD"| "Colon Adenocarcinoma" |
        |  "ACC"| "Adrenocortical Carcinoma" |
        | "KIRC"| "Kidney Renal Clear Cell Carcinoma" |
        |  "GBM"| "Glioblastoma Multiforme" |
        | "BLCA"| "Bladder Urothelial Carcinoma"|
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <span style="color: brown"> 2. the Normal_CancerType_Samples</spam>

        This text file list each normal samples respectively! lets take a look
        """
    )
    return


@app.cell
def _(create_supl_df):
    Normal_Sample_gene,_,_ = create_supl_df('../../data/SuplementoryFiles/TCGA_Normal_CancerType_Samples .txt','BRCA')
    Normal_Sample_gene
    return (Normal_Sample_gene,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""As we can see there are 113 normal cancer samples where i moght do [%60, %20, %20] distribution for train test and val datasets witch would be [67, 23, 23] samples for healthy dataset!""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <span style="color: brown"> 3. the Malignant_CancerType_Samples</spam>

        list of mutated samples!
        """
    )
    return


@app.cell
def _(create_supl_df):
    melignent_df, _, _=create_supl_df('../../data/SuplementoryFiles/TCGA_20_CancerType_Samples.txt', 'BRCA')
    melignent_df
    return (melignent_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # <span style='color: brown'>Raw Gene Expression Data</sapn>

        now that we have all the samples data for healthy and cancerous data GSE62944_RAW.tar data.

        | File Name | Sample ID | Date | Platform | Condition | File Type | Use|
        |-----------|----------|------|----------|-----------|-----------|-----------|
        | GSM1536837_01_27_15_TCGA_20.Illumina.tumor_Rsubread_FeatureCounts.txt | GSM1536837 | 01_27_15 | Illumina | Tumor | FeatureCounts |  |
        | GSM1536837_01_27_15_TCGA_20.Illumina.tumor_Rsubread_FPKM.txt | GSM1536837 | 01_27_15 | Illumina | Tumor | FPKM | |
        | GSM1536837_01_27_15_TCGA_20.Illumina.tumor_Rsubread_TPM.txt | GSM1536837 | 01_27_15 | Illumina | Tumor | TPM | |
        | GSM1536837_06_01_15_TCGA_24.tumor_Rsubread_FeatureCounts.txt | GSM1536837 | 06_01_15 | TCGA | Tumor | FeatureCounts | |
        | GSM1536837_06_01_15_TCGA_24.tumor_Rsubread_FPKM.txt | GSM1536837 | 06_01_15 | TCGA | Tumor | FPKM | |
        | GSM1536837_06_01_15_TCGA_24.tumor_Rsubread_TPM.txt | GSM1536837 | 06_01_15 | TCGA | Tumor | TPM | ✅ |
        | GSM1697009_06_01_15_TCGA_24.normal_Rsubread_FeatureCounts.txt | GSM1697009 | 06_01_15 | TCGA | Normal | FeatureCounts | |
        | GSM1697009_06_01_15_TCGA_24.normal_Rsubread_FPKM.txt | GSM1697009 | 06_01_15 | TCGA | Normal | FPKM | |
        | GSM1697009_06_01_15_TCGA_24.normal_Rsubread_TPM.txt | GSM1697009 | 06_01_15 | TCGA | Normal | TPM | ✅ |

        Since the gene expressions are in text format I need to as well extract the gene expression matrix parsing through data!
        """
    )
    return


@app.cell
def _(pd):
    def get_samples_df(source, sample_list):
        """
            this function takes in the file source and sample list of BRCA
            then return the data frame that contains the all the genes and related benign and melignant samples
        """
        with open(source, 'r') as file:
            extracted_data = [line.strip().split("\t") for line in file]
        df=pd.DataFrame(extracted_data)
        # Extract first row as column names and shift right
        new_columns = [""] + df.iloc[0, :-1].tolist()  # Shift column names one position right

        # Drop the first row since it's now the header
        df = df.iloc[1:].reset_index(drop=True)

        # Assign the new column names
        df.columns = new_columns

        #we have all the data we need 
        #lest extract the samples that are relate to normal or melignant dataset

        df=df.set_index(df.columns[0])
        df=df.loc[:,sample_list]
        return df
    return (get_samples_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""###1. Normal Samples from GSM1697009_06_01_15_TCGA_24.normal_Rsubread_TPM.txt:""")
    return


@app.cell
def _(Normal_Sample_gene, get_samples_df):
    healthy_dataSet = get_samples_df('../../data/GSE62944_RAW/GSM1697009_06_01_15_TCGA_24.normal_Rsubread_TPM.txt',Normal_Sample_gene['Samples'].to_list())
    healthy_dataSet=healthy_dataSet.T
    healthy_dataSet=healthy_dataSet.astype(float)
    # healthy_dataSet.to_csv('../../data/ModelDataSets/helthyExpressions.csv')
    return (healthy_dataSet,)


@app.cell
def _(healthy_dataSet, mo):
    mo.ui.table(healthy_dataSet)
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
def _(healthy_dataSet, mo, pd):
    import plotly.express as px
    # Extract genes and compute mean expression
    genes = healthy_dataSet.columns
    average = healthy_dataSet.mean(axis=0)  # Compute mean across all samples

    # Create a new DataFrame with gene names and their average expression
    plot_df = pd.DataFrame({'Genes': genes, 'avg_expr_level': average})

    # Sort and select the top 300 genes by expression level
    plot_df = plot_df.sort_values(by='avg_expr_level', ascending=False).iloc[:300]

    # Create an interactive bar plot
    fig = mo.ui.plotly(
        px.bar(plot_df, x='Genes', y='avg_expr_level',
        title="Gene Expression Visualization for Top 300 Genes",
        labels={'avg_expr_level': 'Mean Expression Level'},
        color='avg_expr_level'  # Optional: color for better visualization
        ).update_layout(xaxis_tickangle=-45)
    )

    # Rotate x-axis labels for better visibility
    fig
    return average, fig, genes, plot_df, px


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ###2. Melignant samples Samples from GSM1697009_06_01_15_TCGA_24.normal_Rsubread_TPM.txt:

        Now we are doing the same thing for toumor samples!
        """
    )
    return


@app.cell
def _(pd):
    """
    df1 =get_samples_df('../../data/GSE62944_RAW/GSM1536837_06_01_15_TCGA_24.tumor_Rsubread_TPM.txt', cancerBRCA_type_df['Samples'].to_list())
    df2 =get_samples_df('../../data/GSE62944_RAW/GSM1536837_06_01_15_TCGA_24.tumor_Rsubread_TPM.txt', melignent_df['Samples'].to_list())
    cancer_dataSet = pd.concat([df1.T,df2.T],ignore_index=False)
    cancer_dataSet=cancer_dataSet.astype(float)
    """

    cancer_dataSet = pd.read_csv('../../data/ModelDataSets/cancerExpressions.csv')
    return (cancer_dataSet,)


@app.cell
def _(cancer_dataSet, mo):
    cancer_dataSet.rename(columns={'Unnamed: 0':'Samples'}, inplace=True)
    mo.ui.table(cancer_dataSet)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""It seams like we have 2,200 samples and for the model training using 60,20,20 distribution. We are going to have [1321, 350, 350] sample distribution for test train and val dataset!""")
    return


@app.cell(hide_code=True)
def _(cancer_dataSet, mo, pd, plot_df, px):
    plot_genes=plot_df['Genes'].tolist()
    average2 = cancer_dataSet[plot_genes].mean(axis=0)
    plot2_df= pd.DataFrame({'Genes':plot_genes,'avg_expr_level':average2})
    fig2 = mo.ui.plotly(
        px.bar(plot2_df, x='Genes', y='avg_expr_level',
        title="Gene Expression Visualization for Top 300 Genes",
        labels={'avg_expr_level': 'Mean Expression Level'},
        color='avg_expr_level'  # Optional: color for better visualization
        ).update_layout(xaxis_tickangle=-45)
    )

    # Rotate x-axis labels for better visibility
    fig2
    return average2, fig2, plot2_df, plot_genes


@app.cell(hide_code=True)
def _(pd, plot2_df, plot_df):
    plot_df.reset_index(drop=True,inplace=True)
    plot_df.rename(columns={'avg_expr_level':'normal_avg_expr_level'}, inplace=True)
    plot2_df.reset_index(drop=True,inplace=True)
    plot2_df.rename(columns={'avg_expr_level':'malignant_avg_expr_level'}, inplace=True)
    merged_df=pd.merge(plot_df, plot2_df, on='Genes')
    merged_df['Mean_Expression_absolute_diff']=abs(merged_df['normal_avg_expr_level'] - merged_df['malignant_avg_expr_level'])
    merged_df=merged_df.sort_values(by='Mean_Expression_absolute_diff', ascending=False)
    merged_df.head(30)
    return (merged_df,)


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


@app.cell
def _(mo):
    mo.image('AHPexplnation.png', caption='The hierarchy of factors for gene selection by AHP.')
    return


@app.cell
def _(pd):
    ahp_df= pd.read_csv('Mod_ahp_scores.csv')
    # mo.ui.table(ahp_df.sort_values(by='Scores', ascending=False))
    return (ahp_df,)


@app.cell(hide_code=True)
def _(ahp_df, mo):
    import altair as alt
    # Ensure Altair handles large datasets
    alt.data_transformers.enable("vegafusion")
    # Sort and take top 300 genes
    ahp_top = ahp_df.sort_values(by='Scores', ascending=False).iloc[:1000,:]

    # Ensure "Gene" column is retained and treated as string
    ahp_top_scaled = ahp_top.copy()
    ahp_top_scaled.iloc[:, 1:] *= 1e6  # Apply scaling
    ahp_top_scaled['Gene'] = ahp_top_scaled['Gene'].astype(str)

    # Selection for interactive brushing
    brush = alt.selection_interval(encodings=['x', 'y'])

    # Scatter Plot (Interactive)
    scatter = alt.Chart(ahp_top_scaled).mark_circle(size=60).encode(
        x='Scores:Q',
        y='t_test:Q',
        tooltip=['Gene:N', 'Scores:Q', 't_test:Q', 'entropy:Q', 'roc_auc:Q', 'snr:Q'],
        color=alt.condition(brush, alt.value('steelblue'), alt.value('lightgray'))
    ).add_params(brush).properties(
        width=700,
        height=400,
        title="Gene Scores vs. t-test"
    )

    # Get top 10 genes for default view
    top_10_genes = ahp_top_scaled.nlargest(10, 'Scores')

    plot_to_show = top_10_genes if not brush else ahp_top_scaled
    # Bar Chart (t_test & entropy) with only top 10 genes before selection
    bar = (
        alt.Chart(ahp_top_scaled)
        .transform_fold(['t_test', 'entropy'], as_=['Metric', 'Value'])
        .mark_bar()
        .encode(
            x=alt.X('Gene:N', sort='-y'),
            y=alt.Y('Value:Q'),
            color='Metric:N',
            tooltip=['Gene:N', 'Metric:N', 'Value:Q']
        )
        .transform_filter(
            # Show any gene that the brush has selected (brush)
            # OR any gene in the top 10 (FieldOneOfPredicate)
            brush | alt.FieldOneOfPredicate(field='Gene', oneOf=top_10_genes['Gene'].tolist())
        )
        .properties(width=700, height=200, title="t_test & Entropy for Selected Genes")
    )

    # Table that dynamically updates on selection

    def gene_table():
        selected_genes = ahp_top_scaled.loc[ahp_top_scaled['Gene'].isin(top_10_genes['Gene'])]

        if not brush.empty:
            selected_genes = ahp_top_scaled  # Show selected genes when brush is used

        return mo.ui.table(selected_genes)

    table = gene_table()

    # Combine all charts and table
    interactive_chart = alt.vconcat(scatter, bar)
    mo.vstack([interactive_chart, table])
    return (
        ahp_top,
        ahp_top_scaled,
        alt,
        bar,
        brush,
        gene_table,
        interactive_chart,
        plot_to_show,
        scatter,
        table,
        top_10_genes,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## <span style="color: green">PairWize MAtrix Visualization""")
    return


@app.cell
def _():
    import pickle
    # from scipy.sparse import csr_matrix
    # # Open the large Pickle file without fully loading it
    # with open("pwm_pickl.pkl", "rb") as f:
    #     while True:
    #         try:
    #             pwm_chunk = pickle.load(f)  # Load one pairwise matrix at a time
    #             for key, sparse_matrix in pwm_chunk.items():
    #                 print(f"Processing sparse matrix for: {key}")
    #                 sparse_matrix = csr_matrix(sparse_matrix)  # Convert if necessary
    #         except EOFError:
    #             break  # Stop when end of file is reached
    with open("pwm_pickl.pkl", "rb") as fs:
        pwm = pickle.load(fs)
        print(pwm.keys())
        #'t_test', 'entropy', 'roc_auc', 'snr
        t_test_matrix = pwm['t_test']
        entropy_matrix = pwm['entropy']
        roc_matrix = pwm['roc_auc']
        snr_matrix = pwm['snr']
    return (
        entropy_matrix,
        fs,
        pickle,
        pwm,
        roc_matrix,
        snr_matrix,
        t_test_matrix,
    )


@app.cell
def _(
    ahp_df,
    ahp_top,
    entropy_matrix,
    mo,
    pd,
    px,
    roc_matrix,
    snr_matrix,
    t_test_matrix,
):
    # Convert sparse matrix to dense format
    t_test_dense_matrix = t_test_matrix.toarray()  # Convert sparse to dense numpy array
    entropy_dense_matrix = entropy_matrix.toarray()
    roc_dense_matrix = roc_matrix.toarray()
    snr_dense_matrix = snr_matrix.toarray()
    # top genes 
    top_genes = ahp_top.Gene.tolist()[:101]
    # Create a DataFrame with gene names as both index and columns
    t_test_pairwise_df = pd.DataFrame(t_test_dense_matrix, index=ahp_df.Gene, columns=ahp_df.Gene).loc[top_genes, top_genes]
    entropy_pairwise_df = pd.DataFrame(entropy_dense_matrix, index=ahp_df.Gene, columns=ahp_df.Gene).loc[top_genes, top_genes]
    roc_pairwise_df = pd.DataFrame(roc_dense_matrix, index=ahp_df.Gene, columns=ahp_df.Gene).loc[top_genes, top_genes]
    snr_pairwise_df = pd.DataFrame(snr_dense_matrix, index=ahp_df.Gene, columns=ahp_df.Gene).loc[top_genes, top_genes]

    # Replace NaN values (if any) with 0
    def creat_heatmap(df):
        fig = px.imshow(df)
        return mo.ui.plotly(fig)
    mo.ui.tabs(
        {
            "T_Test": mo.vstack([creat_heatmap(t_test_pairwise_df), mo.ui.table(t_test_pairwise_df)]),
            "Entropy": mo.vstack([creat_heatmap(entropy_pairwise_df), mo.ui.table(entropy_pairwise_df)]),
            "ROC": mo.vstack([creat_heatmap(roc_pairwise_df), mo.ui.table(roc_pairwise_df)]),
            "SNR": mo.vstack([creat_heatmap(snr_pairwise_df), mo.ui.table(snr_pairwise_df)])
        }
    )
    return (
        creat_heatmap,
        entropy_dense_matrix,
        entropy_pairwise_df,
        roc_dense_matrix,
        roc_pairwise_df,
        snr_dense_matrix,
        snr_pairwise_df,
        t_test_dense_matrix,
        t_test_pairwise_df,
        top_genes,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ##<span style="color: brown">Clinical_Variables Suplemantory file:<span>

        This suplementary file gives cancer related senses such as **age, gender, tebaco and alchohol consumption, and family historyy, ...**.

        we need to analy how does each of these corolates with the cancer status.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, pd):
    with open('../../data/SuplementoryFiles/GSE62944_01_27_15_TCGA_20_420_Clinical_Variables_7706_Samples.txt', 'r') as file:
        exractedData1=[line.split('\t') for line in file.readlines()]
    with open('../../data/SuplementoryFiles/GSE62944_06_01_15_TCGA_24_548_Clinical_Variables_9264_Samples.txt', 'r') as file:
        exractedData2=[line.split('\t') for line in file.readlines()]

    # Convert to DataFrames
    df1 = pd.DataFrame(exractedData1)
    df2 = pd.DataFrame(exractedData2)

    # Transpose the DataFrames
    df1 = df1.T
    df2 = df2.T

    # Concatenate
    concat_df = pd.concat([df1, df2])

    # Rename columns using the first row
    concat_df.columns = concat_df.iloc[0]
    concat_df = concat_df.drop(0)  # Remove the first row

    #remove nan colomns
    concat_df = concat_df.loc[:, ~concat_df.columns.isna()]

    # Rename index column
    if '' in concat_df.columns:
        concat_df.rename(columns={'': 'Samples'}, inplace=True)

    # Display the DataFrame
    mo.ui.table(concat_df)
    return concat_df, df1, df2, exractedData1, exractedData2, file


@app.cell(hide_code=True)
def _(mo):
    mo.md("""After Creating a data fram from the two <span style="color: brown">two clinical suplimentary files</span>, now we have to extract all the BRCA data from it and the following table displays all the brca related logistics. Now we can delve in furthure to create a dataset for stage prognosis and furthermore analyze patients demografic and see how they are corolated.""")
    return


@app.cell(hide_code=True)
def _(cancer_dataSet, concat_df, healthy_dataSet, mo):
    healthyBRCAList = healthy_dataSet.index
    cancerBRCAList = cancer_dataSet['Samples']

    BRCA_CV = concat_df[concat_df['Samples'].isin(healthyBRCAList) | concat_df['Samples'].isin(cancerBRCAList)].copy()
    # BRCA_CV.drop(columns='index', inplace=True)
    mo.ui.table(BRCA_CV)
    return BRCA_CV, cancerBRCAList, healthyBRCAList


@app.cell(hide_code=True)
def _(BRCA_CV):
    BRCA_CV['tumor_status'].value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""as we can see not all tumor status for the samples are properly labled. thus we are going to relable them.""")
    return


@app.cell
def _(BRCA_CV, cancerBRCAList, healthyBRCAList):
    def label_sample(samples):
        stats_list=[]
        for smpl in samples:
            if smpl in healthyBRCAList:
                stats_list.append("Normal")
            elif smpl in cancerBRCAList:
                stats_list.append("Malignant")
            else:
                stats_list.append(None)
        return stats_list
    BRCA_CV["Status"] = label_sample(BRCA_CV['Samples'].tolist())

    BRCA_CV["Status"]
    return (label_sample,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ###<span style="color:green">Medical Terminalogies for Cancer Staging</span>

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
    mo.image(src='BRCA_StageGrouping.png', caption='From AJCC cancer staging manual. 6th Edition. New York: Springer-Verlag, 2002 with permission.')
    return


@app.cell
def _(BRCA_CV):
    stage_col=[col for col in BRCA_CV.columns if str.find(col, 'ajcc')>=0]
    # stage_col
    patient_dmg=['Samples',
                 'age_at_diagnosis',
                 'gender', 
                 'race', 
                 'ethnicity', 
                 'family_history_cancer_indicator',
                 'family_history_cancer_type',
                 'family_history_cancer_relationship',
                 'history_other_malignancy',
                 'history_neoadjuvant_treatment',
                 'history_colon_polyps',
                 'clinical_M',
                 'clinical_N',
                 'clinical_T',
                 'pathologic_M'
                ]
    return patient_dmg, stage_col


@app.cell(hide_code=True)
def _(mo, pd):
    # Function to create tables from lists
    def create_table(data):
        df = pd.DataFrame(data, columns=["Feature", "Description"])
        return mo.ui.table(df)

    # Define tables for each category
    patient_demographics_table = create_table([
        ["age_at_diagnosis", "Age at the time of cancer diagnosis"],
        ["gender", "Gender of the patient"],
        ["race", "Racial background of the patient"],
        ["ethnicity", "Ethnic background of the patient"],
        ["family_history_cancer_indicator", "Whether there is a family history of cancer"],
        ["family_history_cancer_type", "Type of cancer in the family history"],
        ["family_history_cancer_relationship", "Relationship of family members with cancer"],
        ["history_other_malignancy", "History of other malignancies"],
        ["history_neoadjuvant_treatment", "Prior neoadjuvant treatment history"],
        ["history_colon_polyps", "History of colon polyps"],
        ["history_hematologic_disorder", "History of hematologic disorders"],
        ["history_exposure_leukemogenic_agents", "Exposure to leukemogenic agents"],
        ["history_chemical_exposure_other", "Exposure to other chemical agents"],
        ["history_radiation_exposure", "History of radiation exposure"],
        ["history_hormonal_contraceptives_use", "Use of hormonal contraceptives"],
        ["history_menopausal_hormone_therapy", "Use of menopausal hormone therapy"],
        ["history_tamoxifen_use", "Use of tamoxifen"]
    ])

    tumor_characteristics_table = create_table([
        ["tumor_status", "Status of the tumor"],
        ["tumor_type", "Type of tumor"],
        ["tumor_tissue_site", "Location of the tumor"],
        ["anatomic_neoplasm_subdivision", "Anatomical subdivision of the tumor"],
        ["clinical_stage", "Clinical stage of cancer"],
        ["histologic_diagnosis", "Histologic diagnosis"],
        ["histologic_subtype", "Histologic subtype of the tumor"],
        ["neoplasm_histologic_grade", "Histologic grade of the neoplasm"],
        ["tumor_size_width", "Width of the tumor"],
        ["tumor_grade", "Grade of the tumor"],
        ["icd_10", "ICD-10 classification"],
        ["icd_o_3_histology", "ICD-O-3 histologic classification"],
        ["icd_o_3_site", "ICD-O-3 site classification"],
        ["gleason_pattern_primary", "Primary Gleason pattern (for prostate cancer)"],
        ["gleason_pattern_secondary", "Secondary Gleason pattern"],
        ["gleason_score", "Overall Gleason score"]
    ])

    cancer_staging_table = create_table([
        ["ajcc_staging_edition", "AJCC staging edition"],
        ["ajcc_tumor_pathologic_pt", "Pathologic primary tumor stage"],
        ["ajcc_nodes_pathologic_pn", "Pathologic lymph node stage"],
        ["ajcc_metastasis_pathologic_pm", "Pathologic metastasis stage"],
        ["ajcc_pathologic_tumor_stage", "Overall pathologic tumor stage"],
        ["ajcc_clinical_tumor_stage", "Overall clinical tumor stage"],
        ["clinical_M", "Clinical metastasis stage"],
        ["clinical_N", "Clinical lymph node stage"],
        ["clinical_T", "Clinical tumor stage"],
        ["pathologic_M", "Pathologic metastasis stage"],
        ["pathologic_N", "Pathologic lymph node stage"],
        ["pathologic_T", "Pathologic tumor stage"],
        ["pathologic_stage", "Overall pathologic stage"]
    ])

    metastasis_table = create_table([
        ["metastatic_tumor_indicator", "Indicates metastatic tumor presence"],
        ["metastasis_site", "Site of metastasis"],
        ["extracapsular_extension", "Presence of extracapsular extension"],
        ["extracapsular_extension_present", "Whether extracapsular extension is present"],
        ["lymphovascular_invasion", "Presence of lymphovascular invasion"],
        ["vascular_invasion", "Presence of vascular invasion"],
        ["lymph_nodes_examined", "Number of lymph nodes examined"],
        ["lymph_nodes_examined_count", "Count of examined lymph nodes"],
        ["lymph_nodes_examined_positive", "Number of positive lymph nodes"],
        ["perineural_invasion", "Presence of perineural invasion"]
    ])

    treatment_data_table = create_table([
        ["radiation_treatment_adjuvant", "Adjuvant radiation therapy"],
        ["pharmaceutical_tx_adjuvant", "Adjuvant pharmaceutical treatment"],
        ["treatment_outcome_first_course", "Outcome of the first course of treatment"],
        ["targeted_molecular_therapy", "Targeted molecular therapy"],
        ["definitive_surgical_procedure", "Definitive surgical procedure performed"],
        ["surgical_procedure_first", "First surgical procedure performed"],
        ["margin_status", "Surgical margin status"],
        ["surgery_for_positive_margins", "Surgery performed for positive margins"],
        ["chemotherapy", "Chemotherapy treatment"],
        ["hormonal_therapy", "Hormonal therapy treatment"],
        ["immunotherapy", "Immunotherapy treatment"],
        ["ablation_embolization_tx_adjuvant", "Ablation or embolization therapy"]
    ])

    genetic_mutations_table = create_table([
        ["kras_gene_analysis_indicator", "Indicator of KRAS gene analysis"],
        ["kras_mutation_found", "KRAS mutation found"],
        ["kras_mutation_codon", "KRAS mutation codon"],
        ["braf_gene_analysis_indicator", "Indicator of BRAF gene analysis"],
        ["braf_gene_analysis_result", "BRAF gene analysis result"],
        ["p53_gene_analysis", "p53 gene analysis result"],
        ["egfr_mutation_status", "EGFR mutation status"],
        ["egfr_mutation_identified_type", "Type of EGFR mutation identified"],
        ["eml4_alk_translocation_status", "EML4-ALK translocation status"],
        ["idh1_mutation_test_indicator", "IDH1 mutation test indicator"],
        ["idh1_mutation_found", "IDH1 mutation found"],
        ["her2_status_by_ihc", "HER2 status determined by IHC"]
    ])

    # Create tabs using a dictionary
    tabs = mo.ui.tabs({
        "Patient Demographics & History": patient_demographics_table,
        "Tumor Characteristics & Diagnosis": tumor_characteristics_table,
        "Cancer Staging":cancer_staging_table,
        "Metastasis Cancer Data": metastasis_table,
        "Treatments Data": treatment_data_table,
        "Genetic Mutations": genetic_mutations_table

    })

    tabs
    return (
        cancer_staging_table,
        create_table,
        genetic_mutations_table,
        metastasis_table,
        patient_demographics_table,
        tabs,
        treatment_data_table,
        tumor_characteristics_table,
    )


@app.cell
def _(BRCA_CV, mo):
    mo.ui.data_explorer(BRCA_CV)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ###<span style="color: Green">Use correlation matrices to find relationships between genes and patient features.</span>

        I would like to see the correlation between the cancerous and non cancerous genes and patiens demographic
        """
    )
    return


if __name__ == "__main__":
    app.run()
