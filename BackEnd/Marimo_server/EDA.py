import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import io
    import scipy.stats as stats
    import plotly.graph_objects as go
    import pickle as pkl
    import pandas as pd
    return io, mo, pd, pkl


@app.cell(hide_code=True)
def _():
    import plotly.io as pio
    pio.renderers.default = "iframe_connected"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # <span style='color:brown'>GeneScope and Breast Cancer</span>

    ### <span style='color:brown'>Objective:</span>

    By training a deep learning model, optimized through the backpropagation algorithm, we are aimong to further analyze biomarkers and prognoses. Cancer biomarkers are biological indicators, such as genes, proteins, or other substances, that can reveal important details about a person's cancer. We are particularly interested in exploring whether the model can help identify genes that are more susceptible to cancer, causing mutations (<span style='color:brown'>indicating the bio markers</span>). For prognosis, we are intrested to how factors such as tumor size, the number of regional lymph nodes affected, distant metastasis, and patient demographics—such as age, ethnicity and race, play a critical role in cancer progression and patient outcomes(<span style='color:brown'>Prognosis</span>).

    ### <span style='color:brown'>Why Gene Expression Matters in Cancer Research</span>

    Genes carry the instructions for making proteins, which are the functional molecules responsible for nearly every biological process in the body—from cell division and repair to immune responses and tissue structure. Each cell selectively expresses certain genes to produce the specific proteins it needs, shaping the behavior and identity of that cell. When gene expression becomes dysregulated—either overactive (overexpression) or underactive (underexpression)—it can lead to abnormal protein levels. This imbalance can disrupt normal cellular function and trigger mutations, uncontrolled growth, or even metastasis, all of which are hallmarks of cancer. Thus studying Gene expression can provide a valuable inoformation that scientists could laverage for treatment strategies. 

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## <span style="color: brown">Expression Collection Methods: 

    The gene expressoions have been extracted using **Next-Generation Sequencing (NGS)**, a high-throughput method for sequencing DNA by following a series of steps. First, in library preparation, the genome is fragmented into smaller pieces, and short adapter sequences are attached to both ends, forming a DNA library. Next, in bridge amplification, these fragments are hybridized to a solid surface and undergo multiple amplification cycles, creating clusters of identical DNA fragments. In sequencing, fluorescently labeled nucleotides are incorporated into the DNA strands, and a camera captures the emitted signals to determine the sequence of each fragment through multiple sequencing cycles. Finally, in alignment and data analysis, the short DNA reads are aligned to a reference genome, overlapping regions are assembled into contigs, and the final sequence is reconstructed. This efficient and scalable technique is widely used in genomic research and cancer studies.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(src="NGS.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    # <span style="color:brown">Data Collection and Sources</span>
    ###<span style="color:brown">Challenges</span>

    Collecting data for our cancer study turned out to be much harder than we expected. Cancer is such a complex and heterogeneous disease that finding enough consistent data was a real challenge. We often ran into two major problems. Sometimes the datasets we found were high quality but only had around 200 samples — not nearly enough for the kind of deep analysis we wanted to do. Other times, even when more data was available, the methods used for gene extraction and normalization were completely different from one study to another. That made it impossible to simply combine multiple datasets without risking serious errors. Overall, getting a large, clean, and consistent dataset proved to be one of the biggest challenges in our project.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    ### <span style="color:brown">Gene Expression Data Source</span>

    The Gene Expression dataset has been extracted from the [NCBI Gene Expression Omnibus GSEGSE62944](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE62944). The dataset consists of 9264 tumor samples and 741 normal samples across 24 cancer types from The Cancer Genome Atlas. This data set is created from all the TCGA samples that have an expression of 23,000 genes!
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(mo.image(src="./img/TCGAcancerTypes.png",caption="The Cancer Types available in TCGA database. This image is from the TCGA officcial website!", width=500,  height=400))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""According to [NCBI]("") 20 years after the original publication of the human genome, the number of protein-coding genes is stabilizing around 19,500 (Figure 2), although the number of isoforms of these genes is still a subject of intensive study and discussion.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The gene expression data is normalized using FPKM and TPM where value of matrix represent the ratio of read per killobase or million. 

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

    This means we dont need to do any normalization, and make the data preprocessing easier. Our research sujest TPM (Transcripts Per Million) over FPKM (Fragments Per Kilobase per Million reads) because it ensures consistency across samples by normalizing for gene length before sequencing depth. Unlike FPKM, TPM guarantees that the total expression across all genes sums to one million, making values directly comparable across different datasets. This prevents biases from varying sequencing depths and avoids inflation of expression values in smaller samples. Additionally, TPM corrects for discrepancies in lowly expressed genes by proportionally scaling read counts, leading to more accurate cross-sample comparisons. This means this data can be used to classify any gene expression samples as long as they are within the 24 cancer type that this dataset represent including breast cancer.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    # <span style="color: brown">Data Extraction and EDA</span>
    ### <span style="color: brown">Supplementary Data Files in GSE62944</span>

    The publisher of dataset provided the following supplementary files for extracting and labling gene exoressions.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    sup_table = mo.md("""
    | Supplementary file | Discription |File type/resource |
    |--------------------|------|--------------------|
    | GSE62944_01_27_15_TCGA_20_420_Clinical_Variables_7706_Samples.txt.gz | 548 clinical variables for each sample are provided in the | TXT |
    | GSE62944_01_27_15_TCGA_20_CancerType_Samples.txt.gz | list of mutated samples | TXT |
    | GSE62944_06_01_15_TCGA_24_548_Clinical_Variables_9264_Samples.txt.gz | 548 clinical variables for each sample are provided in the | TXT |
    | GSE62944_06_01_15_TCGA_24_CancerType_Samples.txt.gz | describes 24 cancers types by sample | TXT |
    | GSE62944_06_01_15_TCGA_24_Normal_CancerType_Samples.txt.gz | list each sample normal samples respectively | TXT |
    | GSE62944_RAW.tar | 5.9 Gb of raw data | TAR (of TXT) |
    """)
    gene_Raw_file = mo.md("""
    After extracting all the refrencing and maping of all the sample IDs, we can leverage them to create healthy and cancerous data_set.

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
    """)

    mo.ui.tabs({
        "Suplementary Files List": sup_table,
        "Raw-Gene-Expressiong Datas": gene_Raw_file
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <span style="color:brown">Step-by-Step: Cleaning and Extracting Gene Expression Data</span>

    To ensure the quality and usability of the gene expression data, we followed a structured process combining supplementary file references and raw TPM expression matrices:

    🧬 Step-by-Step Workflow
    1. Load Supplementary Files

    Loaded TCGA_24_CancerType_Samples.txt to identify and isolate breast cancer samples (labeled as BRCA).

    Used two additional files: one for normal (Normal_CancerType_Samples.txt) and one for malignant (20_CancerType_Samples.txt) BRCA sample IDs.

    2. Extract Sample IDs by Condition

    Parsed and filtered patient sample IDs based on whether they were healthy or cancerous using the create_supl_df() function.

    Created two distinct ID lists: Normal_Sample_gene and melignent_df.

    3. Reference Raw Gene Expression Files

    Mapped sample IDs to raw TPM gene expression files:

    Normal: GSM1697009_06_01_15_TCGA_24.normal_Rsubread_TPM.txt

    Tumor: GSM1536837_06_01_15_TCGA_24.tumor_Rsubread_TPM.txt

    4. Extract Gene Expression Data

    Used a custom parser (get_samples_df) to:

    Load entire gene expression tables.

    Select only columns matching the filtered sample IDs.

    Transpose the matrix so that rows represent patients and columns represent genes.

    Convert values to numeric format.

    5. Clean the Expression Matrices

    Handled inconsistent formatting and column shifts (due to raw file headers).

    Ensured consistent float type across all values to prepare for downstream machine learning.

    6. Export Final Datasets

    Saved two curated dataframes:

    helthyExpressions.csv

    cancerExpressions.csv

    These were used to train models for classification, biomarker identification, and stage prediction.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, pd):
    healthy_data = pd.read_csv('scripts/healthy_trimed.csv')
    canncer_data = pd.read_csv('scripts/cancer_trimed.csv')
    ctype = pd.read_csv('scripts/cType.csv')

    # THEN construct your tab view
    geneEda = mo.ui.tabs({
        "Healthy": healthy_data,
        "Cancer": canncer_data,
        "Cancer Types": ctype,
    })

    geneEda

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ##<span style="color: brown">Clinical_Variables Suplemantory file:<span>

    This suplementary file gives cancer related senses such as **age, gender, tebaco and alchohol consumption, and family historyy, ...**.

    we need to analyze how does each of these corolates with the cancer status.
    """
    )
    return


@app.cell(hide_code=True)
def _(pkl):
    with open('./scripts/pkl_files/clinical1_html.pkl', 'rb') as f:
        clinical_tabs = pkl.load(f)
    clinical_tabs
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""After Creating a data fram from the two <span style="color: brown">two clinical suplimentary files</span>, extracted all the breast cancer related samples and started the cleaning up proccess. Jy by a glance at the headers of data it seamd very promissing but during our data cleanning process we chose a different route.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The following table shows the headers of the columns categories for type of study could be done utilizing them.""")
    return


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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""##<span style="color: brown">Data Clening and Analysis and Labling</span>""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    **Issues with the clinical data provided by the publisher of this dataset.**

    * Lot of columns have more that %80 NaN values.
    * Even after droping all the NaN collumns and rows we notice there were lot of Not Available values in our data. Proving that the clinical data was not filled up properly
    * After converting thos Not available rows to NaN we found out the clinical data of this data source is not sufficient enough to help us do any further analysis.
    * There was not proper labling for cancer stages.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    | Step | Cleaning Action | Purpose |
    |:----|:-----------------|:--------|
    | **1** | Converting the dtypes to the correct ones | All the columns were transfered as string even the integers and NAN values. This allowed me to see the true nan value counts and dtypes |
    | **2** | Dropped columns that had more than 80% Nan Values | After analzing the true data types and numnbe of nan values it was vert numeric-looking columns to numbers | Fix columns where numbers are stored as text, enabling proper analysis and modeling. |
    | **3** | Drop columns after the 189th column | Retain only clinically meaningful columns and remove irrelevant features. |
    | **4** | Keep only malignant samples (melig≠ntdfmelignent_df) | Focus the study specifically on malignant (cancerous) patient cases. |

    The following Tabke show why this clinical variable was not usefull. The are many **Not Available** string data in the data frame so we started to look for a secondary data source.
    """
    )
    return


@app.cell(hide_code=True)
def _(pkl):
    with open('./scripts/pkl_files/kl1.pkl', 'rb') as _f:
        cleaned_up = pkl.load(_f)
    cleaned_up
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #<span style="color: brown">Using a secondary Clinical Variable dataset Directy extracted from TCGA</span>
    It seams like the suplementary files for clinical variables is not structed verywell. How ever using R pakage we extracted a better clinical values from TCGA dataset that could be used the same gene expression the NCBI GEO dataset so instead we are going to use this instead.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <span style="color: brown">Collection Mthode:</span>
    ### Extracting Breast Cancer Data from TCGA Using R

    In order to find an apropriate dataset for our models We utilized TCGAbiol∈ksTCGAbiolinks, an R package designed to query and download cancer genomics data from Genomic Data Commons (GDC). This package enables researchers to access large-scale RNA sequencing (RNA-seq) gene expression data and clinical metadata for breast cancer patients.The Cancer Genome Atlas (TCGA) exists as a pioneering project developed jointly between National Cancer Institute (NCI) and National Human Genome Research Institute (NHGRI). This extensive cancer genomics database represents one of the biggest collections that many researchers can access publicly with containing in-depth information up to 33 different cancer types.  The main objective of TCGA centers on pushing cancer research forward by revealing extensive details regarding genomic modifications and expression patterns with their role in forming and advancing cancer. The dataset includes multiple types of molecular data such as DNA sequencing, RNA sequencing, epigenetic modifications, and clinical metadata, making it a valuable resource prinding a more accurate prognonsis.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    r_script = f"""
    ## Implementation using R

    library(TCGAbiolinks)
    library(SummarizedExperiment)
    library(dplyr)
    # query for fetching RNA-seq data
    query = GDCquery(project = "TCGA-BRCA",
                      data.category = "Transcriptome Profiling",
                      data.type = "Gene Expression Quantification",
                      workflow.type = "STAR - Counts")

    # now downloading the data using the constructed query
    GDCdownload(query)

    # Prepare the data into an R object
    data = GDCprepare(query)
    #retreiving both clinical and gene expression data from the 
    expression_data = assay(data)  
    clinical_data = colData(data)  
    str(clinical_data)
    write.csv(expression_data, file = "brca_expression_data.csv")

    clinical_df <- as.data.frame(clinical_data)
    list_cols <- sapply(clinical_df, is.list)
    clinical_df[list_cols] <- lapply(clinical_df[list_cols], function(x) sapply(x, toString))
    write.csv(clinical_df, file = "brca_clinical_data.csv", row.names = FALSE)
    """
    mo.ui.code_editor(r_script)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""The following Tabs display the extracted Data as well as decription of all the columns their values type and count of their missing vlues.""")
    return


@app.cell(hide_code=True)
def _(io, mo, pd):
    # Load dataset
    other_sup_files = pd.read_csv("head1.csv")

    # Capture DataFrame info as a string
    info_buffer = io.StringIO()
    other_sup_files.info(verbose=True, show_counts=True, buf=info_buffer)
    info_text = info_buffer.getvalue()

    # Create Marimo UI tabs
    mo.ui.tabs({
        "DataSet": mo.ui.table(other_sup_files),
        "Info": mo.ui.code_editor(f"`\n{info_text}\n`")  # Display info in a nicely formatted code block
    })
    return (other_sup_files,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ### <span style="color: brown">DataSet Clean up Proccess</span>

    Many columns in the dataset contain zero non-null values. To avoid unintended data loss during cleaning, we dropped all columns with more than 80% missing values. We also removed rows with missing values in key staging-related columns. Since we are working with biological data, applying imputation methods, such as filling missing values with the mean, can introduce bias and compromise the integrity of downstream analyses. Therefore, we opted for conservative filtering rather than data aggregation.

    ---
    <span style="color: brown">Result after clean up:</span>
    """
    )
    return


@app.cell
def _(other_sup_files):
    thresholds = 0.8
    other_sup_files.dropna(thresh=int(thresholds * len(other_sup_files)), axis=1, inplace=True)
    return


@app.cell(hide_code=True)
def _(io, mo, other_sup_files):
    other_sup_files.dropna(inplace=True)
    buffer = io.StringIO()
    other_sup_files.info(verbose=True, show_counts=True, buf=buffer)
    moded_info = buffer.getvalue()  # Store the captured output as a string

    # Display tabs correctly
    mo.ui.tabs({
        "Cleaned Table": mo.ui.table(other_sup_files.head(100)),
        "Info": mo.ui.code_editor(f"`\n{moded_info}\n`")  # Properly formatted info
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ##<span style="color:brown">Number of stages</span>

    Since our classifier models are labling stage for classification, we wanted to check the number of samples for each data, so we can plan accordingly in advance.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, pd):
    stg_count = pd.read_csv('scripts/pkl_files/stage_count.csv')
    mo.ui.table(stg_count)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Our data looks inbalanced. Researches sujest that to either adjust the weight during training or perform Data Augmentation (with Biological Knowledge) using techniques called **SMOTE** or **(VAE) or GANs**:

    ###<span style="color: brown"> Augmentation technique</span>

    * **SMOTE (Synthetic Minority Over-sampling Technique)**: Creates synthetic samples by interpolating between existing samples of the minority class.

    * **Variational Autoencoders (VAE) or GANs (Generative Adversarial Networks)**: Generate realistic gene expression profiles for under-represented classes.

    * **Resampling**:

    Oversampling: Randomly duplicate samples from the minority classes to increase their representation.

    Undersampling: Randomly remove samples from the majority classes, but this risks losing valuable information.

    * **Data Augmentation Using HMM (Hidden Markov Model)**: 

    Train an HMM on gene expression profiles of the minority classes to generate similar but distinct sequences.

    <span style='color:brown'>Since our data is not time seried, we cant use HMM model.</span>

    Finally we mixed the clinical data and gene expressions. The following is the final data set ready to be splited for train, test, val dataset and be used for furthur analysis.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, pkl):
    with open('./scripts/pkl_files/stage_data.pkl', 'rb') as _f:
        stage = pkl.load(_f)
    mo.ui.table(stage)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""##<span style="color:brown"> Corrolation Analysis of age and Cancer</span>""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The clinical dataset utilized in this study includes patient demographic variables such as **age**, **race**, and **ethnicity**, alongside **prognostic indicators** including tumor **size** and **lymph node** involvement. The objective of this analysis is to quantify the strength of association between these features and two critical clinical outcomes:

    **Cancer Stage**

    **Vital Status (Alive or Deceased)**

    Given the categorical nature of most variables in the dataset, **<span style="color:brown">Cramér's V</span>**
    was employed to assess the association strength. For continuous variables, such as age at diagnosis, **<span style="color:brown">the Pearson correlation coefficient</span>** was utilized.

    Cramér's V is a normalized measure of association based on the Chi-square statistic, suitable for evaluating relationships between two categorical variables. It produces a value between 0 and 1, where:

    * 0 indicates no association.

    * 1 indicates a perfect association.

    ### Cramér's V Formula


    \[
    V = \sqrt{ \frac{ \chi^2 }{ n \times (k - 1) } }
    \]

    Where:

    - \( V \) is Cramér's V, the measure of association strength.
    - \( \chi^2 \) is the Chi-square test statistic computed from the contingency table.
    - \( n \) is the total number of observations in the dataset.
    - \( k \) is the smaller number of categories between the two variables compared.
    """
    )
    return


@app.cell(hide_code=True)
def _(pkl):
    with open('./scripts/pkl_files/corr_plot.pkl', 'rb') as _f:
        corr_fig = pkl.load(_f)
    corr_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ###<span style="color:brown"> Key Observation</span>

    The association analysis revealed that tumor characteristics, particularly primary tumor size (ajcc_pathologic_t), regional lymph node involvement (ajcc_pathologic_n), and distant metastasis (ajcc_pathologic_m), exhibit strong associations with both cancer stage and vital status. This result is consistent with clinical expectations, as the extent of tumor spread is a critical determinant of both disease staging and patient survival outcomes.

    Interestingly, demographic variables such as race and ethnicity also demonstrated measurable associations with clinical outcomes. This suggests potential underlying disparities or biological differences that may influence disease progression and survival, warranting further investigation.

    Additionally, age at diagnosis showed both positive and negative associations with the outcomes: while advancing age generally correlated with poorer vital status (increased mortality), it had a weaker or variable relationship with cancer staging. These findings highlight the complex interplay between biological, demographic, and clinical variables in cancer prognosis.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, pkl):
    with open('./scripts/pkl_files/corr_df.pkl', 'rb') as _f:
        corr_df = pkl.load(_f)
    mo.ui.table(corr_df)
    return


if __name__ == "__main__":
    app.run()
