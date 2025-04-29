import marimo

__generated_with = "0.12.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import io
    import numpy as np
    import scipy.stats as stats
    import plotly.graph_objects as go
    return go, io, mo, np, stats

@app.cell
def _():
    import plotly.io as pio
    pio.renderers.default = "iframe_connected"
    return (pio,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # <span style='color:brown'>GeneScope and Breast Cancer</span>

        ### <span style='color:brown'>Objective:</span>

        By training a deep learning model, optimized through the backpropagation algorithm, we are aimong to further analyze biomarkers and prognoses. Cancer biomarkers are biological indicators, such as genes, proteins, or other substances, that can reveal important details about a person's cancer. We are particularly interested in exploring whether the model can help identify genes that are more susceptible to cancer, causing mutations (<span style='color:brown'>indicating the bio markers</span>). For prognosis, we are intrested to how factors such as tumor size, the number of regional lymph nodes affected, distant metastasis, and patient demographics—such as age, ethnicity and race, play a critical role in cancer progression and patient outcomes(<span style='color:brown'>Prognosis</span>).



        ### <span style="color:brown">Data Source</span>

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
        """
        <span style="color:brown">Data Collection Challenges</span>

        Collecting data for our cancer study turned out to be much harder than we expected. Cancer is such a complex and heterogeneous disease that finding enough consistent data was a real challenge. We often ran into two major problems. Sometimes the datasets we found were high quality but only had around 200 samples — not nearly enough for the kind of deep analysis we wanted to do. Other times, even when more data was available, the methods used for gene extraction and normalization were completely different from one study to another. That made it impossible to simply combine multiple datasets without risking serious errors. Overall, getting a large, clean, and consistent dataset proved to be one of the biggest challenges in our project.
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
        # <span style="color: brown">Data Extraction and EDA</span>
        ### <span style="color: brown">Supplementary Data Files in GSE62944</span>

        The publisher of dataset provided the following supplementary files for extracting and labling gene exoressions.


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
    mo.md(
        r"""
        <span style="color: brown">1. EDA for 24 cancer type file</span>

        This suplementary files referances each patient sample id to their related diagnosed cancer. and it had 1119 breast cancer samples(Healthy and Melignant). We seporated all the breast cancer's related samples and later used it to label our gene expressions.

        **Note:** this data set labels breast cancer as BRCA and its nor related to the supressor gene BRCA.
        """
    )
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(create_supl_df):
    cancerBRCA_type_df, data, cType_count= create_supl_df('./data/SuplementoryFiles/TCGA_24_CancerType_Samples.txt', 'BRCA')

    cancer_descriptions_list = [
        "Breast Cancer - A malignant tumor that develops from breast cells.",
        "Uterine Corpus Endometrial Carcinoma - A cancer originating in the lining of the uterus.",
        "Kidney Renal Clear Cell Carcinoma - A common kidney cancer originating in renal tubules.",
        "Lung Adenocarcinoma - A non-small cell lung cancer starting in mucus-secreting glands.",
        "Lower-Grade Glioma - A slow-growing brain tumor arising from glial cells.",
        "Thyroid Carcinoma - A cancer that develops in the thyroid gland.",
        "Head and Neck Squamous Cell Carcinoma - A cancer originating in the mucosal linings of the head and neck.",
        "Prostate Adenocarcinoma - A cancer that forms in the prostate gland.",
        "Lung Squamous Cell Carcinoma - A type of lung cancer arising from squamous cells lining the airways.",
        "Colon Adenocarcinoma - A common type of colorectal cancer originating in glandular cells.",
        "Skin Cutaneous Melanoma - A dangerous type of skin cancer that develops from melanocytes.",
        "Ovarian Serous Cystadenocarcinoma - A type of ovarian cancer developing in the epithelial cells of the ovary.",
        "Stomach Adenocarcinoma - A malignant tumor forming in the stomach lining.",
        "Bladder Urothelial Carcinoma - A cancer that arises from the bladder lining.",
        "Liver Hepatocellular Carcinoma - The most common type of liver cancer, often linked to hepatitis or cirrhosis.",
        "Cervical Squamous Cell Carcinoma and Endocervical Adenocarcinoma - A cancer arising from the cervix.",
        "Kidney Renal Papillary Cell Carcinoma",
        "Acute Myeloid Leukemia",
        "Glioblastoma Multiforme - An aggressive brain tumor arising from glial cells.",
        "Rectum Adenocarcinoma - A form of colorectal cancer affecting the rectum.",
        "Adrenocortical Carcinoma - A rare cancer that originates in the adrenal cortex.",
        "Kidney Chromophobe",
        "Uterine Carcinosarcoma",
        "Genomic Variation in Diffuse Large B Cell Lymphomas"
    ]
    cType_count=cType_count.to_frame()
    cType_count['Description'] = cancer_descriptions_list
    return cType_count, cancerBRCA_type_df, cancer_descriptions_list, data


@app.cell(hide_code=True)
def _(cType_count, mo):
    mo.tabs(
        tabs={
            'Cancers Count': mo.ui.table(cType_count),
            'Extraction Method': mo.vstack([
                mo.ui.code_editor(
                    """
    def create_supl_df(source, cancerType):
        \"\"\"
        This function creates a data frame from a supplementary file.
        Input: supplementary file path, cancerType of interest
        Returns: a filtered DataFrame, raw data, and cancer type counts
        \"\"\"

        # Read supplementary file    
        with open(source, 'r') as file:
            data = [line.split('\\t') for line in file.readlines()]

        data_dic = {"Samples": [], "CancerType": []}
        for line in data:
            data_dic['Samples'].append(line[0])
            data_dic['CancerType'].append(line[1])

        cancer_type_df = pd.DataFrame(data_dic)
        cancer_type_df['CancerType'] = cancer_type_df['CancerType'].str.strip()
        cType_count = cancer_type_df.CancerType.value_counts()
        cancer_type_df = cancer_type_df[cancer_type_df['CancerType'] == cancerType]
        return cancer_type_df, data, cType_count
                    """,
                    language="python",
                    max_height=400  # optional: sets height so it scrolls if big
                ),
                mo.ui.code_editor(
                    """
    cancerBRCA_type_df, data, cType_count = create_supl_df(
        'TCGA_24_CancerType_Samples.txt', 'BRCA'
    )

    cancer_descriptions_list = [
        "Breast Cancer - A malignant tumor that develops from breast cells.",
        "Uterine Corpus Endometrial Carcinoma - A cancer originating in the lining of the uterus.",
        "Kidney Renal Clear Cell Carcinoma - A common kidney cancer originating in renal tubules.",
        "Lung Adenocarcinoma - A non-small cell lung cancer starting in mucus-secreting glands.",
        "Lower-Grade Glioma - A slow-growing brain tumor arising from glial cells.",
        "Thyroid Carcinoma - A cancer that develops in the thyroid gland.",
        "Head and Neck Squamous Cell Carcinoma - A cancer originating in the mucosal linings of the head and neck.",
        "Prostate Adenocarcinoma - A cancer that forms in the prostate gland.",
        "Lung Squamous Cell Carcinoma - A type of lung cancer arising from squamous cells lining the airways.",
        "Colon Adenocarcinoma - A common type of colorectal cancer originating in glandular cells.",
        "Skin Cutaneous Melanoma - A dangerous type of skin cancer that develops from melanocytes.",
        "Ovarian Serous Cystadenocarcinoma - A type of ovarian cancer developing in the epithelial cells of the ovary.",
        "Stomach Adenocarcinoma - A malignant tumor forming in the stomach lining.",
        "Bladder Urothelial Carcinoma - A cancer that arises from the bladder lining.",
        "Liver Hepatocellular Carcinoma - The most common type of liver cancer, often linked to hepatitis or cirrhosis.",
        "Cervical Squamous Cell Carcinoma and Endocervical Adenocarcinoma - A cancer arising from the cervix.",
        "Kidney Renal Papillary Cell Carcinoma",
        "Acute Myeloid Leukemia",
        "Glioblastoma Multiforme - An aggressive brain tumor arising from glial cells.",
        "Rectum Adenocarcinoma - A form of colorectal cancer affecting the rectum.",
        "Adrenocortical Carcinoma - A rare cancer that originates in the adrenal cortex.",
        "Kidney Chromophobe",
        "Uterine Carcinosarcoma",
        "Genomic Variation in Diffuse Large B Cell Lymphomas"
    ]

    cType_count = cType_count.to_frame()
    cType_count['Description'] = cancer_descriptions_list
                    """,
                    language="python",
                    max_height=400  
                )
            ])
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <span style="color: brown"> 2. the Normal_CancerType_Samples and Malignant_CancerType_Samples</spam>

        These text file are used to seporate and lable healtyh and cancerous sample ID. Later they are used to create healthy and cancerus gene expression datasets.
        """
    )
    return


@app.cell(hide_code=True)
def _(create_supl_df, mo):
    Normal_Sample_gene,_,_ = create_supl_df('./data/SuplementoryFiles/TCGA_Normal_CancerType_Samples .txt','BRCA')
    melignent_df, _, _=create_supl_df('./data/SuplementoryFiles/TCGA_20_CancerType_Samples.txt', 'BRCA')

    mo.tabs({'Benine Samples': mo.ui.table(Normal_Sample_gene), 'Melignanr Samples':mo.ui.table(melignent_df)})
    return Normal_Sample_gene, melignent_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <span style="color: brown"> 3. the Malignant_CancerType_Samples</spam>

        list of mutated samples!
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # <span style='color: brown'>Raw Gene Expression Data</sapn>

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
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""We extracter the gene expressions from the **Normal Samples from GSM1697009_06_01_15_TCGA_24.normal_Rsubread_TPM.txt** and **Samples from GSM1697009_06_01_15_TCGA_24.normal_Rsubread_TPM.txt** for both benine and cancerous datasets.""")
    return


@app.cell(hide_code=True)
def _(mo, pd):
    extraction_function = f"""
    def get_samples_df(source, sample_list):

            #this function takes in the file source and sample list of BRCA
            #then return the data frame that contains the all the genes and related benign and melignant samples
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
    """
    helthy_code = f"""
    healthy_dataSet = get_samples_df('GSE62944_RAW/GSM1697009_06_01_15_TCGA_24.normal_Rsubread_TPM.txt',Normal_Sample_gene['Samples'].to_list())
    healthy_dataSet=healthy_dataSet.T
    healthy_dataSet=healthy_dataSet.astype(float)
    """
    cancer_code = f"""
    df1 =get_samples_df('/GSE62944_RAW/GSM1536837_06_01_15_TCGA_24.tumor_Rsubread_TPM.txt', cancerBRCA_type_df['Samples'].to_list())
    df2 =get_samples_df('../../data/GSE62944_RAW/GSM1536837_06_01_15_TCGA_24.tumor_Rsubread_TPM.txt', melignent_df['Samples'].to_list())
    cancer_dataSet = pd.concat([df1.T,df2.T],ignore_index=False)
    """
    helthy_code = f"""
    healthy_dataSet = get_samples_df('/GSE62944_RAW/GSM1697009_06_01_15_TCGA_24.normal_Rsubread_TPM.txt',Normal_Sample_gene['Samples'].to_list())
    healthy_dataSet=healthy_dataSet.T
    healthy_dataSet=healthy_dataSet.astype(float)
    cancer_dataSet=cancer_dataSet.astype(float)
    """
    healthy_dataSet = pd.read_csv('./data/helthyExpressions.csv')
    cancer_dataSet = pd.read_csv('./data/cancerExpressions.csv')
    cancer_dataSet.rename(columns={'Unnamed: 0':'Samples'}, inplace=True)

    extraction_tab = mo.vstack([
        mo.ui.code_editor(extraction_function),
        mo.ui.code_editor(helthy_code),
        mo.ui.code_editor(cancer_code)
    ])

    # --- Assemble the tabs ---
    mo.tabs({
        "Healthy Gene Expression": mo.ui.table(healthy_dataSet),
        "Malignant Gene Expression": mo.ui.table(cancer_dataSet),
        "Extraction Method": extraction_tab
    })
    return (
        cancer_code,
        cancer_dataSet,
        extraction_function,
        extraction_tab,
        healthy_dataSet,
        helthy_code,
    )


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
def _(io, mo, pd):
    extraction_code = f"""with open('GSE62944_01_27_15_TCGA_20_420_Clinical_Variables_7706_Samples.txt', 'r') as file:
        exractedData1 = [line.split('\\t') for line in file.readlines()]
    with open('GSE62944_06_01_15_TCGA_24_548_Clinical_Variables_9264_Samples.txt', 'r') as file:
        exractedData2 = [line.split('\\t') for line in file.readlines()]

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

    # Remove NaN columns
    concat_df = concat_df.loc[:, ~concat_df.columns.isna()]

    # Rename index column
    if '' in concat_df.columns:
        concat_df.rename(columns={{'': 'Samples'}}, inplace=True)

    # Capture df.info() output
    info_buf = io.StringIO()
    concat_df.info(verbose=True, show_counts=True, buf=info_buf)
    """

    # Load pre-processed DataFrame for display
    concat_df = pd.read_csv('./AHPresults/initial_clinical_df.csv')

    # Capture df.info() output separately (you missed this)
    info_buf = io.StringIO()
    concat_df.info(verbose=True, show_counts=True, buf=info_buf)
    info_str = info_buf.getvalue()

    # Build Marimo UI
    cl_tabs = mo.tabs({
        'Data': mo.ui.table(concat_df),
        'Info': mo.md(f"```\n{info_str}\n```"),
        'Extraction': mo.ui.code_editor(extraction_code, language="python")
    })

    cl_tabs
    return cl_tabs, concat_df, extraction_code, info_buf, info_str


@app.cell
def _(mo):
    mo.md("""After Creating a data fram from the two <span style="color: brown">two clinical suplimentary files</span>, now we have to extract all the BRCA data from it and the following table displays all the brca related logistics. Now we can delve in furthure to create a dataset for stage prognosis and furthermore analyze patients demografic and see how they are corolated.""")
    return


@app.cell(hide_code=True)
def _(cancer_dataSet, concat_df, info_str, mo):
    # healthyBRCAList = healthy_dataSet.index
    cancerBRCAList = cancer_dataSet['Samples']

    BRCA_CV = concat_df[concat_df['Samples'].isin(cancerBRCAList)].copy()
    # BRCA_CV.drop(columns='index', inplace=True)
    mo.tabs({
        'Data': mo.ui.table(BRCA_CV),
        'Info': mo.md(f"`\n{info_str}\n`")
    })
    return BRCA_CV, cancerBRCAList


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""By over looking of the columns of this dataset columns we initially thout the data can be used for the following analysis.""")
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
    tabs = mo.tabs({
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
def _(BRCA_CV, melignent_df, np, pd):
    from io import StringIO

    # --- Initial Dataset ---
    new_brca_df = BRCA_CV.copy()

    # --- Utility functions ---
    def get_value_counts_table(df):
        """Return a long table of (column, value, count) for value_counts across all columns."""
        records = []
        for col in df.columns:
            counts = df[col].value_counts(dropna=False)
            for val, count in counts.items():
                records.append({
                    "Column": col,
                    "Value": val,
                    "Count": count
                })
        return pd.DataFrame(records)

    def get_info_str(df):
        """Capture df.info() output as a string."""
        buffer = StringIO()
        df.info(buf=buffer, verbose=True, show_counts=True)
        return buffer.getvalue()

    def convert_numeric_columns(df):
        """Convert columns that look numeric into actual numeric types."""
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except Exception:
                pass
        return df
    # --- Step 1: Original Dataset ---
    original_info = get_info_str(new_brca_df)
    original_value_counts = get_value_counts_table(new_brca_df)

    # --- Step 2: Replace "NA" with np.nan ---
    new_brca_df.replace("NA", np.nan, inplace=True)
    after_replace_info = get_info_str(new_brca_df)
    after_replace_value_counts = get_value_counts_table(new_brca_df)

    # --- Step 3: Convert numeric-looking columns ---
    new_brca_df = convert_numeric_columns(new_brca_df)
    after_convert_info = get_info_str(new_brca_df)
    after_convert_value_counts = get_value_counts_table(new_brca_df)

    # --- Step 4: Drop columns after 189th and filter Samples ---
    droped_Df = new_brca_df.copy()
    droped_Df = droped_Df.iloc[:, :189]
    droped_Df = droped_Df[droped_Df['Samples'].isin(melignent_df['Samples'])]

    after_drop_info = get_info_str(droped_Df)
    after_drop_value_counts = get_value_counts_table(droped_Df)

    # --- Step 5: Create Column Layout Table ---
    col_list = droped_Df.columns.tolist()
    total_cells = 21 * 9
    col_list += [""] * (total_cells - len(col_list))
    col_array = np.array(col_list).reshape(21, 9)
    col_table_df = pd.DataFrame(col_array, columns=[f"Col {i+1}" for i in range(9)])

    # --- Step 6: Final Fill NAs ---
    droped_Df.fillna("Not Available", inplace=True)
    final_info = get_info_str(droped_Df)
    final_value_counts = get_value_counts_table(droped_Df)
    final_value_counts
    return (
        StringIO,
        after_convert_info,
        after_convert_value_counts,
        after_drop_info,
        after_drop_value_counts,
        after_replace_info,
        after_replace_value_counts,
        col_array,
        col_list,
        col_table_df,
        convert_numeric_columns,
        droped_Df,
        final_info,
        final_value_counts,
        get_info_str,
        get_value_counts_table,
        new_brca_df,
        original_info,
        original_value_counts,
        total_cells,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        | Step | Cleaning Action | Purpose |
        |:----|:-----------------|:--------|
        | **1** | Replace "NA" with np.nannp.nan | Standardize missing values so pandas can detect and handle them correctly. |
        | **2** | Convert numeric-looking columns to numbers | Fix columns where numbers are stored as text, enabling proper analysis and modeling. |
        | **3** | Drop columns after the 189th column | Retain only clinically meaningful columns and remove irrelevant features. |
        | **4** | Keep only malignant samples (melig≠ntdfmelignent_df) | Focus the study specifically on malignant (cancerous) patient cases. |
        | **5** | Display cleaned column names in a 21×9 grid | Organize and inspect the final set of column names easily for verification. |

        The following Tabke show why this clinical variable was not usefull. The are many Not Available data Abailable.
        """
    )
    return


@app.cell(hide_code=True)
def _(droped_Df, final_info, mo):
    ext_code = f"""
    # --- Utility functions ---
    def get_value_counts_table(df):
        \"\"\"Return a long table of (column, value, count) for value_counts across all columns.\"\"\"
        records = []
        for col in df.columns:
            counts = df[col].value_counts(dropna=False)
            for val, count in counts.items():
                records.append({{
                    "Column": col,
                    "Value": val,
                    "Count": count
                }})
        return pd.DataFrame(records)

    def get_info_str(df):
        \"\"\"Capture df.info() output as a string.\"\"\"
        buffer = StringIO()
        df.info(buf=buffer, verbose=True, show_counts=True)
        return buffer.getvalue()

    def convert_numeric_columns(df):
        \"\"\"Convert columns that look numeric into actual numeric types.\"\"\"
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except Exception:
                pass
        return df

    # --- Step 1: Original Dataset ---
    original_info = get_info_str(new_brca_df)
    original_value_counts = get_value_counts_table(new_brca_df)

    # --- Step 2: Replace "NA" with np.nan ---
    new_brca_df.replace("NA", np.nan, inplace=True)
    after_replace_info = get_info_str(new_brca_df)
    after_replace_value_counts = get_value_counts_table(new_brca_df)

    # --- Step 3: Convert numeric-looking columns ---
    new_brca_df = convert_numeric_columns(new_brca_df)
    after_convert_info = get_info_str(new_brca_df)
    after_convert_value_counts = get_value_counts_table(new_brca_df)

    # --- Step 4: Drop columns after 189th and filter Samples ---
    dropped_Df = new_brca_df.copy()
    dropped_Df = dropped_Df.iloc[:, :189]
    dropped_Df = dropped_Df[dropped_Df['Samples'].isin(melignent_df['Samples'])]

    after_drop_info = get_info_str(dropped_Df)
    after_drop_value_counts = get_value_counts_table(dropped_Df)

    # --- Step 5: Create Column Layout Table ---
    col_list = dropped_Df.columns.tolist()
    total_cells = 21 * 9
    col_list += [""] * (total_cells - len(col_list))
    col_array = np.array(col_list).reshape(21, 9)
    col_table_df = pd.DataFrame(col_array, columns=[f"Col {{i+1}}" for i in range(9)])

    # --- Step 6: Final Fill NAs ---
    dropped_Df.fillna("Not Available", inplace=True)
    final_info = get_info_str(dropped_Df)
    final_value_counts = get_value_counts_table(dropped_Df)
    """

    mo.tabs({
        'Data': droped_Df,
        'Info': mo.md(f"```\n{final_info}\n```"),
        'Extraction': mo.ui.code_editor(ext_code)
    })
    return (ext_code,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ###<span style="color: brown">Using a secondary Clinical Variable dataset Directy extracted from TCGA</span>
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
    return (r_script,)


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
    mo.tabs({
        "DataSet": mo.ui.table(other_sup_files),
        "Info": mo.ui.code_editor(f"`\n{info_text}\n`")  # Display info in a nicely formatted code block
    })
    return info_buffer, info_text, other_sup_files


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### <span style="color: brown">DataSet Clean up Proccess</span>

        There are many columns that contains zero non_Null values. To prevent pandas dropping other columns or rows mistakenly I drop the columns that had more than threshold of %80 Nan values. and drop the rows of staging related columns respectively. The following data set is the cleaned up version of our dataset.
        """
    )
    return


@app.cell
def _(other_sup_files):
    thresholds = 0.8
    other_sup_files.dropna(thresh=int(thresholds * len(other_sup_files)), axis=1, inplace=True)
    return (thresholds,)


@app.cell(hide_code=True)
def _(io, mo, other_sup_files):
    other_sup_files.dropna(inplace=True)
    buffer = io.StringIO()
    other_sup_files.info(verbose=True, show_counts=True, buf=buffer)
    moded_info = buffer.getvalue()  # Store the captured output as a string

    # Display tabs correctly
    mo.tabs({
        "Cleaned Table": mo.ui.table(other_sup_files),
        "Info": mo.ui.code_editor(f"`\n{moded_info}\n`")  # Properly formatted info
    })
    return buffer, moded_info


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""##<span style="color:brown">Number of stages</span>""")
    return


@app.cell(hide_code=True)
def _(pd):
    stagedata = pd.read_csv('./AHPresults/fina_Stage_unaugmented.csv')
    stagedata['Stage'].value_counts()
    return (stagedata,)


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

        Prepared dataset for analysis:
        """
    )
    return


@app.cell
def _(mo, stagedata):
    mo.ui.table(stagedata)
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
def _(go, mo, np, pd, stagedata, stats):
    def cramers_v(x, y):
        confusion_matrix = pd.crosstab(x, y)
        chi2 = stats.chi2_contingency(confusion_matrix, correction=False)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        return np.sqrt(phi2 / min(k-1, r-1))

    # --------------- Your Data Preparation ---------------
    corr_df = stagedata[['Stage', 'year_of_diagnosis', 'ajcc_pathologic_t', 'ajcc_pathologic_n',
                         'ajcc_pathologic_m', 'paper_miRNA.Clusters', 'ethnicity', 'race',
                         'age_at_diagnosis', 'vital_status']].copy()

    # Encode categorical columns
    for cols in ['Stage', 'vital_status', 'ethnicity', 'race', 'paper_miRNA.Clusters']:
        corr_df[cols] = corr_df[cols].astype('category')

    # Drop rows with missing values
    corr_df = corr_df.dropna()

    # Now calculate association
    stage_corr = []
    vital_corr = []

    featuress = [col for col in corr_df.columns if col not in ['Stage', 'vital_status']]

    for _col in featuress:
        if corr_df[_col].dtype.name == 'category' or corr_df[_col].dtype == object:
            # Use Cramer's V for categorical
            corr_stage = cramers_v(corr_df['Stage'], corr_df[_col])
            corr_vital = cramers_v(corr_df['vital_status'], corr_df[_col])
        else:
            # Use Pearson correlation for numeric (like age)
            corr_stage = np.corrcoef(corr_df['Stage'].cat.codes, corr_df[_col])[0, 1]
            corr_vital = np.corrcoef(corr_df['vital_status'].cat.codes, corr_df[_col])[0, 1]

        stage_corr.append(corr_stage)
        vital_corr.append(corr_vital)

    # Results DataFrame
    correlation_results = pd.DataFrame({
        'Feature': featuress,
        'Association_with_Stage': stage_corr,
        'Association_with_Vital_Status': vital_corr
    }).dropna()

    # --------------- Plot ---------------
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=correlation_results['Feature'],
        y=correlation_results['Association_with_Stage'],
        name='Association with Stage'
    ))

    fig.add_trace(go.Bar(
        x=correlation_results['Feature'],
        y=correlation_results['Association_with_Vital_Status'],
        name='Association with Vital Status'
    ))

    fig.update_layout(
        barmode='group',
        title='Feature Associations with Stage and Vital Status (Cramér\'s V and Pearson)',
        xaxis_title='Features',
        yaxis_title='Association Strength',
        width=1000,
        height=600
    )

    mo.ui.plotly(fig)
    return (
        cols,
        corr_df,
        corr_stage,
        corr_vital,
        correlation_results,
        cramers_v,
        featuress,
        fig,
        stage_corr,
        vital_corr,
    )


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
def _(corr_df, mo):
    mo.ui.table(corr_df)
    return


if __name__ == "__main__":
    app.run()
