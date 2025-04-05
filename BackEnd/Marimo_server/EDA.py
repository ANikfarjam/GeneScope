import marimo

__generated_with = "0.8.22"
app = marimo.App(width="full")


@app.cell
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
        r"""
        # <span style="color:green">Data Source</span>

        The Gene Expression dataset has been extracted from the [NCBI Gene Expression Omnibus GSEGSE62944](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE62944). The dataset consists of 9264 tumor samples and 741 normal samples across 24 cancer types from The Cancer Genome Atlas. This data set is created from all the TCGA samples that have an expression of 23,000 genes!
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(mo.image(src="TCGAcancerTypes.png",caption="The Cancer Types available in TCGA database. This image is from the TCGA officcial website!", width=500,  height=400))
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
def _(mo):
    mo.md("""* These are the unique cancer types in dataset""")
    return


@app.cell
def _(cType_count, mo):
    mo.ui.table(cType_count)
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
    mo.md(
        """
        It seams like we have 2,200 samples and for the model training using 60,20,20 distribution. We are going to have [1321, 350, 350] sample distribution for test train and val dataset!

        As we can see how the expression of the same top 300 genes in the same order deffer significantly.
        """
    )
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
def _(io, mo, pd):
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
    # Capture df.info() output
    info_buf = io.StringIO()
    concat_df.info(verbose=True, show_counts=True, buf=info_buf)
    info_str = info_buf.getvalue()

    # Now pass tabs correctly
    mo.ui.tabs({
        'Data': mo.ui.table(concat_df),
        'Info': mo.md(f"```\n{info_str}\n```")
    })
    return (
        concat_df,
        df1,
        df2,
        exractedData1,
        exractedData2,
        file,
        info_buf,
        info_str,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""After Creating a data fram from the two <span style="color: brown">two clinical suplimentary files</span>, now we have to extract all the BRCA data from it and the following table displays all the brca related logistics. Now we can delve in furthure to create a dataset for stage prognosis and furthermore analyze patients demografic and see how they are corolated.""")
    return


@app.cell(hide_code=True)
def _(cancer_dataSet, concat_df, info_str, mo):
    # healthyBRCAList = healthy_dataSet.index
    cancerBRCAList = cancer_dataSet['Samples']

    BRCA_CV = concat_df[concat_df['Samples'].isin(cancerBRCAList)].copy()
    # BRCA_CV.drop(columns='index', inplace=True)
    mo.ui.tabs({
        'Data': mo.ui.table(BRCA_CV),
        'Info': mo.md(f"```\n{info_str}\n```")
    })
    return BRCA_CV, cancerBRCAList


@app.cell
def _():
    #BRCA_CV.to_csv('clinicalDF.csv', index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""By over looking of the columns of this dataset we can see that the data can be used for the following analysis:""")
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


@app.cell
def _(BRCA_CV):
    import numpy as np
    # value_dic={}
    # for col in BRCA_CV.columns:
    #     value_dic[col] = BRCA_CV[col].value_counts()
    # value_dic
    new_brca_df = BRCA_CV.copy()
    new_brca_df.replace("NA", np.nan, inplace=True)
    new_brca_df.info(verbose=True, show_counts=True)
    return new_brca_df, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        As we can see lot of unprovided values for these clinical varuables and also there is no logical way to replace these NaN values usin inputation or using mean beacuse its either string or each of these values in biology are not inputable because the symantical meaning it has. 

        For Cleaning up we are going to drop columns [189:]. How ever these three columns are going to leh us alot to label our gene expressions for staging and prepare the data for our deep learning model. 

         | Column | Variable | non Null count | Dtype |
         |--------|----------|----------------|-------|
         | 64 | clinical_M | 994 non-null | object |
         | 65 | clinical_N | 994 non-null | object |
         | 66 | clinical_T | 994 non-null | object |
        """
    )
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md("""###<span style="color: green">Labaling Data:</span>""")
    return


@app.cell(hide_code=True)
def _(info_str, melignent_df, mo, new_brca_df, np, pd):
    #drop the Nan colons
    droped_Df = new_brca_df.copy()
    droped_Df = droped_Df.iloc[:,:189]

    droped_Df = droped_Df[droped_Df['Samples'].isin(melignent_df['Samples'])]
    # Get column names as a flat list
    col_list = droped_Df.columns.tolist()

    # Fill the list with empty strings if not exactly 189 (21×9)
    total_cells = 21 * 9
    col_list += [""] * (total_cells - len(col_list))

    # Reshape into 21x9
    col_array = np.array(col_list).reshape(21, 9)

    # Create DataFrame for display
    col_table_df = pd.DataFrame(col_array, columns=[f"Col {i+1}" for i in range(9)])

    # Show it in the 'columns' tab
    mo.ui.tabs({
        'Table': mo.ui.table(droped_Df),
        'info': mo.md(f"```\n{info_str}\n```"),
        'columns': mo.ui.table(col_table_df, page_size=10)
    })
    return col_array, col_list, col_table_df, droped_Df, total_cells


@app.cell
def _(droped_Df, mo):
    mo.ui.tabs=(
        {'1st':{col: droped_Df[col].value_counts() for col in droped_Df.iloc[:,4:100]},
        '2nd':{col: droped_Df[col].value_counts() for col in droped_Df.iloc[:,101:]}}
    )
    return


@app.cell
def _(droped_Df):
    #'clinical_stage', ajcc_staging_edition, tumor_status
    clinical_MTN = droped_Df[['clinical_M', 'clinical_N', 'clinical_T']].value_counts()
    stage_ajcc = droped_Df['ajcc_staging_edition'].value_counts()
    droped_Df['tumor_status'].value_counts()
    return clinical_MTN, stage_ajcc


@app.cell
def _(droped_Df):
    droped_Df.fillna("Not Available", inplace=True)
    droped_Df.info(verbose=True, show_counts=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Now we cleaned up the data frame, its time to lable the samples based on <span style="color:green">M, T, N</span> markers related to breast cancer that we mentioned earlier.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ###<span style="color: green">Using a secondary Clinical Variable dataset Directy extracted from TCGA</span>
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
def _(mo, pd):
    import io
    # Load dataset
    other_sup_file = pd.read_csv("head1.csv")

    # Capture DataFrame info as a string
    info_buffer = io.StringIO()
    other_sup_file.info(verbose=True, show_counts=True, buf=info_buffer)
    info_text = info_buffer.getvalue()

    # Create Marimo UI tabs
    mo.ui.tabs({
        "DataSet": mo.ui.table(other_sup_file),
        "Info": mo.ui.code_editor(f"`\n{info_text}\n`")  # Display info in a nicely formatted code block
    })
    return info_buffer, info_text, io, other_sup_file


@app.cell
def _(mo):
    mo.md(
        """
        ### <span style="color: green">DataSet Clean up Proccess</span>

        There are many columns that contains zero non_Null values. To prevent pandas dropping other columns or rows mistakenly I drop the columns that had more than threshold of %80 Nan values. and drop the rows of staging related columns respectively. The following data set is the cleaned up version of our dataset.
        """
    )
    return


@app.cell
def _(other_sup_file):
    # cleaning up
    ### Step 1: Drop Columns with Too Many Missing Values 
    threshold = 0.8  # If more than 80% values are missing, drop the column


    other_sup_file.dropna(thresh=int(threshold * len(other_sup_file)), axis=1, inplace=True)
    return (threshold,)


@app.cell(hide_code=True)
def _(io, mo, other_sup_file):
    other_sup_file.dropna(inplace=True)
    buffer = io.StringIO()
    other_sup_file.info(verbose=True, show_counts=True, buf=buffer)
    moded_info = buffer.getvalue()  # Store the captured output as a string

    # Display tabs correctly
    mo.tabs({
        "Cleaned Table": mo.ui.table(other_sup_file),
        "Info": mo.ui.code_editor(f"`\n{moded_info}\n`")  # Properly formatted info
    })
    return buffer, moded_info


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now we can count the numer of samples available for each staging""")
    return


@app.cell(hide_code=True)
def _(other_sup_file):
    other_sup_file['ajcc_pathologic_stage'].value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        Since the number of sample for some stages are not suffitient enough to have datset fo reach individual Type <span style="color: brown">Stage II A, B, ...</span>, we are going to combine each stage. Thus our new stages would be:

        | Stages |||
        |--------|--------|--------|
        | Stage I | Stage II | Stage III |
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        With all stages combined still our data looks inbalanced. Researches sujest that to either adjust the weight during training or perform Data Augmentation (with Biological Knowledge) using techniques called **SMOTE** or **(VAE) or GANs**:

        ###<span style="color: green"> Augmentation technique</span>

        * **SMOTE (Synthetic Minority Over-sampling Technique)**: Creates synthetic samples by interpolating between existing samples of the minority class.

        * **Variational Autoencoders (VAE) or GANs (Generative Adversarial Networks)**: Generate realistic gene expression profiles for under-represented classes.

        * **Resampling**:

        Oversampling: Randomly duplicate samples from the minority classes to increase their representation.

        Undersampling: Randomly remove samples from the majority classes, but this risks losing valuable information.

        * **Data Augmentation Using HMM (Hidden Markov Model)**: 

        Train an HMM on gene expression profiles of the minority classes to generate similar but distinct sequences.

        <span style='color:green'>Since we have already done the AHP analysis for biomarkers analysis We are going to employ HMM to generate symentically gene expression for minority classes.</span>
        """
    )
    return


@app.cell
def _(cancer_dataSet, other_sup_file, pd):
    """
    creating and saving the stage dfs
    gender                                     737 non-null    object 
     45  ethnicity                                  737 non-null    object 
     46  vital_status
    """
    other_sup_file.rename(columns={'ajcc_pathologic_stage':'Stage'}, inplace=True)
    stages_df = pd.merge(other_sup_file[['barcode', 'Stage', 'age_at_diagnosis', 'gender', 'ethnicity', 'vital_status']], cancer_dataSet, left_on='barcode', right_on='Samples')
    stages_df.iloc[:,:4].info(verbose=True, show_counts=True)
    return (stages_df,)


@app.cell
def _(stages_df):
    stages_df.Stage.value_counts()
    return


@app.cell
def _(stages_df):
    stages_df.drop(columns='barcode', inplace=True)
    return


@app.cell
def _(stages_df):
    stages_df.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""For final clean up we are going to convert the age from days to year, and our data is ready for visualizatoin and further analysis.""")
    return


@app.cell
def _():
    #stages_df.to_csv('./AHPresults/stage_dataSet.csv', index=False)
    return


@app.cell
def __(mo, pd):
    df = pd.read_csv("./random_forest/clinical_data_20.csv")
    mo.ui.table(df)
    return (df,)


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
