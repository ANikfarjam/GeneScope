import marimo

__generated_with = "0.8.22"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""Author: Ashkan Nikfarjam""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # <span style="color:green">Data Source</span>

        The Gene Expression dataset has been extracted from the [NCBI Gene Expression Omnibus GSEGSE62944](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE62944). The dataset consists of 9264 tumor samples and 741 normal samples across 24 cancer types from The Cancer Genome Atlas. This data set is created from all the TCGA samples that have an expression of 23,000 genes!
        """
    )
    return


@app.cell
def _(mo):
    mo.center(mo.image(src="TCGAcancerTypes.png",caption="The Cancer Types available in TCGA database. This image is from the TCGA officcial website!", width=500,  height=400))
    return


@app.cell
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


@app.cell
def _(mo):
    mo.md(
        """
        ### <span style="color: brown">Data Files in GSE62944

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


@app.cell
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


@app.cell
def _(mo):
    mo.md(r"""For final clean up we are going to convert the age from days to year, and our data is ready for visualizatoin and further analysis.""")
    return


@app.cell
def _():
    #stages_df.to_csv('./AHPresults/stage_dataSet.csv', index=False)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### <span style="color: brown">DataSet Clean up Proccess</span>
        Before diving into model training, I started by preparing the clinical dataset. The target variable I wanted to predict was Stage which refers to the stage of breast cancer for each patient. To get the data into a usable format, I encoded all categorical variables using LabelEncoder, since machine learning models generally require numerical inputs. I also filled in any missing values with the mean of each column — not perfect, but a reasonable starting point for handling gaps in the data. When everything was cleaned up I then split the dataset into training and test sets using an 80/20 split. This gave me a good balance between having enough data to train the model and enough to evaluate it afterward. With that done, I trained a baseline RandomForestClassifier using the default settings (100 trees, no max depth, etc.). This gave me a first look at how well the model could predict cancer stages using all the features, and it set the foundation for deeper analysis.

        The initial classification report looked promising. For example, classes like Stage IIA and Stage IIB were predicted quite well, while classes like Stage II and Stage IIIC showed lower performance — probably due to class imbalance or overlap in clinical features. This gave me a sense of which labels were harder for the model to get right and where there might be room to improve.
        """
    )
    return


@app.cell
def __(pd):
    other_sup_file = pd.read_csv("random_forest\clinical_data.csv")
    return (other_sup_file,)


@app.cell
def _(other_sup_file):
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix


    clinical_df = other_sup_file.copy()
    target = 'Stage'
    features = clinical_df.columns.drop(target)

    le = LabelEncoder()
    for col in features:
        if clinical_df[col].dtype == 'object':  
            clinical_df[col] = le.fit_transform(clinical_df[col].astype(str))

    X = clinical_df[features].copy()
    y = clinical_df[target]
    X.fillna(X.mean(), inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Predict on the testing set
    y_pred = rf.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=1))
    return (
        GridSearchCV,
        LabelEncoder,
        RandomForestClassifier,
        X,
        X_test,
        X_train,
        classification_report,
        clinical_df,
        col,
        confusion_matrix,
        features,
        le,
        rf,
        target,
        train_test_split,
        y,
        y_pred,
        y_test,
        y_train,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ### <span style="color: brown">Feature Analysis of Clinical Data</span>
        With the model trained, I wanted to see which features actually mattered. Random Forests make this easy since they provide feature importance scores out of the box. These scores show how much each variable contributed to reducing uncertainty in the model’s predictions. I sorted the features by importance and created a bar chart of the top 15.
        """
    )
    return


@app.cell
def _(features, mo, np, pd, rf):
    feature_importances = rf.feature_importances_
    indices = np.argsort(feature_importances)[::-1]
    non_zero_indices = [i for i in indices if feature_importances[i] != 0]

    feature_ranking_df = pd.DataFrame({
        'Feature': [features[i] for i in non_zero_indices],
        'Importance': [feature_importances[i] for i in non_zero_indices]
    })

    feature_ranking_df = feature_ranking_df.sort_values(by='Importance', ascending=False)
    feature_ranking_df.reset_index(drop=True, inplace=True)
    mo.ui.table(feature_ranking_df)
    return (
        feature_importances,
        feature_ranking_df,
        indices,
        non_zero_indices,
    )


@app.cell
def __(mo):
    mo.md(r"""From this results we can see that a few features such as `paper_pathologic_stage`, `ajcc_pathologic_n`, and `ajcc_pathologic_t` were by far the most influential. This makes sense because these features are directly related to the clinical staging process. Features like days_to_collection, days_to_last_followup, and treatments also showed up, which might reflect how the progression and treatment timing relate to cancer stage.""")
    return


@app.cell(hide_code=True)
def _(mo):
    """import matplotlib.pyplot as plt

    top_10_indices = non_zero_indices[:15]
    plt.figure()
    plt.title("Top 15 Feature importances")
    plt.bar(range(len(top_10_indices)), [feature_importances[i] for i in top_10_indices], color="r", align="center")
    plt.xticks(range(len(top_10_indices)), [features[i] for i in top_10_indices], rotation=90)  # Rotate labels to 45 degrees
    plt.xlim([-1, len(top_10_indices)])
    plt.tight_layout() 
    plt.savefig('./top15..png')
    """

    mo.image('./top15..png')
    return


@app.cell
def __(mo):
    mo.md(r"""Visualizing this not only helped with model interpretability, but also gave me a clearer idea of which features were worth keeping for future models. Instead of throwing in the entire dataset, I could now focus on a cleaner and more impactful set of predictors, improving both speed and accuracy.""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        After identifying the key features, I narrowed down my list to those that had the most impact on the model. I stored them in a list called features_important. This included clinical observations, demographic info, and a few identifiers that surprisingly had predictive power. Using only these features, I retrained the model — but this time, I wanted to do it right.
        Now instead of using the default settings, I ran a GridSearchCV to find the best combination of hyperparameters. I defined a grid that tested different numbers of trees (n_estimators), maximum depths, split criteria, and whether to use bootstrapping. The grid search ran 5-fold cross-validation on each parameter combination to get the most reliable estimate of performance.
        """
    )
    return


@app.cell
def _(mo):
    mo.ui.code_editor("""
    features_important = ['paper_pathologic_stage', 'ajcc_pathologic_n', 'ajcc_pathologic_t', 'days_to_collection', 
                'paper_days_to_last_followup', 'year_of_diagnosis', 'treatments', 'initial_weight', 'sample_id', 
                'days_to_birth', 'pathology_report_uuid', 'demographic_id', 'paper_days_to_birth', 'diagnosis_id', 
                'Unnamed: 0', 'age_at_diagnosis', 'sample_submitter_id', 'paper_patient', 'patient', 
                'paper_age_at_initial_pathologic_diagnosis', 'bcr_patient_barcode', 'barcode', 'submitter_id', 
                'sample', 'age_at_index', 'sites_of_involvement', 'paper_PARADIGM.Clusters', 'paper_Mutation.Clusters', 
                'primary_diagnosis', 'paper_CNV.Clusters', 'paper_BRCA_Subtype_PAM50', 'morphology', 'ajcc_pathologic_m', 
                'method_of_diagnosis', 'paper_miRNA.Clusters', 'paper_mRNA.Clusters', 'race', 'paper_DNA.Methylation.Clusters', 
                'laterality', 'ethnicity', 'preservation_method', 'paper_Included_in_previous_marker_papers', 
                'oct_embedded', 'paper_vital_status', 'vital_status', 'prior_malignancy', 'synchronous_malignancy', 
                'age_is_obfuscated', 'prior_treatment', 'tissue_or_organ_of_origin', 'icd_10_code']

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print("best parameters:", grid_search.best_params_)
    print("best score:", grid_search.best_score_)

    y_pred_grid = grid_search.best_estimator_.predict(X_test)
    print(classification_report(y_test, y_pred_grid, zero_division=1))
    """)
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        This process took longer to run, but it paid off. I ended up with a model that was better tuned to the dataset and less prone to overfitting. Grid search also gave me valuable insight into which settings worked best for this particular problem , whether deeper trees helped or hurt the results, and whether using more estimators led to meaningful improvements.

        The best-performing model used the following parameters:

        - `n_estimators = 100`
        - `max_depth = None` (allowing trees to grow fully)
        - `min_samples_split = 5`
        - `min_samples_leaf = 1`
        - `bootstrap = False`

        """
    )
    return


@app.cell
def __(
    LabelEncoder,
    RandomForestClassifier,
    classification_report,
    other_sup_file,
    train_test_split,
):
    selected_brca_features  = ['paper_pathologic_stage', 'ajcc_pathologic_n',
                          'ajcc_pathologic_t', 'days_to_collection', 
                          'paper_days_to_last_followup', 'year_of_diagnosis',
                          'treatments', 'initial_weight', 'sample_id', 
                          'days_to_birth', 'pathology_report_uuid',
                          'demographic_id', 'paper_days_to_birth', 'diagnosis_id', 
                          'Unnamed: 0', 'age_at_diagnosis', 'sample_submitter_id',
                          'paper_patient', 'patient', 
                          'paper_age_at_initial_pathologic_diagnosis',
                          'bcr_patient_barcode', 'barcode', 'submitter_id', 
                          'sample', 'age_at_index', 'sites_of_involvement',
                          'paper_PARADIGM.Clusters', 'paper_Mutation.Clusters', 
                          'primary_diagnosis', 'paper_CNV.Clusters',
                          'paper_BRCA_Subtype_PAM50', 'morphology',
                          'ajcc_pathologic_m', 'method_of_diagnosis', 
                          'paper_miRNA.Clusters','paper_mRNA.Clusters',
                          'race', 'paper_DNA.Methylation.Clusters', 
                          'laterality', 'ethnicity', 'preservation_method',
                          'paper_Included_in_previous_marker_papers', 
                          'oct_embedded', 'paper_vital_status', 'vital_status',
                          'prior_malignancy', 'synchronous_malignancy', 
                          'age_is_obfuscated', 'prior_treatment', 
                          'tissue_or_organ_of_origin','icd_10_code']


    # Load and prepare dataset
    brca_clinical_df = other_sup_file.copy()
    brca_target_column = 'Stage'
    brca_input_features = selected_brca_features

    # Encode categorical variables
    brca_label_encoder = LabelEncoder()
    for brca_column in brca_input_features:
        if brca_clinical_df[brca_column].dtype == 'object':  
            brca_clinical_df[brca_column] = brca_label_encoder.fit_transform(brca_clinical_df[brca_column].astype(str))

    # Prepare feature and target data
    brca_X = brca_clinical_df[brca_input_features].copy()
    brca_y = brca_clinical_df[brca_target_column]
    brca_X.fillna(brca_X.mean(), inplace=True)

    # Train-test split
    brca_X_train, brca_X_test, brca_y_train, brca_y_test = train_test_split(brca_X, brca_y, test_size=0.2, random_state=42)

    # Define and train tuned Random Forest model
    brca_rf_model = RandomForestClassifier(
        n_estimators=100,
        min_samples_split=5,
        min_samples_leaf=1,
        bootstrap=False,
        random_state=42
    )
    brca_rf_model.fit(brca_X_train, brca_y_train)

    # Evaluate model
    brca_y_pred = brca_rf_model.predict(brca_X_test)
    print(classification_report(brca_y_test, brca_y_pred, zero_division=1))
    return (
        brca_X,
        brca_X_test,
        brca_X_train,
        brca_clinical_df,
        brca_column,
        brca_input_features,
        brca_label_encoder,
        brca_rf_model,
        brca_target_column,
        brca_y,
        brca_y_pred,
        brca_y_test,
        brca_y_train,
        selected_brca_features,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        After looking at the new classification report, the final model definitely performs better than the baseline. Accuracy went up slightly from 91% to 93%, which is already a good sign. But more importantly, the model is doing a better job overall especially with the harder-to-predict classes. For example, Stage IIIC improved a lot recall jumped from 0.43 to 0.86, and the F1-score went from 0.60 to 0.92. That’s a huge improvement and shows the model can now catch more of those cases correctly. Other classes like Stage IIA, IIB, and IIIA also stayed strong, with both precision and recall staying high.

        Stage II is still an issue the model got 0 recall again but to be fair, there are only 3 samples, so that’s probably more of a data problem than a model issue. Overall though, the macro and weighted averages for recall and F1-score improved, meaning the model is handling class imbalance better than before. Tuning the parameters and using only the most important features really paid off.
        """
    )
    return


if __name__ == "__main__":
    app.run()
