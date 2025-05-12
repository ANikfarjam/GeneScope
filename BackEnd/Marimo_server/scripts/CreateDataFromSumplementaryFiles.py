import pandas as pd
import marimo as mo
import pickle as pkl


create_sampledata = """
\"""
This script create dataset from the suplementary files provided by the data source
first i created a function that all it needs source(suplementary file) and cancer type
Breast cancer is labeled as BRCA in this data set and does not refer to the gene BRCA
\"""
def create_supl_df(source, cancerType):
    \"""
        this function creates data frame from a suplementart file
        inout: suplementery file path, cancerType of interes
        returns -> a data_frame

    \"""

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


cancerBRCA_type_df, data, cType_count= create_supl_df('./data/SuplementoryFiles/TCGA_24_CancerType_Samples.txt', 'BRCA')
# create a Data Frame that displays the name and count of each cancer in this data set
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
"""
##################################################
"""
This script create dataset from the suplementary files provided by the data source
first i created a function that all it needs source(suplementary file) and cancer type
Breast cancer is labeled as BRCA in this data set and does not refer to the gene BRCA
"""
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


cancerBRCA_type_df, data, cType_count= create_supl_df('../../../data/SuplementoryFiles/TCGA_24_CancerType_Samples.txt', 'BRCA')
# create a Data Frame that displays the name and count of each cancer in this data set
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
cType_count.to_csv('cType.csv', index=True)
####normal samples and cancer samples extraction
Normal_Sample_gene,_,_ = create_supl_df('../../../data/SuplementoryFiles/TCGA_Normal_CancerType_Samples .txt','BRCA')
melignent_df, _, _=create_supl_df('../../../data/SuplementoryFiles/TCGA_20_CancerType_Samples.txt', 'BRCA')


### Data Tabs ###
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
# Load just the header
full_header = pd.read_csv('../../../data/ModelDataSets/helthyExpressions.csv', nrows=0).columns.tolist()

# Load partial data with proper headers
healthy_dataSet = pd.read_csv('../../../data/ModelDataSets/helthyExpressions.csv', skiprows=1, nrows=20, names=full_header)
cancer_dataSet = pd.read_csv('../../../data/ModelDataSets/cancerExpressions.csv', skiprows=1, nrows=20, names=full_header)
cancer_dataSet.rename(columns={'Unnamed: 0':'Samples'}, inplace=True)

cType_count = cType_count.to_html(classes='CancerCount', index=True)
healthy_dataSet = healthy_dataSet.to_html(classes='Healthy', index=True)
cancer_dataSet = cancer_dataSet.to_html(classes='Melignant', index=True)

# geneEda = mo.ui.tabs({
#     'Healthy Samples':mo.Html(healthy_dataSet),
#     'Cancer_dataSet':mo.Html(cancer_dataSet),
#     'Number of Cancer Types in our Data':mo.Html(cType_count),
#     'Extraction_Method': mo.vstack([
#         mo.ui.code_editor(create_sampledata, language="python"),
#         mo.ui.code_editor(extraction_function , language="python"),
#     ])
# })

# with open('pkl_files/geneEDA.pkl', 'wb') as f:
#     pkl.dump(geneEda, f)

healthy_df = pd.read_csv('../../../data/ModelDataSets/helthyExpressions.csv', skiprows=1, nrows=20)
cancer_df = pd.read_csv('../../../data/ModelDataSets/cancerExpressions.csv', skiprows=1, nrows=20)
cancer_df.rename(columns={'Unnamed: 0': 'Samples'}, inplace=True)


# Build mo.Html UI objects directly (as requested)
healthy_component = mo.Html(healthy_df.to_html(classes="Healthy", index=False))
cancer_component = mo.Html(cancer_df.to_html(classes="Melignant", index=False))
cType_component = cType_count
# # Your extraction code (not modified)
# create_sampledata = "..."  # insert your code string here if not importing
# extraction_function = "..."  # same here

# Make UI elements now

c2_doce = """
\"""
This script create dataset from the suplementary files provided by the data source
first i created a function that all it needs source(suplementary file) and cancer type
Breast cancer is labeled as BRCA in this data set and does not refer to the gene BRCA
\"""
def create_supl_df(source, cancerType):
    \"""
        this function creates data frame from a suplementart file
        inout: suplementery file path, cancerType of interes
        returns -> a data_frame

    \"""

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

"""
code_editor_1 = mo.ui.code_editor(c2_doce, language="python")
code_editor_2 = mo.ui.code_editor(extraction_function, language="python")

# Now pickle full UI objects
components = {
    "healthy": healthy_component,
    "cancer": cancer_component,
    "cType": cType_component
}

with open("pkl_files/full_html_ui.pkl", "wb") as f:
    pkl.dump(components, f)

print("✅ UI components pickled as full `mo.Html()` objects.")




