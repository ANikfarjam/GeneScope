import marimo

__generated_with = "0.12.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import plotly.express as px
    return mo, pd, px


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #<span style="color:brown">Breast Cancer Classification usibg Deep Learning Neural Network</span>

        Gene Scop utilizes a predictive model for breast cancer stages classification given gene expression and clinical variables. We had a huge data set of data consist of 2100 samples. Each sample is consist of clinical data such as Age, tumore related data(TNM), vital status and Gene expressions.

        <span style='color: brown'>Challege:</span> 

        The main challange with working data like this is the course of dimentionality. Human body consist of 23000 genes. We had lot cilinical varible that not all could be used. Thus we needed to do thoural feature selection.

        ### <span style="color:brown">Intro to he Multimodal Deep Neural Network for Multi-dimensional Data </span>

        The Multimodal Deep Neural Network for Multi-dimensional Data (MDNNMD) is a neural architecture designed to integrate and learn from multiple heterogeneous data sources simultaneously. Instead of feeding all features into a single model, MDNNMD builds independent deep neural networks for each data modality—such as gene expression, clinical records, and genomic alterations—allowing each to learn specialized feature representations. These subnetworks are then combined using score-level fusion, where the outputs of each modality-specific DNN are weighted and aggregated to produce a final prediction. 
         [Dongdong Sun etal](https://ieeexplore.ieee.org/abstract/document/8292801?casa_token=J6Bt__TE05sAAAAA:P7rbhfoJDDM9DSFFDfsc5kUYGSFO-9c0UPnTINRVZkvUjYe9EUufViBbfkZ23pJ_XL2SQMHG1dfN
        ) pulished an article utilizing this modol on cancer prognosis strudy. They emphasize the modular and flexible architecture enables MDNNMD to capture complementary information across different data types, improving overall performance and robustness in complex biomedical classification tasks like disease prognosis.

        <span style='color: brown'>Metigate Dimentionality issue and Modol Architecture Diagram</span>

        For feature selection of **Clinical Variables** we used random forest to select important feature. And we used Analytical Hieracial Processes AHP.
        """
    )
    return


@app.cell
def _(mo):
    mo.image('./DMM.png')
    return


@app.cell
def _(mo):
    mo.md(
        """
        ###<span style='color:brown'>Common Technique used in MDNNMD</span>

        There are five strategies used in multimodal deep learning, especially when combining different types of data,like gene expression (numeric, high-dimensional) and clinical variables (tabular, categorical).
        """
    )
    return


@app.cell
def _(mo):
    def multimodal_learning_columns():
        return mo.md("""
        <style>
          .column-container {
            display: flex;
            gap: 20px;
            overflow-x: auto;
            padding: 20px;
            background: #eeeeee;
          }

          .column {
            flex: 0 0 300px;
            height: 460px;
            overflow-y: auto;
            padding: 16px;
            background: #4b3a4d;
            border-radius: 14px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            color: #f2e6ff;
            border: 1px solid #7d5f85;
            transition: transform 0.3s ease;
          }

          .column:hover {
            transform: translateY(-4px);
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.25);
          }

          .column h3 {
            margin-top: 0;
            color: #d6b1ff;
          }

          .column h4 {
            margin-bottom: 4px;
            color: #eac9ff;
            font-size: 1em;
            margin-top: 14px;
          }

          .column p {
            font-size: 0.95em;
            line-height: 1.5;
            color: #e3d6ec;
          }

          .column-container::-webkit-scrollbar {
            height: 8px;
          }

          .column::-webkit-scrollbar {
            width: 8px;
          }

          .column::-webkit-scrollbar-thumb {
            background: #9b7fa2;
            border-radius: 4px;
          }
        </style>

        <div class="column-container">
          <div class="column">
            <h3>🔗 Fusion</h3>
            <p>Fusion refers to combining information from different data modalities—like gene expression and clinical variables—to make a unified prediction. It’s a key strategy in multimodal deep learning, and comes in three main types:</p>

            <h4>Early Fusion</h4>
            <p>All raw features from each modality are concatenated before being input into a single model. This works well if modalities are aligned and comparable in scale, but may suffer from noise or high dimensionality.</p>

            <h4>Intermediate Fusion</h4>
            <p>Each modality is processed by its own neural network, and their intermediate feature representations are combined—usually through concatenation or attention—before passing through shared layers for prediction. This balances independence and interaction.</p>

            <h4>Late Fusion</h4>
            <p>Each modality has a fully independent network, and predictions from each are aggregated (e.g., averaged, weighted sum, or voted) at the decision level. This approach is modular and robust, ideal when modalities are heterogeneous or separately trained.</p>
          </div>

          <div class="column">
            <h3>🧭 Alignment</h3>
            <p>Alignment ensures features from multiple modalities refer to the same subject. For example, a sample’s gene profile must match its clinical record. Techniques like normalization, timestamp syncing, and shared encoding layers help maintain consistent correspondence across views.</p>
          </div>

          <div class="column">
            <h3>🔄 Translation</h3>
            <p>Translation models learn to predict or reconstruct one modality from another, such as estimating gene expression from clinical data. This is useful for imputing missing data, or for discovering latent cross-modal relationships that help generalization.</p>
          </div>

          <div class="column">
            <h3>🤝 Co-Learning</h3>
            <p>Co-learning enables two or more modalities to learn interactively. Rather than training them separately, shared gradients or intermediate feedback are used so that progress in one modality improves representations in the other. This increases synergy and model robustness.</p>
          </div>
        """)

    multimodal_learning_columns()
    return (multimodal_learning_columns,)


if __name__ == "__main__":
    app.run()
