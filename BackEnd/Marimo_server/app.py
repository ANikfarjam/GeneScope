import marimo

__generated_with = "0.11.12"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    # Define slider without marks
    slider = mo.ui.slider(
        start=0,
        stop=3,
        step=1,
        value=0,
        label="Select Analysis Model"
    )

    # Manual mapping for label display
    model_labels = ["AHP", "CatBoost", "Cox Hazard Model", "Random Forest"]
    return model_labels, slider


@app.cell(hide_code=True)
def _(mo, model_labels, slider):
    # Markdown blocks per model
    descriptions = {
        0: mo.md("""
    ### 🧠 Analytic Hierarchy Process (AHP)
    AHP ranks genes using multiple statistical methods (t-test, Entropy, Wilcoxon, ROC, SNR), combining them through pairwise comparison matrices and eigenvector-based scoring to identify key biomarkers in breast cancer progression.
    """),

        1: mo.md("""
    ### 🚀 CatBoost Classifier
    CatBoost is a gradient boosting model that learns the likelihood of a patient being diagnosed at a specific cancer stage using gene expression and clinical features. It handles class imbalance and delivers calibrated stage probabilities.
    """),

        2: mo.md("""
    ### 🧬 Cox Proportional Hazards Model
    The Cox model estimates the impact of clinical and molecular features on patient survival risk. It outputs hazard ratios and identifies statistically significant factors affecting prognosis (like metastasis, age, lymph nodes).
    """),

        3: mo.md("""
    ### 🌲 Random Forest
    This ensemble method ranks the most important clinical features by their influence on cancer classification. It’s robust against overfitting and captures nonlinear patterns in demographic and clinical data.
    """)
    }

    # Display current label + markdown
    mo.vstack([
        mo.md(f"### 🔬 Breast Cancer Model Analysis: {model_labels[slider.value]}"),
        descriptions[slider.value]
    ])
    return (descriptions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        <div style="font-family:sans-serif;">
          <h2>Breast Cancer Analysis Walkthrough</h2>

          <div id="stepper" style="background: #f7f7f7; padding: 20px; border-left: 4px solid #663399; border-radius: 8px; min-height: 160px; margin-bottom: 15px; transition: opacity 0.3s ease-in-out;">
            <h3>AHP - Analytic Hierarchy Process</h3>
            <p>AHP ranks genes using t-test, entropy, Wilcoxon, ROC, and SNR. Scores are computed with eigenvectors from pairwise comparison matrices to identify strong biomarkers.</p>
          </div>

          <div style="display: flex; justify-content: space-between;">
            <button id="prev-btn" style="padding: 10px 20px; background-color: #ccc; border: none; border-radius: 6px; font-weight: bold; color: #333;">← Previous</button>
            <button id="next-btn" style="padding: 10px 20px; background-color: #663399; color: white; border: none; border-radius: 6px; font-weight: bold;">Next →</button>
          </div>
        </div>

        <script>
          const steps = [
            {
              title: "AHP - Analytic Hierarchy Process",
              body: "AHP ranks genes using t-test, entropy, Wilcoxon, ROC, and SNR. Scores are computed with eigenvectors from pairwise comparison matrices to identify strong biomarkers."
            },
            {
              title: "CatBoost Classifier",
              body: "CatBoost uses gene expression and clinical features to predict stage probabilities. It handles imbalanced data and outputs likelihoods per stage."
            },
            {
              title: "Cox Proportional Hazards Model",
              body: "The Cox model estimates how variables like tumor size, lymph nodes, and demographics affect survival risk. Outputs hazard ratios and significance levels."
            },
            {
              title: "Random Forest Feature Importance",
              body: "Random Forest ranks clinical features by how much they improve classification. It captures nonlinear interactions and identifies key predictors in diagnosis."
            }
          ];

          let current = 0;
          const stepper = document.getElementById("stepper");
          const prev = document.getElementById("prev-btn");
          const next = document.getElementById("next-btn");

          function updateStepper(index) {
            stepper.style.opacity = 0;
            setTimeout(() => {
              stepper.innerHTML = `
                <h3>${steps[index].title}</h3>
                <p>${steps[index].body}</p>
              `;
              stepper.style.opacity = 1;
            }, 200);

            prev.disabled = index === 0;
            next.disabled = index === steps.length - 1;

            prev.style.backgroundColor = prev.disabled ? '#eee' : '#663399';
            prev.style.color = prev.disabled ? '#aaa' : '#fff';

            next.style.backgroundColor = next.disabled ? '#eee' : '#663399';
            next.style.color = next.disabled ? '#aaa' : '#fff';
          }

          prev.addEventListener("click", () => {
            if (current > 0) {
              current--;
              updateStepper(current);
            }
          });

          next.addEventListener("click", () => {
            if (current < steps.length - 1) {
              current++;
              updateStepper(current);
            }
          });

          // Initialize
          updateStepper(0);
        </script>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    import base64
    def create_card(icon_html: str, title: str, desc: str, button_label: str, path: str):
        return mo.md(f"""
        <style>
        .card-container {{
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 20px;
            padding: 20px;
        }}
        .card {{
            width: 250px;
            height: 300px;
            border: 1px solid #ccc;
            padding: 16px;
            border-radius: 8px;
            background: #fff;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
            text-align: center;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}
        .card-icon img {{
            display: block;
            margin: 0 auto 10px;
            width: 50px;
            height: 50px;
        }}
        .card-title {{
            font-size: 1.2em;
            margin-bottom: 10px;
        }}
        .card-desc {{
            flex-grow: 1;
            margin-bottom: 10px;
        }}
        .card-button {{
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            background-color: #007bff;
            color: white !important;
            cursor: pointer;
            text-decoration: none;
        }}
        </style>
        <div class="card">
            <div class="card-icon">{icon_html}</div>
            <div class="card-title"><strong>{title}</strong></div>
            <div class="card-desc">{desc}</div>
            <a href="{path}" class="card-button">{button_label}</a>
        </div>
        """)



    # Define the icon HTML (e.g., an image tag)
    with open("chip.png", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    data_uri = f"data:image/png;base64,{encoded_string}"

    icon_ml = f'<img src="{data_uri}" alt="ML Icon">'

    # Create cards
    card1 = create_card(
        icon_html=icon_ml,
        title="Machine Learning",
        desc="Explore predictive models used for breast cancer classification and compare their performance.",
        button_label="Learn More",
        path="/ml"
    )
    # Define the icon HTML (e.g., an image tag)
    with open("exploratory-analysis.png", "rb") as image_file:
        encoded_string2 = base64.b64encode(image_file.read()).decode('utf-8')
    data_uri2 = f"data:image/png;base64,{encoded_string2}"
    icon_ml2 = f'<img src="{data_uri2}" alt="EDA Icon">'
    card2 = create_card(
        icon_html=icon_ml2,
        title="Objective and Data Exploration",
        desc="Explore extracted data, perform feature selection, and prepare data from supplementary files.",
        button_label="Read More",
        path="/eda"
    )
    # Define the icon HTML (e.g., an image tag)
    with open("monitor.png", "rb") as image_file:
        encoded_string3 = base64.b64encode(image_file.read()).decode('utf-8')
    data_uri3 = f"data:image/png;base64,{encoded_string3}"
    icon_ml3 = f'<img src="{data_uri3}" alt="EDA Icon">'
    card3 = create_card(
        icon_html=icon_ml3,
        title="Major Findings",
        desc="Perform deep analysis of breast cancer biomarkers and prognosis with intuitive visuals.",
        button_label="Read More",
        path="/major_findings"
    )

    # Display the cards within a container
    mo.md(f"""
    <div class="card-container">
        {card1}
        {card2}
        {card3}
    </div>
    """)
    return (
        base64,
        card1,
        card2,
        card3,
        create_card,
        data_uri,
        data_uri2,
        data_uri3,
        encoded_string,
        encoded_string2,
        encoded_string3,
        icon_ml,
        icon_ml2,
        icon_ml3,
        image_file,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""We implemented this data exploration tools for user to explore our data and provide ability to gain some insight vrom the visuals:""")
    return


@app.cell(hide_code=True)
def _():
    import pandas as pd
    # Load datasets
    ahp_df = pd.read_csv('./AHPresults/final_Mod_ahp_scores.csv')
    stage_df = pd.read_csv('./AHPresults/fina_Stage_unaugmented.csv')
    stage_df = stage_df.iloc[:,:-1700]
    return ahp_df, pd, stage_df


@app.cell
def _(stage_df):
    valid_stages = stage_df['Stage'].value_counts()
    valid_stages = valid_stages[valid_stages >= 14].index

    # Sample 14 rows from each of those stages
    sampled_df = (
        stage_df[stage_df['Stage'].isin(valid_stages)]
        .groupby("Stage", group_keys=False)
        .apply(lambda x: x.sample(n=14, random_state=42))
        .reset_index(drop=True)
    )
    return sampled_df, valid_stages


@app.cell(hide_code=True)
def _(ahp_df, mo, sampled_df):
    # Dictionary of datasets
    datasets = {
        "AHP Analysis": ahp_df,
        "Cancer Stage DataSet": sampled_df,
    }
    # UI: Dropdown
    dropdown = mo.ui.dropdown(
        label="Select Dataset",
        options=list(datasets.keys()),
        value="AHP Analysis" # default
    )
    return datasets, dropdown


@app.cell
def _(datasets, dropdown, mo):
    def show_data():
            selected = dropdown.value
            df = datasets[selected]
            return mo.ui.data_explorer(df)
    mo.vstack([
        dropdown,
        show_data()
    ])
    return (show_data,)


if __name__ == "__main__":
    app.run()
