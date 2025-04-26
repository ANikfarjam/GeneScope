import marimo

__generated_with = "0.11.12"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#<span style="color: brown">Summery of Analysis Used in GeneScope's Researches</span>""")
    return


@app.cell(hide_code=True)
def _(mo):
    import anywidget
    import traitlets

    class ShowAnalysis(anywidget.AnyWidget):
        current_step = traitlets.Int(0).tag(sync=True)
    
        _esm = """
        export function render({ model, el }) {
            const container = document.createElement("div");
            container.classList.add("walkthrough");

            const stepper = document.createElement("div");
            stepper.classList.add("stepper");
            container.appendChild(stepper);

            const indicators = document.createElement("div");
            indicators.classList.add("step-indicators");
            container.appendChild(indicators);

            const nav = document.createElement("div");
            nav.classList.add("navigation");

            const prev = document.createElement("button");
            prev.classList.add("prev-btn");
            prev.textContent = "← Previous";
            nav.appendChild(prev);

            const next = document.createElement("button");
            next.classList.add("next-btn");
            next.textContent = "Next →";
            nav.appendChild(next);

            container.appendChild(nav);
            el.appendChild(container);

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

            function renderIndicators(current) {
                indicators.innerHTML = steps.map((_, idx) => `
                    <span class="dot ${idx === current ? 'active' : ''}"></span>
                `).join('');
            }

            function updateStepper(current) {
                stepper.style.opacity = 0;
                setTimeout(() => {
                    stepper.innerHTML = `
                        <h3>${steps[current].title}</h3>
                        <p>${steps[current].body}</p>
                    `;
                    stepper.style.opacity = 1;
                }, 200);

                prev.disabled = current === 0;
                next.disabled = current === steps.length - 1;

                prev.classList.toggle("disabled", prev.disabled);
                next.classList.toggle("disabled", next.disabled);

                renderIndicators(current);
            }

            model.on("change:current_step", () => {
                updateStepper(model.get("current_step"));
            });

            prev.addEventListener("click", () => {
                if (model.get("current_step") > 0) {
                    model.set("current_step", model.get("current_step") - 1);
                    model.save_changes();
                }
            });

            next.addEventListener("click", () => {
                if (model.get("current_step") < steps.length - 1) {
                    model.set("current_step", model.get("current_step") + 1);
                    model.save_changes();
                }
            });

            updateStepper(model.get("current_step"));
        }

        export default { render };
        """

        _css = """
        .walkthrough {
            font-family: sans-serif;
            max-width: 700px;
            margin: auto;
        }
        .stepper {
            background: #f7f7f7;
            padding: 20px;
            border-left: 4px solid #663399;
            border-radius: 8px;
            min-height: 160px;
            margin-bottom: 15px;
            transition: opacity 0.3s ease-in-out;
        }
        .step-indicators {
            text-align: center;
            margin-bottom: 10px;
        }
        .dot {
            height: 12px;
            width: 12px;
            margin: 0 5px;
            background-color: #ccc;
            border-radius: 50%;
            display: inline-block;
            transition: background-color 0.3s;
        }
        .dot.active {
            background-color: #663399;
        }
        .navigation {
            display: flex;
            justify-content: space-between;
        }
        .prev-btn, .next-btn {
            padding: 10px 20px;
            border: none;
            border-radius: 6px;
            font-weight: bold;
        }
        .prev-btn {
            background-color: #eee;
            color: #aaa;
        }
        .prev-btn:not(.disabled) {
            background-color: #663399;
            color: #fff;
        }
        .next-btn {
            background-color: #663399;
            color: white;
        }
        .next-btn.disabled {
            background-color: #eee;
            color: #aaa;
        }
        """

    # Initialize properly in Marimo
    show_analysis = mo.ui.anywidget(ShowAnalysis())
    show_analysis

    return ShowAnalysis, anywidget, show_analysis, traitlets


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
    mo.md(r"""##<span style="color:brown">Data Expoler</span>""")
    return


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


@app.cell(hide_code=True)
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