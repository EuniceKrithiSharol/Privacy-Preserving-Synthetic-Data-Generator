import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances


st.set_page_config(
    page_title="Synthetic Data Generator",
    page_icon="🧬",
    layout="wide"
)


st.title("🧬 Privacy-Preserving Synthetic Data Generator")

st.write(
    "Generate synthetic datasets while preserving the statistical "
    "characteristics of the original data."
)


st.sidebar.header("Dataset Upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload Original Dataset",
    type=["csv"]
)


def generate_synthetic_data(data):

    synthetic_data = pd.DataFrame()

    numeric_columns = data.select_dtypes(
        include=np.number
    ).columns

    categorical_columns = data.select_dtypes(
        exclude=np.number
    ).columns


    for column in numeric_columns:

        column_data = data[column].dropna()

        mean = column_data.mean()
        std = column_data.std()

        synthetic_values = np.random.normal(
            loc=mean,
            scale=std,
            size=len(data)
        )

        synthetic_data[column] = synthetic_values


    for column in categorical_columns:

        probabilities = (
            data[column]
            .value_counts(normalize=True)
        )

        synthetic_values = np.random.choice(
            probabilities.index,
            size=len(data),
            p=probabilities.values
        )

        synthetic_data[column] = synthetic_values


    return synthetic_data


if uploaded_file is not None:

    original_df = pd.read_csv(
        uploaded_file
    )

    st.success(
        "Dataset loaded successfully!"
    )


    st.header("📁 Original Dataset")

    st.dataframe(
        original_df.head(),
        use_container_width=True
    )


    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Rows",
            original_df.shape[0]
        )

    with col2:

        st.metric(
            "Columns",
            original_df.shape[1]
        )

    with col3:

        st.metric(
            "Missing Values",
            original_df.isnull().sum().sum()
        )


    if st.button(
        "Generate Synthetic Dataset"
    ):

        synthetic_df = generate_synthetic_data(
            original_df
        )

        st.session_state[
            "synthetic_data"
        ] = synthetic_df


    if "synthetic_data" in st.session_state:

        synthetic_df = (
            st.session_state[
                "synthetic_data"
            ]
        )


        st.divider()

        st.header(
            "🧬 Generated Synthetic Dataset"
        )

        st.dataframe(
            synthetic_df.head(),
            use_container_width=True
        )


        st.divider()

        st.header(
            "📊 Statistical Similarity Analysis"
        )


        numeric_columns = original_df.select_dtypes(
            include=np.number
        ).columns


        comparison_results = []


        for column in numeric_columns:

            original_mean = (
                original_df[column]
                .mean()
            )

            synthetic_mean = (
                synthetic_df[column]
                .mean()
            )

            original_std = (
                original_df[column]
                .std()
            )

            synthetic_std = (
                synthetic_df[column]
                .std()
            )


            mean_difference = abs(
                original_mean -
                synthetic_mean
            )

            comparison_results.append(
                {
                    "Feature": column,
                    "Original Mean": round(
                        original_mean,
                        2
                    ),
                    "Synthetic Mean": round(
                        synthetic_mean,
                        2
                    ),
                    "Original Std": round(
                        original_std,
                        2
                    ),
                    "Synthetic Std": round(
                        synthetic_std,
                        2
                    ),
                    "Mean Difference": round(
                        mean_difference,
                        2
                    )
                }
            )


        comparison_df = pd.DataFrame(
            comparison_results
        )


        st.subheader(
            "Feature Statistics Comparison"
        )

        st.dataframe(
            comparison_df,
            use_container_width=True
        )


        st.divider()

        st.header(
            "📈 Distribution Comparison"
        )


        selected_column = st.selectbox(
            "Select a Numeric Feature",
            numeric_columns
        )


        original_plot = pd.DataFrame(
            {
                "Value": original_df[
                    selected_column
                ],
                "Dataset": "Original"
            }
        )


        synthetic_plot = pd.DataFrame(
            {
                "Value": synthetic_df[
                    selected_column
                ],
                "Dataset": "Synthetic"
            }
        )


        plot_data = pd.concat(
            [
                original_plot,
                synthetic_plot
            ],
            ignore_index=True
        )


        fig = px.histogram(
            plot_data,
            x="Value",
            color="Dataset",
            barmode="overlay",
            nbins=30,
            title=(
                f"Original vs Synthetic Distribution: "
                f"{selected_column}"
            )
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        st.divider()

        st.header(
            "🔐 Privacy Assessment"
        )


        st.write(
            "The synthetic dataset is generated from statistical "
            "patterns rather than copying individual original records."
        )


        numeric_data = (
            original_df.select_dtypes(
                include=np.number
            )
            .dropna()
        )


        if len(numeric_data) > 1:

            sample_size = min(
                len(numeric_data),
                len(synthetic_df)
            )


            original_sample = (
                numeric_data
                .iloc[:sample_size]
            )


            synthetic_numeric = (
                synthetic_df[
                    numeric_data.columns
                ]
                .iloc[:sample_size]
            )


            scaler = StandardScaler()

            original_scaled = scaler.fit_transform(
                original_sample
            )

            synthetic_scaled = scaler.transform(
                synthetic_numeric
            )


            distances = np.linalg.norm(
                original_scaled -
                synthetic_scaled,
                axis=1
            )


            average_distance = distances.mean()


            st.metric(
                "Average Synthetic Record Distance",
                f"{average_distance:.2f}"
            )


            if average_distance > 1:

                st.success(
                    "Privacy Level: Strong"
                )

            elif average_distance > 0.5:

                st.warning(
                    "Privacy Level: Moderate"
                )

            else:

                st.error(
                    "Privacy Level: Low"
                )


        st.divider()


        st.header(
            "⬇️ Download Synthetic Dataset"
        )


        csv = synthetic_df.to_csv(
            index=False
        ).encode("utf-8")


        st.download_button(
            label="Download Synthetic CSV",
            data=csv,
            file_name="synthetic_dataset.csv",
            mime="text/csv"
        )


else:

    st.info(
        "Upload a CSV dataset to generate "
        "a privacy-preserving synthetic version."
    )
