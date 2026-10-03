#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 21:37:33 2026

@author: beehive
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from distribution_fitter import (
    load_from_paste,
    fit_best_distribution,
    extract_winning_parameters,
    calculate_moments
)


# ==================================================
# PAGE SETUP
# ==================================================

st.set_page_config(
    page_title="Distribution Fit Explorer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Distribution Fit Explorer")

st.write(
    "Upload your dataset or paste numerical values "
    "to find the best-fitting probability distribution."
)

st.info(
    "👋 **How to use:** Paste numerical values or upload a CSV/Excel file, "
    "then click **Analyse Data**. The app will automatically identify whether "
    "your data is continuous or discrete, test multiple probability "
    "distributions, and identify the best-fitting model."
)


# ==================================================
# 1. DATA INPUT
# ==================================================

st.header("1. Input Your Data")


# ==================================================
# INPUT FORM
# ==================================================

with st.form("data_input_form"):

    input_method = st.radio(
        "How would you like to enter your data?",
        ["Paste Values", "Upload CSV/Excel"]
    )

    pasted_data = ""
    uploaded_file = None


    if input_method == "Paste Values":

        pasted_data = st.text_area(
            "Paste your numerical values:",
            placeholder="Example: 12.5, 13.2, 14.8, 15.1..."
        )

    else:

        uploaded_file = st.file_uploader(
            "Upload your dataset",
            type=["csv", "xlsx", "xls"]
        )


    analyse_button = st.form_submit_button(
        "🔍 Analyse Data",
        type="primary"
    )


# ==================================================
# RUN ANALYSIS
# ==================================================

if analyse_button:

    try:

        # ==================================================
        # STEP 1: LOAD DATA
        # ==================================================

        if input_method == "Paste Values":

            # Empty input
            if pasted_data.strip() == "":

                st.error(
                    "⚠️ Please enter some numerical values before analysing."
                )

                st.stop()


            # Try reading pasted values
            try:
                # Allow commas, spaces and new lines
                cleaned_data = pasted_data.replace(",", " ")
            
                values = cleaned_data.split()
            
                data = np.array(
                    [float(value) for value in values],
                    dtype=float
                )
            
            except ValueError:
                st.error(
                    "⚠️ Invalid data detected. Please enter numerical values "
                    "only, separated by commas, spaces, or new lines."
                )
                st.stop()


            # NaN or infinite values
            if not np.all(
                np.isfinite(data)
            ):

                st.error(
                    "⚠️ Your data contains missing or invalid values. "
                    "Please remove them and try again."
                )

                st.stop()


        # ==================================================
        # FILE UPLOAD
        # ==================================================

        else:

            if uploaded_file is None:

                st.error(
                    "⚠️ Please upload a CSV or Excel file first."
                )

                st.stop()


            file_name = uploaded_file.name.lower()


            # ----------------------------------------------
            # READ FILE
            # ----------------------------------------------

            try:

                if file_name.endswith(".csv"):

                    df = pd.read_csv(
                        uploaded_file
                    )

                elif (
                    file_name.endswith(".xlsx")
                    or file_name.endswith(".xls")
                ):

                    df = pd.read_excel(
                        uploaded_file
                    )

                else:

                    st.error(
                        "⚠️ Unsupported file type. "
                        "Please upload a CSV or Excel file."
                    )

                    st.stop()

            except Exception:

                st.error(
                    "⚠️ The file could not be read. "
                    "Please check that it is a valid CSV or Excel file."
                )

                st.stop()


            # ----------------------------------------------
            # KEEP NUMERICAL COLUMNS
            # ----------------------------------------------

            numeric_df = df.select_dtypes(
                include=[np.number]
            )


            if numeric_df.empty:

                st.error(
                    "⚠️ No numerical data was found in the uploaded file."
                )

                st.stop()


            # Convert to one numerical array
            data = (
                numeric_df
                .to_numpy()
                .flatten()
            )


            # Remove missing values
            data = data[
                ~np.isnan(data)
            ]


            # Too few valid observations
            if len(data) < 5:

                st.error(
                    "⚠️ Too few valid observations were found. "
                    "Please provide at least 5 numerical values."
                )

                st.stop()


            # Infinite / invalid values
            if not np.all(
                np.isfinite(data)
            ):

                st.error(
                    "⚠️ The uploaded data contains invalid or "
                    "infinite numerical values."
                )

                st.stop()


        # ==================================================
        # STEP 2: FIT DISTRIBUTIONS
        # ==================================================

        results, discrete = fit_best_distribution(
            data
        )


        # Check fitting results
        if results.empty:

            raise ValueError(
                "No distributions could be fitted to the data."
            )


        # ==================================================
        # STEP 3: FIND WINNING DISTRIBUTION
        # ==================================================

        winning_dist, param_dict = (
            extract_winning_parameters(
                results,
                discrete
            )
        )


        st.success(
            "✅ Analysis complete!"
        )


        # ==================================================
        # 2. RESULTS
        # ==================================================

        st.header("2. Results")

        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                label="🏆 Best Fit",
                value=winning_dist
            )


        with col2:

            if discrete:
                data_type = "Discrete"

            else:
                data_type = "Continuous"


            st.metric(
                label="📊 Data Type",
                value=data_type
            )


        with col3:

            st.metric(
                label="🔢 Observations",
                value=len(data)
            )


        # ==================================================
        # 3. DISTRIBUTION RANKING
        # ==================================================

        st.header(
            "3. Distribution Ranking"
        )


        display_results = results.copy()


        # Remove distributions that failed to fit
        display_results = display_results.dropna(
            subset=["aic"]
        )


        if discrete:

            display_results = display_results[
                [
                    "distribution",
                    "aic",
                    "chi2_stat",
                    "chi2_pvalue"
                ]
            ]


            display_results.columns = [
                "Distribution",
                "AIC",
                "Chi-Square Statistic",
                "p-value"
            ]


        else:

            display_results = display_results[
                [
                    "distribution",
                    "aic",
                    "ks_stat",
                    "ks_pvalue"
                ]
            ]


            display_results.columns = [
                "Distribution",
                "AIC",
                "KS Statistic",
                "p-value"
            ]


        display_results = display_results.round(
            4
        )


        display_results.insert(
            0,
            "Rank",
            range(
                1,
                len(display_results) + 1
            )
        )


        st.dataframe(
            display_results,
            hide_index=True,
            use_container_width=True
        )


        st.caption(
            "Distributions are ranked using AIC. "
            "A lower AIC indicates a better relative fit "
            "among the distributions tested."
        )


        # ==================================================
        # 4. ESTIMATED PARAMETERS
        # ==================================================

        st.header(
            "4. Estimated Parameters"
        )


        st.write(
            f"Estimated parameters for the "
            f"**{winning_dist} distribution**:"
        )


        friendly_names = {

            "normal": [
                "Mean",
                "Standard Deviation"
            ],

            "exponential": [
                "Location",
                "Scale"
            ],

            "gamma": [
                "Shape",
                "Location",
                "Scale"
            ],

            "lognormal": [
                "Shape",
                "Location",
                "Scale"
            ],

            "beta": [
                "Alpha",
                "Beta",
                "Location",
                "Scale"
            ],

            "uniform": [
                "Location",
                "Scale"
            ],

            "poisson": [
                "Lambda"
            ],

            "binomial": [
                "n",
                "p"
            ]
        }


        parameters = list(
            param_dict.items()
        )


        # Replace generic parameter names
        if (
            len(parameters) > 0
            and str(
                parameters[0][0]
            ).lower().startswith("param")
        ):

            names = friendly_names.get(
                winning_dist.lower(),
                [
                    f"Parameter {i + 1}"
                    for i in range(
                        len(parameters)
                    )
                ]
            )


            parameters = [
                (name, value)
                for name, (_, value)
                in zip(
                    names,
                    parameters
                )
            ]


        if len(parameters) > 0:

            param_cols = st.columns(
                len(parameters)
            )


            for col, (
                parameter,
                value
            ) in zip(
                param_cols,
                parameters
            ):

                with col:

                    st.metric(
                        label=parameter,
                        value=f"{value:.4f}"
                    )


        # ==================================================
        # 5. MOMENTS COMPARISON
        # ==================================================

        st.header(
            "5. Moments Comparison"
        )


        st.write(
            "Compare the moments of the raw data with those "
            f"of the fitted **{winning_dist} distribution**."
        )


        (
            moment_dist,
            raw_moments_df,
            characteristics_df
        ) = calculate_moments(
            data,
            results,
            discrete
        )


        # ----------------------------------------------
        # FIRST FOUR RAW MOMENTS
        # ----------------------------------------------

        st.subheader(
            "First Four Raw Moments"
        )


        moments_display = (
            raw_moments_df.copy()
        )


        numeric_columns = [
            "Raw Data",
            "Winning Distribution",
            "Absolute Difference"
        ]


        moments_display[
            numeric_columns
        ] = moments_display[
            numeric_columns
        ].round(4)


        st.dataframe(
            moments_display,
            hide_index=True,
            use_container_width=True
        )


        st.caption(
            "A smaller absolute difference means the "
            "fitted distribution more closely matches "
            "that moment of the observed data."
        )


        # ----------------------------------------------
        # DISTRIBUTION CHARACTERISTICS
        # ----------------------------------------------

        st.subheader(
            "Distribution Characteristics"
        )


        characteristics_display = (
            characteristics_df.copy()
        )


        characteristics_display[
            numeric_columns
        ] = characteristics_display[
            numeric_columns
        ].round(4)


        st.dataframe(
            characteristics_display,
            hide_index=True,
            use_container_width=True
        )


        st.caption(
            "These statistics compare the shape and spread "
            "of the raw data with the fitted distribution."
        )


        # ==================================================
        # 6. DATA VISUALIZATION
        # ==================================================

        st.header(
            "6. Data Visualization"
        )


        st.write(
            f"Visual comparison of the raw data with the "
            f"best-fitted **{winning_dist} distribution**."
        )


        best_row = results.iloc[0]


        params = tuple(
            best_row["params"]
        )


        # ==================================================
        # IDENTIFY FITTED DISTRIBUTION
        # ==================================================

        if not discrete:

            continuous_distributions = {

                "Normal": stats.norm,

                "Exponential": stats.expon,

                "Gamma": stats.gamma,

                "Lognormal": stats.lognorm,

                "Beta": stats.beta,

                "Uniform": stats.uniform
            }


            fitted_distribution = (
                continuous_distributions[
                    winning_dist
                ]
            )


        else:

            discrete_distributions = {

                "Poisson": stats.poisson,

                "Binomial": stats.binom,

                "Negative Binomial": stats.nbinom,

                "Geometric": stats.geom,

                "Bernoulli": stats.bernoulli,

                "Discrete Uniform": stats.randint
            }


            fitted_distribution = (
                discrete_distributions[
                    winning_dist
                ]
            )


        fitted = fitted_distribution(
            *params
        )


        # ==================================================
        # GRAPH 1
        # ==================================================

        st.subheader(
            "Empirical vs Fitted Distribution"
        )


        fig1, ax1 = plt.subplots(
            figsize=(10, 5)
        )


        # ----------------------------------------------
        # CONTINUOUS
        # ----------------------------------------------

        if not discrete:

            ax1.hist(
                data,
                bins="auto",
                density=True,
                alpha=0.6,
                label="Raw Data"
            )


            data_min = np.min(
                data
            )

            data_max = np.max(
                data
            )


            data_range = (
                data_max - data_min
            )


            if data_range == 0:

                small_offset = 0

            else:

                small_offset = (
                    data_range * 0.005
                )


            x = np.linspace(
                data_min + small_offset,
                data_max - small_offset,
                500
            )


            y = fitted.pdf(
                x
            )


            valid = (
                np.isfinite(y)
                & (y >= 0)
            )


            ax1.plot(
                x[valid],
                y[valid],
                linewidth=2.5,
                label=f"Fitted {winning_dist}"
            )


            ax1.set_ylabel(
                "Density"
            )


        # ----------------------------------------------
        # DISCRETE
        # ----------------------------------------------

        else:

            x = np.arange(
                int(np.min(data)),
                int(np.max(data)) + 1
            )


            empirical_probabilities = np.array(
                [
                    np.mean(
                        data == value
                    )
                    for value in x
                ]
            )


            fitted_probabilities = fitted.pmf(
                x
            )


            ax1.bar(
                x,
                empirical_probabilities,
                alpha=0.6,
                label="Raw Data"
            )


            ax1.plot(
                x,
                fitted_probabilities,
                marker="o",
                linewidth=2.5,
                label=f"Fitted {winning_dist}"
            )


            ax1.set_ylabel(
                "Probability"
            )


        # ----------------------------------------------
        # GRAPH FORMATTING
        # ----------------------------------------------

        ax1.set_title(
            f"Raw Data vs Fitted "
            f"{winning_dist} Distribution"
        )


        ax1.set_xlabel(
            "Observed Value"
        )


        ax1.legend()

        ax1.grid(
            alpha=0.2
        )


        plt.tight_layout()


        st.pyplot(
            fig1
        )


        plt.close(
            fig1
        )


        st.caption(
            "The bars represent the empirical distribution "
            "of the observed data, while the fitted curve "
            "represents the best-fitting theoretical distribution."
        )


        # ==================================================
        # GRAPH 2: CDF
        # ==================================================

        st.subheader(
            "Empirical CDF vs Fitted CDF"
        )


        fig2, ax2 = plt.subplots(
            figsize=(10, 5)
        )


        # ----------------------------------------------
        # CONTINUOUS CDF
        # ----------------------------------------------

        if not discrete:

            sorted_data = np.sort(
                data
            )


            empirical_cdf = (
                np.arange(
                    1,
                    len(sorted_data) + 1
                )
                / len(sorted_data)
            )


            fitted_cdf = fitted.cdf(
                sorted_data
            )


            ax2.step(
                sorted_data,
                empirical_cdf,
                where="post",
                linewidth=2,
                label="Empirical CDF"
            )


            ax2.plot(
                sorted_data,
                fitted_cdf,
                linewidth=2.5,
                label=f"Fitted {winning_dist} CDF"
            )


        # ----------------------------------------------
        # DISCRETE CDF
        # ----------------------------------------------

        else:

            x_cdf = np.arange(
                int(np.min(data)),
                int(np.max(data)) + 1
            )


            empirical_cdf = np.array(
                [
                    np.mean(
                        data <= value
                    )
                    for value in x_cdf
                ]
            )


            fitted_cdf = fitted.cdf(
                x_cdf
            )


            ax2.step(
                x_cdf,
                empirical_cdf,
                where="post",
                linewidth=2,
                label="Empirical CDF"
            )


            ax2.plot(
                x_cdf,
                fitted_cdf,
                marker="o",
                linewidth=2.5,
                label=f"Fitted {winning_dist} CDF"
            )


        # ----------------------------------------------
        # CDF FORMATTING
        # ----------------------------------------------

        ax2.set_title(
            f"Empirical CDF vs Fitted "
            f"{winning_dist} CDF"
        )


        ax2.set_xlabel(
            "Observed Value"
        )


        ax2.set_ylabel(
            "Cumulative Probability"
        )


        ax2.set_ylim(
            0,
            1.05
        )


        ax2.legend()

        ax2.grid(
            alpha=0.2
        )


        plt.tight_layout()


        st.pyplot(
            fig2
        )


        plt.close(
            fig2
        )


        st.caption(
            "The closer the empirical and fitted CDF lines are, "
            "the more closely the theoretical distribution "
            "matches the observed data."
        )


        # ==================================================
        # 7. INTERPRETATION
        # ==================================================

        st.header(
            "7. Interpretation"
        )


        st.write(
            "A simple summary of the distribution fitting results."
        )


        best_result = results.iloc[0]


        # ----------------------------------------------
        # BEST FIT
        # ----------------------------------------------

        st.subheader(
            "🏆 Best-Fitting Distribution"
        )


        st.write(
            f"The **{winning_dist} distribution** was selected "
            f"as the best fit because it achieved the lowest "
            f"AIC of **{best_result['aic']:.4f}** among the "
            f"distributions tested."
        )


        # ----------------------------------------------
        # GOODNESS OF FIT
        # ----------------------------------------------

        st.subheader(
            "📊 Goodness-of-Fit"
        )


        if discrete:

            gof_stat = best_result[
                "chi2_stat"
            ]

            p_value = best_result[
                "chi2_pvalue"
            ]


            st.write(
                f"The Chi-Square statistic is "
                f"**{gof_stat:.4f}**, with a p-value of "
                f"**{p_value:.4f}**."
            )


            st.caption(
                "A smaller Chi-Square statistic indicates "
                "that the fitted probabilities are closer "
                "to the observed frequencies."
            )


        else:

            gof_stat = best_result[
                "ks_stat"
            ]

            p_value = best_result[
                "ks_pvalue"
            ]


            st.write(
                f"The Kolmogorov-Smirnov (KS) statistic is "
                f"**{gof_stat:.4f}**, with a p-value of "
                f"**{p_value:.4f}**."
            )


            st.caption(
                "The KS statistic measures the maximum "
                "difference between the empirical and fitted "
                "cumulative distributions. A smaller value "
                "indicates a closer fit."
            )


        # ----------------------------------------------
        # STATISTICAL INTERPRETATION
        # ----------------------------------------------

        st.subheader(
            "🔎 Statistical Interpretation"
        )


        if p_value >= 0.05:

            st.success(
                f"At the 5% significance level, the "
                f"goodness-of-fit test does not provide "
                f"sufficient evidence to reject the "
                f"{winning_dist} distribution."
            )


        else:

            st.warning(
                f"At the 5% significance level, the "
                f"goodness-of-fit test provides evidence "
                f"against the {winning_dist} distribution. "
                f"Although it ranks best among the distributions "
                f"tested, its absolute fit should be interpreted "
                f"with caution."
            )


        # ----------------------------------------------
        # SUMMARY
        # ----------------------------------------------

        st.subheader(
            "💡 Summary"
        )


        st.info(
            f"Overall, **{winning_dist}** provides the best "
            f"relative fit among the candidate distributions "
            f"tested. Review the ranking table, estimated "
            f"parameters, moment comparisons and visualizations "
            f"above for a complete assessment of the fit."
        )


        # ==================================================
        # GLOSSARY
        # ==================================================

        with st.expander(
            "ℹ️ Understanding the Results"
        ):

            st.markdown(
                """
**AIC (Akaike Information Criterion)**  
Used to compare candidate distributions. A lower AIC indicates
a better relative fit among the distributions tested.

**KS Statistic**  
Used for continuous data. A smaller value indicates that the
fitted cumulative distribution is closer to the empirical data.

**Chi-Square Statistic**  
Used for discrete data. It measures differences between observed
and expected frequencies.

**p-value**  
A small p-value, commonly below 0.05, provides evidence against
the fitted distribution under the goodness-of-fit test.

**Moments**  
Numerical measures describing characteristics of a distribution,
including its location, spread and shape.
                """
            )


    # ==================================================
    # FRIENDLY FALLBACK ERROR
    # ==================================================

    except Exception as e:

        st.error(
            "⚠️ The analysis could not be completed. "
            "Please check your data and try again."
        )

        with st.expander(
            "Technical details"
        ):

            st.code(
                str(e)
            )

