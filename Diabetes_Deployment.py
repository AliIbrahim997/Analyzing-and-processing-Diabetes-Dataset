
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(layout= 'wide', page_title= 'Diabetes Analysis')

html_title = """<h1 style="color:white;text-align:center;"> Diabetes Exploratory Data Analysis </h1>"""
st.markdown(html_title, unsafe_allow_html=True)

# Add Image
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("Snipaste_2025-12-01_20-48-13.png",  use_container_width=True)


# Read Data
df = pd.read_csv('diabetes_dataset-updated.csv', index_col= 0)

page = st.sidebar.radio('Page', ["Data Exploration", "Univariate Analysis", "Bivariate Analysis", "Conclusion"])

if page == "Data Exploration":

    # Dataframe
    st.subheader('Dataset Overview')
    st.dataframe(df)

    # Data Description
    column_descriptions = {
    "age": "Age of patient in years.",
    "gender": "Patient gender.",
    "ethnicity": "Ethnic background.",
    "education_level": "Highest completed education.",
    "income_level": "Income category.",
    "employment_status": "Employment type.",
    "smoking_status": "Smoking behavior.",
    "alcohol_consumption_per_week": "Drinks consumed per week.",
    "physical_activity_minutes_per_week": "Physical activity (weekly minutes).",
    "diet_score": "Diet quality (higher = healthier).",
    "sleep_hours_per_day": "Average daily sleep hours.",
    "screen_time_hours_per_day": "Average daily screen time hours.",
    "family_history_diabetes": "Family history of diabetes.",
    "hypertension_history": "Hypertension history.",
    "cardiovascular_history": "Cardiovascular history.",
    "bmi": "Body Mass Index (kg/m²).",
    "waist_to_hip_ratio": "Waist-to-hip ratio.",
    "systolic_bp": "Systolic blood pressure (mmHg).",
    "diastolic_bp": "Diastolic blood pressure (mmHg).",
    "heart_rate": "Resting heart rate (bpm).",
    "cholesterol_total": "Total cholesterol (mg/dL).",
    "hdl_cholesterol": "High-Density Lipoprotein Cholesterol (mg/dL).",
    "ldl_cholesterol": "Low-Density Lipoprotein Cholesterol (mg/dL).",
    "triglycerides": "Triglycerides (mg/dL).",
    "glucose_fasting": "Fasting glucose (mg/dL).",
    "glucose_postprandial": "Post-meal glucose (mg/dL).",
    "insulin_level": "Blood insulin level (µU/mL).",
    "hba1c": "HbA1c (%) — percentage of hemoglobin with glucose attached.",
    "diabetes_risk_score": "Risk score (calculated, 0–100).",
    "diabetes_stage": "Stage of diabetes.",
    "diagnosed_diabetes": "Target — Diabetes diagnosis (e.g., 0/1 or label)."}

    # Create a table for descriptions
    desc_df = pd.DataFrame(list(column_descriptions.items()), columns=["Column Name", "Description"])

    # Display table
    st.subheader("📝 Column Descriptions")
    st.table(desc_df)

elif page == "Univariate Analysis":

    st.title('Choose a feature to visualize')

    # Create Numerical and Categorical Tabs
    tab_num, tab_cat = st.tabs(['Numerical', 'Categorical'])

    # Select Numerical Column
    num_cols = df.select_dtypes(include= 'number').columns.drop(['alcohol_consumption_per_week', 'diagnosed_diabetes'])
    selected_num_col = tab_num.selectbox('Column', num_cols)

    # Select Chart
    tab_num.plotly_chart(px.histogram(data_frame = df, x = selected_num_col, title = selected_num_col))

    # Select Catgorical Column
    cat_cols = df.select_dtypes(include= 'object').columns.drop(["employment_status"])
    selected_cat_col = tab_cat.selectbox('Column', cat_cols)

    # Select Chart
    selected_chart = tab_cat.selectbox('Chart', ['Bar', 'Pie'])

    if selected_chart == 'Bar':
        tab_cat.plotly_chart(px.histogram(data_frame = df, x = selected_cat_col, title = selected_cat_col).update_xaxes(categoryorder = 'max descending'))

    elif selected_chart == 'Pie':
        tab_cat.plotly_chart(px.pie(data_frame = df, names = selected_cat_col, title = selected_cat_col))

elif page == "Bivariate Analysis":

    st.title("Choose a feature to visualize against another")

    # Work on a copy so we don't mutate original df
    df_clean = df.copy()

    # Helper: try converting object columns that are numeric-like to numeric
    def smart_convert(col):
        if pd.api.types.is_object_dtype(df_clean[col]):
            # strip whitespace from strings
            df_clean[col] = df_clean[col].astype(str).str.strip()
            # attempt numeric conversion (coerce invalids to NaN)
            coerced = pd.to_numeric(df_clean[col], errors="coerce")
            # if many values convert, replace column
            non_na_ratio = coerced.notna().mean()
            if non_na_ratio > 0.6:  # threshold, tune as needed
                df_clean[col] = coerced
            else:
                # keep as categorical-like (but still strip)
                df_clean[col] = df_clean[col].astype("category")
        elif pd.api.types.is_categorical_dtype(df_clean[col]):
            # ensure categories are trimmed strings
            df_clean[col] = df_clean[col].astype(str).str.strip().astype("category")
        else:
            # numeric columns: nothing to do
            pass

    # Apply to all columns
    for c in df_clean.columns:
        smart_convert(c)

    # Now let user pick columns
    col1, col2 = st.columns([1, 1])
    with col1:
        x_col = st.selectbox("Select X-axis Column", df_clean.columns, index=0, key="biv_x")
    with col2:
        y_col = st.selectbox("Select Y-axis Column", df_clean.columns, index=1 if len(df_clean.columns)>1 else 0, key="biv_y")

    # Show a quick sanity-check summary for the chosen columns (helps catch weird values)
    st.markdown("**Column sanity checks**")
    info_col1, info_col2 = st.columns(2)
    with info_col1:
        st.write(f"**{x_col}** — dtype: `{df_clean[x_col].dtype}` — unique: {df_clean[x_col].nunique()}")
        st.write(df_clean[x_col].dropna().astype(str).unique()[:10])
    with info_col2:
        st.write(f"**{y_col}** — dtype: `{df_clean[y_col].dtype}` — unique: {df_clean[y_col].nunique()}")
        st.write(df_clean[y_col].dropna().astype(str).unique()[:10])

    # Prevent plotting empty columns
    if df_clean[x_col].dropna().empty or df_clean[y_col].dropna().empty:
        st.error("One of the selected columns contains only missing values — choose different columns.")
    else:
        # If user accidentally picked the same column twice: show a meaningful plot
        if x_col == y_col:
            st.warning("You've selected the same column for X and Y — showing a distribution + value counts instead.")
            if pd.api.types.is_numeric_dtype(df_clean[x_col]):
                fig = px.histogram(df_clean, x=x_col, title=f"Distribution of {x_col}")
                st.plotly_chart(fig, use_container_width=True)
                st.write("Summary statistics:")
                st.write(df_clean[x_col].describe())
            else:
                # categorical: show bar and table of top categories
                fig = px.histogram(df_clean, x=x_col, title=f"Value counts of {x_col}").update_xaxes(categoryorder="total descending")
                st.plotly_chart(fig, use_container_width=True)
                st.write(df_clean[x_col].value_counts().head(20))
        else:
            # classify types
            x_is_num = pd.api.types.is_numeric_dtype(df_clean[x_col])
            y_is_num = pd.api.types.is_numeric_dtype(df_clean[y_col])
            x_is_cat = pd.api.types.is_categorical_dtype(df_clean[x_col]) or pd.api.types.is_object_dtype(df_clean[x_col])
            y_is_cat = pd.api.types.is_categorical_dtype(df_clean[y_col]) or pd.api.types.is_object_dtype(df_clean[y_col])

            # Numeric vs Numeric -> scatter + optional trendline
            if x_is_num and y_is_num:
                fig = px.scatter(df_clean, x=x_col, y=y_col, trendline="ols",
                                 title=f"Scatter: {x_col} vs {y_col}")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("**Correlation**")
                corr = df_clean[[x_col, y_col]].dropna().corr().iloc[0,1]
                st.write(f"Pearson r = {corr:.3f}")

            # Categorical vs Numeric -> box + violin
            elif (x_is_cat and y_is_num) or (x_is_num and y_is_cat):
                # ensure category is made explicit
                if x_is_cat:
                    cat_col, num_col = x_col, y_col
                else:
                    cat_col, num_col = y_col, x_col

                # collapse tiny categories into "Other" to keep charts readable
                vc = df_clean[cat_col].value_counts(normalize=True)
                small_categories = vc[vc < 0.02].index  # categories <2% -> other
                if len(small_categories) > 0:
                    df_clean["_cat_cleaned"] = df_clean[cat_col].apply(lambda v: "Other" if v in small_categories else v)
                    cat_to_use = "_cat_cleaned"
                else:
                    cat_to_use = cat_col

                tab1, tab2 = st.tabs(["Boxplot", "Violin Plot"])
                with tab1:
                    fig = px.box(df_clean, x=cat_to_use, y=num_col,
                                 title=f"{num_col} distribution by {cat_col}")
                    st.plotly_chart(fig, use_container_width=True)
                with tab2:
                    fig = px.violin(df_clean, x=cat_to_use, y=num_col, box=True,
                                    title=f"{num_col} violin by {cat_col}")
                    st.plotly_chart(fig, use_container_width=True)

            # Categorical vs Categorical -> grouped bar
            elif x_is_cat and y_is_cat:
                # collapse tiny categories if needed
                def collapse_small(colname):
                    vc = df_clean[colname].value_counts(normalize=True)
                    small = vc[vc < 0.02].index
                    if len(small) > 0:
                        newcol = f"{colname}_collapsed"
                        df_clean[newcol] = df_clean[colname].apply(lambda v: "Other" if v in small else v)
                        return newcol
                    return colname

                x_use = collapse_small(x_col)
                y_use = collapse_small(y_col)

                grouped = df_clean.groupby([x_use, y_use]).size().reset_index(name="count")
                fig = px.bar(grouped, x=x_use, y="count", color=y_use, barmode="group",
                             title=f"{x_col} vs {y_col} (counts)")
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.info("Columns not detected as numeric or categorical cleanly — attempting fallback plot (histogram of X).")
                fig = px.histogram(df_clean, x=x_col, title=f"Distribution of {x_col}")
                st.plotly_chart(fig, use_container_width=True)


elif page == 'Conclusion':


    st.title("Conclusion & Key Insights")

    st.markdown("""
    ## 📌 Overall Insights from the Dataset

    1. **Lifestyle factors** such as physical activity, sleep duration, diet score, and screen time have strong associations with diabetes risk.
    2. **Blood indicators** (fasting glucose, postprandial glucose, insulin, HbA1c) show predictable upward trends across diabetes stages.
    3. **BMI, cholesterol levels, and waist-to-hip ratio** are strong predictors of diabetes progression.
    4. **Family history, hypertension, and cardiovascular history** significantly increase diabetes likelihood.

    ---

    ## 🌍 If Viewed from a Non-Profit Diabetes Awareness Organization Perspective

    1. **Identify high-risk communities** based on demographic factors (age, ethnicity, family history) to allocate resources more efficiently.
    2. Use the dataset to **design targeted awareness campaigns**, focusing on groups with the highest lifestyle-related risks such as:
       - Low physical activity  
       - Poor diet score  
       - Excessive screen time  
    3. Develop **free or low-cost screening programs** in areas with:
       - High cholesterol levels  
       - High glucose or HbA1c averages  
       - High prevalence of obesity (BMI and waist-to-hip ratio)
    4. Build **education programs** emphasizing:
       - Benefits of physical activity  
       - Healthy diet choices  
       - Importance of early testing  
    5. Use the risk patterns in the dataset to **partner with schools, workplaces, and community centers** for broad prevention strategies.
    6. Prioritize interventions for groups with both **lifestyle risks + medical risks** to maximize prevention impact.

    ---

    ## 💰 If Viewed from a “Profit-Driven Medical Company” Perspective

    1. **High-risk groups** (high BMI, high glucose, family history) can be targeted with:
       - Early screening packages  
       - Subscription-based monitoring  
       - Preventive wellness programs  

    2. **Patients with borderline HbA1c** are ideal candidates for:
       - Monthly follow-up plans  
       - Lifestyle coaching upsells  
       - Continuous glucose monitor sales  

    3. **Long-term diabetic patients** represent:
       - Higher pharmaceutical lifetime value  
       - Recurring diagnostics and lab testing  

    4. **Cross-selling opportunities:**
       - Cholesterol-lowering medication  
       - Blood pressure control programs  
       - Heart health diagnostics  

    ---

    ## 🧭 Final Recommendation

    This dataset is powerful for:
    - **Predictive modeling** of diabetes risk  
    - **Identifying profitable patient segments**  
    - **Designing data-driven health programs**  
    - **Public health interventions**  

    Use the risk score + lifestyle variables together for the strongest prediction models.
    """)
