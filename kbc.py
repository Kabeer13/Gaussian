import numpy as np

class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.prior = {}
        
        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0)
            self.prior[c] = X_c.shape[0] / X.shape[0]

    def gaussian_pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(- (x - mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def predict(self, X):
        predictions = []
        
        for x in X:
            posteriors = []
            
            for c in self.classes:
                prior = np.log(self.prior[c])
                likelihood = np.sum(np.log(self.gaussian_pdf(c, x)))
                posterior = prior + likelihood
                posteriors.append(posterior)
            
            predictions.append(self.classes[np.argmax(posteriors)])
        
        return np.array(predictions)


# ----- simple command-line application -----

def _ask(prompt, default=None):
    """Utility to prompt user with an optional default value."""
    if default is not None:
        prompt = f"{prompt} [{default}]: "
    else:
        prompt = f"{prompt}: "
    ans = input(prompt)
    if ans == "" and default is not None:
        return default
    return ans


def run_cli():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, classification_report

    print("Gaussian Naive Bayes CLI")
    path = _ask("Enter path to CSV dataset")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Failed to read dataset: {e}")
        return

    print("Columns in dataset:\n", df.columns.tolist())
    target = _ask("Enter target variable column name")
    if target not in df.columns:
        print(f"Column '{target}' not found in dataset.")
        return

    features_input = _ask("Enter feature column names separated by commas")
    features = [f.strip() for f in features_input.split(",") if f.strip()]
    for f in features:
        if f not in df.columns:
            print(f"Feature column '{f}' not found in dataset.")
            return

    test_size = float(_ask("Enter test split fraction (0-1)", "0.2"))
    if not 0 < test_size < 1:
        print("Test split must be between 0 and 1.")
        return

    X = df[features].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    train_pct = (1 - test_size) * 100
    test_pct = test_size * 100
    print(f"\nTrain set: {len(X_train)} samples ({train_pct:.2f}%)")
    print(f"Test set:  {len(X_test)} samples ({test_pct:.2f}%)\n")

    model = GaussianNaiveBayes()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))


def run_streamlit():
    import pandas as pd
    import streamlit as st
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, classification_report

    st.title("Gaussian Naive Bayes Explorer")
    st.write("Upload a CSV dataset to explore or model with Gaussian Naive Bayes.")

    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is None:
        st.info("Please upload a dataset to get started.")
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Unable to read CSV: {e}")
        return

    # create two tabs: Data Analysis and ML
    tab1, tab2 = st.tabs(["Data Analysis", "ML"])

    with tab1:
        st.header("Data Analysis")
        st.write("### Columns")
        st.write(df.columns.tolist())
        st.write("### Preview")
        st.dataframe(df.head())
        st.write("### Summary statistics")
        st.dataframe(df.describe(include='all'))

    with tab2:
        st.header("Model training")
        st.markdown("""
- **Classification** should use continous values automatically
- **Regretion** should use discrete
""")
        task = st.radio("Task type", ["Classification", "Regression"], key="task")

        # determine which columns are eligible as target based on task
        if task == "Classification":
            # classification prefers categorical/non-numeric targets
            candidates = df.select_dtypes(exclude=[np.number]).columns.tolist()
        else:
            # regression prefers numeric targets
            candidates = df.select_dtypes(include=[np.number]).columns.tolist()

        if not candidates:
            st.error(f"No suitable target columns found for {task}.")
            return

        target = st.selectbox("Target variable", candidates)
        features = st.multiselect("Features", [c for c in df.columns if c != target])

        test_size = st.slider("Test fraction", min_value=0.05, max_value=0.5, value=0.2, step=0.05)

        if st.button("Train & evaluate"):
            if not features:
                st.error("Please select at least one feature.")
                return

            X = df[features].values
            y = df[target].values
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            train_pct = (1 - test_size) * 100
            test_pct = test_size * 100
            st.write(f"Train set: {len(X_train)} samples ({train_pct:.2f}%)")
            st.write(f"Test set: {len(X_test)} samples ({test_pct:.2f}%)")

            model = GaussianNaiveBayes()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            cm = confusion_matrix(y_test, y_pred)
            st.write("### Confusion Matrix")
            st.dataframe(pd.DataFrame(cm, index=np.unique(y), columns=np.unique(y)))
            st.write("### Classification Report")
            st.text(classification_report(y_test, y_pred))


if __name__ == "__main__":
    import sys

    # decide which interface to run
    if "streamlit" in sys.modules:
        run_streamlit()
    else:
        run_cli()