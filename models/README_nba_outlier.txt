NBA Outlier-Style App – Local Version
=====================================

Files included
--------------
- nba_outlier_app.py     -> Main Streamlit app.
- train_over_model.py    -> Model training script (v3.0 target=label_over).

How to use
----------
1. Put these files in your `C:\nba_outlier_app` folder (or any folder you want).
   Make sure your data file `prop_training_data.csv` is in the SAME folder.

2. Training the model
   - Open PowerShell in the folder (SHIFT + right-click -> "Open PowerShell window here")
   - Run:
       python train_over_model.py
   - This will:
       * Load `prop_training_data.csv`
       * Use the `label_over` column as the training target
       * Train the GLOBAL model + per-market models
       * Save them into `models/over_model.pkl` and `models/over_model_<market>.pkl`

3. Running the app
   - In the same folder, run:
       streamlit run nba_outlier_app.py
   - Open the URL Streamlit shows in your browser (usually http://localhost:8501)

Notes
-----
- If you change the structure of `prop_training_data.csv`, re-run the training script.
- The app will automatically pick up the new `models/over_model.pkl` file.
- If you ever get errors about missing models, re-run `train_over_model.py`.
