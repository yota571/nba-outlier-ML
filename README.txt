NBA Outlier-Style App – Packaged Folder

Files:
- nba_outlier_app.py      -> Your main Streamlit app
- train_over_model.py     -> Training script for the over_model.pkl (uses 'label_over' if present)
- prop_training_data.csv  -> NOT included here; you provide this from your tracking/export
- models/over_model.pkl   -> Created after you run the training script

Basic usage:

1) Make sure your training data is saved as 'prop_training_data.csv'
   in the same folder as these scripts.

2) Open a terminal in this folder and install requirements:

   pip install -r requirements.txt

3) Train the model:

   python train_over_model.py

   When it finishes you should see something like:
   "Saved global model to models/over_model.pkl"

4) Run the app:

   streamlit run nba_outlier_app.py
