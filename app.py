import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model_file = pickle.load(open('breast_cancer_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

    features_name = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                     'smoothness_mean', 'compactness_mean', 'concavity_mean',
                     'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                     'radius_se', 'texture_se', 'perimeter_se', 'area_se',
                     'smoothness_se', 'compactness_se', 'concavity_se',
                     'concave points_se', 'symmetry_se', 'fractal_dimension_se',
                     'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
                     'smoothness_worst', 'compactness_worst', 'concavity_worst',
                     'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']

    df = pd.DataFrame(features_value, columns=features_name)
    output = model_file.predict(df)

    if output == 0:
        res_val = "** breast cancer **"
    else:
        res_val = "no breast cancer"

    return render_template('index.html', prediction_text='Patient has {}'.format(res_val))


if __name__ == "__main__":
    app.run()