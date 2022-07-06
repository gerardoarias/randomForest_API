import joblib
import pandas as pd


def load_random_forest_model():

    model = open('../trainedModels/RandomForest_Model.pkl', 'rb')
    return joblib.load(model)
    
def prepare_random_forest_data(departamento, municipio, tipoPerdida, grupoEtario, genero,
                               cabezaFamilia, homicidiosProporcion, ano, indrural, gini,
                                IPM, o_acto_terror):

    cols = [
        'departamento', 'municipio', 'tipoPerdida', 'grupoEtario', 'genero',
        'cabezaFamilia', 'homicidiosProporcion', 'ano', 'indrural', 'gini',
        'IPM', 'o_acto_terror'
    ]

    data = [[
        departamento, municipio, tipoPerdida, grupoEtario, genero,
        cabezaFamilia, homicidiosProporcion, ano, indrural, gini,
        IPM, o_acto_terror
    ]]

    return pd.DataFrame(columns=cols, data=data)
    
def prepare_random_forest_response(prediction):

    decision = {0: 'Compensa', 3: 'No Restituye', 4: 'Restituye'}
    return {"Decisión del Juez": decision[prediction[0]]} 
