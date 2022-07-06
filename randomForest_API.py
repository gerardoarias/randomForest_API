import os
import uvicorn
from typing import Literal, List
from pydantic import BaseModel 
from fastapi import FastAPI,UploadFile, Form
from helperFunctions.functions import *
 
proxy=""
service_prefix = os.getenv('JUPYTERHUB_SERVICE_PREFIX')
if service_prefix:
    proxy = f'{service_prefix}proxy/8050/'
    
fastapi_multiple_predictions = FastAPI(root_path=proxy)

loaded_model=load_random_forest_model()


class ModelInput(BaseModel):

    departamento: int = Form(
        description="Departamento en donde se dio la sentencia", ge=0, le=25)
    municipio: int = Form(
        description="Municipio en donde se dio la sentencia", ge=0, le=368)
    tipoPerdida: int = Form(
        description="Tipo de pérdida involucrada", ge=0, le=3)
    grupoEtario: int = Form(
        description="Grupo etario del/de la solicitante", ge=0, le=5)
    genero: int = Form(
        description="Género del/de la solicitante", ge=0, le=3)
    cabezaFamilia: int = Form(
        description="Se es cabeza de familia", ge=0, le=1)
    homicidiosProporcion: float = Form(
        description="Homicidios totales/Población municipal", ge=0, le=1)
    ano: int = Form(
        description="Año de expedición de la sentencia", ge=0, le=10) 
    indrural: float = Form(
        description="Pob Rural/Pob total municipal", ge=0, le=1)
    gini: float = Form(
        description="Gini municipal", ge=0, le=1)   
    IPM: float = Form(
        description="IPM municipal", ge=0, le=100)
    o_acto_terror: int = Form(
        description="Actos de terror ocurridos en el municipio")
        
@fastapi_multiple_predictions.post('/predict/')
async def multiple_predict(inputs:List[ModelInput]):
    
    response = []
    
    for input_ in inputs:
        
        
        prepared_data = prepare_random_forest_data(**dict(input_))

        prediction = loaded_model.predict(prepared_data)
        
        response.append(prepare_random_forest_response(prediction))
    
    return response


if __name__ == "__main__":
    uvicorn.run(fastapi_multiple_predictions, port=8050,host='0.0.0.0')