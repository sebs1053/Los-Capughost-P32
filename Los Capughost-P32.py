import collections
import io
import json
import os
import pathlib
import re
import tarfile
import zipfile
import textwrap
import warnings
import html
import time
import sys
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import TruncatedSVD

# -----------------------------------------------------
# 1. Configuración y Funciones Auxiliares
# -----------------------------------------------------
pd.set_option('display.max_colwidth', 95)
plt.rcParams.update({'figure.figsize': (10, 4.2), 'font.size': 11})

def titulo(texto, sub=''):
    display(HTML(f"""<div style='background:#eef7fc;border-left:7px solid #1183a5;border-radius:12px;padding:14px 18px;margin:10px 0 12px'>
      <div style='color:#075273;font-size:19px;font-weight:800'>{html.escape(texto)}</div><div style='color:#476574;margin-top:5px'>{html.escape(sub)}</div>
    </div>"""))

def mensaje(texto, color='#17587a'):
    display(HTML(f"<div style='border:1px solid #d5e8ed;border-radius:10px;padding:12px;color:{color};background:#f8fcfd'>{html.escape(str(texto))}</div>"))

# -----------------------------------------------------
# 2. Lectura y Procesamiento de Archivos (Misión 1 & 2)
# -----------------------------------------------------
# Recuperamos las variables cargadas si existen
try:
    from google.colab import files
    # Si no están en memoria local, los solicitamos
    if 'FUENTES' not in globals():
        print("Por favor, selecciona los 3 archivos .tar.gz si se solicita:")
        cargados = files.upload()
        entradas = {}
        for nombre, datos in cargados.items():
            if nombre.lower().endswith('.zip'):
                with zipfile.ZipFile(io.BytesIO(datos)) as z:
                    for interno in z.namelist():
                        if interno.lower().endswith(('.tar.gz', '.tgz')) and not interno.endswith('/'):
                            entradas[pathlib.PurePosixPath(interno).name] = z.read(interno)
            else:
                entradas[pathlib.PurePosixPath(nombre).name] = datos

        def tipo_paquete(nombre):
            n = nombre.lower()
            if 'ejempl' in n: return 'ejemplos'
            if 'criticidad' in n: return 'criticidad'
            if 'rules' in n and 'xml' in n: return 'xml'
            return None

        def leer_tar_en_memoria(contenido):
            archivos = {}
            with tarfile.open(fileobj=io.BytesIO(contenido), mode='r:*') as paquete:
                for miembro in paquete:
                    if miembro.isfile():
                        data = paquete.extractfile(miembro).read()
                        archivos[miembro.name] = data.decode('utf-8', errors='replace')
            return archivos

        FUENTES = {}
        for nombre, datos in entradas.items():
            tipo = tipo_paquete(nombre)
            if tipo:
                FUENTES[tipo] = leer_tar_en_memoria(datos)
except Exception as e:
    print("Nota:", e)

# Estructuración de alertas en DataFrame
ALERTAS = []
for ruta, contenido in sorted(FUENTES['ejemplos'].items()):
    for numero_linea, linea in enumerate(contenido.splitlines(), start=1):
        if not linea.strip():
            continue
        evento = json.loads(linea)
        evento['_origen_archivo'] = ruta
        evento['_linea'] = numero_linea
        ALERTAS.append(evento)

filas = []
for i, evento in enumerate(ALERTAS):
    regla = evento.get('rule') or {}
    filas.append({
        'indice': i,
        'fecha': evento.get('timestamp',''),
        'regla_id': str(regla.get('id','')),
        'descripcion': str(regla.get('description','')),
        'nivel': int(regla.get('level',0)),
        'grupos': ', '.join(regla.get('groups') or []),
        'origen': evento['_origen_archivo'].split('/')[-1]
    })
resumen = pd.DataFrame(filas)

# -----------------------------------------------------
# 3. Preparación de X e y (Misión 3)
# -----------------------------------------------------
MAPA_MESAS = {
'19005':'CUMPLIMIENTO','19014':'CUMPLIMIENTO',
'533':'SISTEMAS_Y_RED','60227':'SISTEMAS_Y_RED','60228':'SISTEMAS_Y_RED',
'60602':'SISTEMAS_Y_RED','60608':'SISTEMAS_Y_RED','60716':'SISTEMAS_Y_RED',
'61102':'SISTEMAS_Y_RED','61103':'SISTEMAS_Y_RED','61104':'SISTEMAS_Y_RED','61110':'SISTEMAS_Y_RED',
'60104':'IDENTIDAD_Y_ACCESOS','60154':'IDENTIDAD_Y_ACCESOS','60204':'IDENTIDAD_Y_ACCESOS',
'67022':'IDENTIDAD_Y_ACCESOS','67023':'IDENTIDAD_Y_ACCESOS','92652':'IDENTIDAD_Y_ACCESOS',
'91703':'IDENTIDAD_Y_ACCESOS','91710':'IDENTIDAD_Y_ACCESOS','91713':'IDENTIDAD_Y_ACCESOS',
'91556':'CORREO_Y_NUBE','91702':'CORREO_Y_NUBE','91707':'CORREO_Y_NUBE','91724':'CORREO_Y_NUBE',
}
resumen['mesa_sugerida'] = resumen.regla_id.map(MAPA_MESAS)

def aplanar_campos(obj, prefijo='', tope=90):
    if not isinstance(obj, dict): return []
    resultado = []
    for clave, valor in obj.items():
        nombre = (prefijo + '.' + clave).strip('.')
        if isinstance(valor, dict):
            resultado.extend(aplanar_campos(valor, nombre, tope))
        elif not isinstance(valor, (list, dict)):
            resultado.append(nombre)
        if len(resultado) > tope:
            break
    return resultado[:tope]

def texto_alerta(a):
    regla = a.get('rule') or {}
    data = a.get('data') or {}
    office = data.get('office365') if isinstance(data.get('office365'), dict) else {}
    win = data.get('win') if isinstance(data.get('win'), dict) else {}
    system = win.get('system') if isinstance(win.get('system'), dict) else {}
    valores_tecnicos = [office.get(k, '') for k in ('Operation','Workload','Verdict','DeliveryAction','PolicyAction','DetectionMethod')]
    valores_tecnicos += [system.get(k, '') for k in ('eventID','channel','providerName')]
    partes = [
        str(regla.get('description', '')),
        ' '.join(str(g) for g in (regla.get('groups') or [])),
        ' '.join(str(v) for v in valores_tecnicos if v is not None),
        ' '.join(aplanar_campos(data)[:45]),
        str(a.get('location', ''))
    ]
    return ' '.join(partes).lower()

resumen['texto_modelo'] = [texto_alerta(a) for a in ALERTAS]

# -----------------------------------------------------
# 4. División Train/Test y Modelado (Misión 4)
# -----------------------------------------------------
REGLAS_EXAMEN = {'60227', '67023', '91707', '19014'}
mascara_prueba = resumen.regla_id.isin(REGLAS_EXAMEN)
train_df = resumen.loc[~mascara_prueba].copy()
test_df = resumen.loc[mascara_prueba].copy()

columnas_X = ['texto_modelo', 'nivel']
X_train, y_train = train_df[columnas_X], train_df['mesa_sugerida']
X_test, y_test = test_df[columnas_X], test_df['mesa_sugerida']

def preparar_entradas():
    return ColumnTransformer([
       ('palabras', TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=1400, sublinear_tf=True), 'texto_modelo'),
       ('nivel', StandardScaler(), ['nivel'])
    ], sparse_threshold=0.3)

MODELOS = {
'Árbol de decisión': Pipeline([('preparacion', preparar_entradas()), ('modelo', DecisionTreeClassifier(max_depth=7, min_samples_leaf=2, random_state=42))]),
'Random Forest': Pipeline([('preparacion', preparar_entradas()), ('modelo', RandomForestClassifier(n_estimators=120, max_depth=10, min_samples_leaf=1, random_state=42, n_jobs=-1))]),
'SVM': Pipeline([('preparacion', preparar_entradas()), ('modelo', LinearSVC(class_weight='balanced', random_state=42, max_iter=4000))]),
'ANN (MLP)': Pipeline([('preparacion', preparar_entradas()), ('reduccion', TruncatedSVD(n_components=24, random_state=42)),
                       ('escala', StandardScaler()), ('modelo', MLPClassifier(hidden_layer_sizes=(24,12), max_iter=600, solver='lbfgs', random_state=42))]),
}

# Entrenamiento
for nombre, modelo in MODELOS.items():
    modelo.fit(X_train, y_train)

# -----------------------------------------------------
# 5. Evaluación (Misión 5)
# -----------------------------------------------------
referencia = DummyClassifier(strategy='most_frequent')
referencia.fit(X_train, y_train)
EVALUADOS = {'Sin IA: mesa más frecuente': referencia, **MODELOS}
PREDICCIONES = {}
resultados = []

for nombre, modelo in EVALUADOS.items():
    predicciones = modelo.predict(X_test)
    PREDICCIONES[nombre] = predicciones
    aciertos = int((predicciones == y_test.to_numpy()).sum())
    resultados.append({
        'Modelo': nombre,
        'Aciertos (de 19)': aciertos,
        'Porcentaje correcto': round(100 * aciertos / len(y_test), 1)
    })
metricas = pd.DataFrame(resultados).sort_values('Aciertos (de 19)', ascending=False).reset_index(drop=True)
titulo('🏆 Resultados de la prueba consolidada', 'Porcentaje de aciertos de cada clasificador')
transposed_res = metricas.copy()
display(transposed_res)

# -----------------------------------------------------
# 6. Evaluación de Ejemplo Inédito (Sin reentrenar)
# -----------------------------------------------------
titulo('🧪 Evaluación de Inferencia en Ejemplo Nuevo', 'Pasando una alerta externa inédita a los modelos entrenados')

# Construimos una nueva alerta totalmente externa
ejemplo_inedito = {
    'texto_modelo': 'suspicious user authentication failed multiple times ExchangeOnline email logon failure',
    'nivel': 10
}

df_inedito = pd.DataFrame([ejemplo_inedito])

res_nuevo = []
for nombre, modelo in MODELOS.items():
    pred = modelo.predict(df_inedito[columnas_X])[0]
    res_nuevo.append({
        'Modelo Evaluado': nombre,
        'Mesa Predicha': pred
    })

df_res_nuevo = pd.DataFrame(res_nuevo)
display(df_res_nuevo)

# -----------------------------------------------------
# 7. Preparación e Inferencia del LLM Local (Misión 6)
# -----------------------------------------------------
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    usa_gpu = torch.cuda.is_available()
    MODELO_LLM = 'Qwen/Qwen2.5-1.5B-Instruct' if usa_gpu else 'Qwen/Qwen2.5-0.5B-Instruct'
    print('🖥️ Entorno LLM:', 'GPU' if usa_gpu else 'CPU')

    TOKENIZER_LLM = AutoTokenizer.from_pretrained(MODELO_LLM)
    LLM_LOCAL = AutoModelForCausalLM.from_pretrained(
        MODELO_LLM,
        torch_dtype=torch.float16 if usa_gpu else torch.float32,
        device_map='auto' if usa_gpu else None,
        low_cpu_mem_usage=True,
    )
    LLM_LOCAL.eval()

    # Inferencia de ejemplo sobre el caso #15 de examen
    fila_caso = test_df.iloc[15]
    alerta_caso = ALERTAS[int(fila_caso['indice'])]
    prediccion_caso = str(MODELOS['ANN (MLP)'].predict(pd.DataFrame([fila_caso[columnas_X].to_dict()]))[0])

    def contexto_concreto(evento):
        regla = evento.get('rule') or {}
        d = evento.get('data') or {}
        salida = {
            'timestamp': evento.get('timestamp'),
            'rule': {k: regla.get(k) for k in ('id', 'level', 'description', 'groups')},
            'agent': {'name': (evento.get('agent') or {}).get('name')},
            'location': evento.get('location')
        }
        if 'office365' in d and isinstance(d['office365'], dict):
            office = d['office365']
            salida['office365'] = {k: office[k] for k in ('Operation', 'Workload', 'Verdict', 'DeliveryAction', 'PolicyAction', 'PhishConfidenceLevel', 'Subject') if k in office}
        return salida

    CONTEXTO_CASO = contexto_concreto(alerta_caso)
    BASE_AL = json.dumps(CONTEXTO_CASO, ensure_ascii=False, indent=2)
    INSTRUCCIONES_LLM = (
        'Eres un asistente educativo para estudiantes de preparatoria. '
        'Escribe UN SOLO PÁRRAFO de español muy claro (máximo 110 palabras) '
        'que explique qué información registra Wazuh, el riesgo posible y qué se debería comprobar.'
    )
    PROMPT_A = f'{INSTRUCCIONES_LLM}\n\nALERTA (JSON):\n{BASE_AL}\n\nExplica solo el registro.'
    PROMPT_B = f'{INSTRUCCIONES_LLM}\n\nALERTA (JSON):\n{BASE_AL}\n\nRECOMENDACIÓN: {prediccion_caso}. Explica cómo verificarlo.'

    def generar_con_llm(mensaje_input):
        mensajes = [{'role': 'system', 'content': 'Redacta explicaciones educativas.'}, {'role': 'user', 'content': mensaje_input}]
        codificado = TOKENIZER_LLM.apply_chat_template(mensajes, tokenize=True, add_generation_prompt=True, return_tensors='pt').to(LLM_LOCAL.device)
        with torch.inference_mode():
            pred = LLM_LOCAL.generate(codificado, max_new_tokens=150, do_sample=False, pad_token_id=TOKENIZER_LLM.eos_token_id)
        return TOKENIZER_LLM.decode(pred[0][codificado.shape[1]:], skip_special_tokens=True).strip()

    print("Generando Explicaciones con el LLM local...")
    RESPUESTA_A = generar_con_llm(PROMPT_A)
    RESPUESTA_B = generar_con_llm(PROMPT_B)

    titulo('📝 Experimento A: Solo Wazuh')
    mensaje(RESPUESTA_A)
    titulo('🧠 Experimento B: Wazuh + Predicción')
    mensaje(RESPUESTA_B)

except Exception as e:
    print("No se pudo completar la inferencia de LLM en este paso o faltan dependencias:", e)

print("\n✅ ¡Flujo consolidado correctamente ejecutado!")
