
import onnxruntime_genai as og
import logging

try:
    print("Version:", og.__version__)
    model_path = r"C:\GEMINI_PROJEKTE\_Pb-studio_V_2\pb_studio\data\ai_models\phi-3-mini-4k-directml\directml\directml-int4-awq-block-128"
    print("Loading model from:", model_path)
    model = og.Model(model_path)
    params = og.GeneratorParams(model)
    print("GeneratorParams attributes:")
    print(dir(params))
except Exception as e:
    print("Error:", e)
