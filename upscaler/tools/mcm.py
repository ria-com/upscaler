import os
from modelhub_client import ModelHub


model_config_urls = [
    "https://models.vsp.net.ua/config_model/sr-hat/model-1.json",
    "https://models.vsp.net.ua/config_model/sr-hat/model-2.json",
    "https://models.vsp.net.ua/config_model/sr-hat/model-3.json",
    "https://models.vsp.net.ua/config_model/sr-hat/model-4.json",
    "https://models.vsp.net.ua/config_model/sr-hat/model-5.json",
    "https://models.vsp.net.ua/config_model/sr-drct/model-1.json",
]

local_storage = os.environ.get('LOCAL_STORAGE', os.path.join(os.path.dirname(__file__), "../../data"))
modelhub = ModelHub(model_config_urls=model_config_urls,
                    local_storage=local_storage)
