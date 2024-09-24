import os
import json

root = os.path.dirname(os.path.dirname(__file__))

project_configuration = dict()

def load_config():
    global project_configuration
    with open("C:/Users/mgflast/PycharmProjects/ontoseg/project_configuration.json", 'r') as f:
        project_configuration = json.load(f)