import os
import pytest
from utils import match_template_from_image, prever_jogo_em_frame
from tensorflow.keras.models import load_model

@pytest.fixture
def test_image_path():
    return "tests/testdata/frame_example.jpg"

@pytest.fixture
def modelo_carregado():
    return load_model("modelo/modelo_pragmatic.keras")

def test_template_matching(test_image_path):
    result = match_template_from_image(test_image_path)
    assert result in [None, "pragmaticplay"]

def test_model_prediction(test_image_path, modelo_carregado):
    result = prever_jogo_em_frame(test_image_path, modelo_carregado)
    assert result in [None, "pragmaticplay"]
