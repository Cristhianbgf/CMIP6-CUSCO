"""Pruebas de la configuración central: el YAML carga y es coherente."""

from cmip6cusco.config import bbox, cargar_config


def test_config_carga():
    config = cargar_config()
    assert config["dominio"]["nombre"] == "Cusco altoandino"
    assert config["semilla"] == 42


def test_bbox_coherente():
    config = cargar_config()
    lon_min, lat_min, lon_max, lat_max = bbox(config)
    assert lon_min < lon_max
    assert lat_min < lat_max
    # El dominio debe contener a la ciudad de Cusco (13.53°S, 71.97°W)
    assert lat_min <= -13.53 <= lat_max
    assert lon_min <= -71.97 <= lon_max


def test_diez_gcms_tres_escenarios():
    config = cargar_config()
    assert len(config["gcms"]) == 10
    assert set(config["escenarios"]) == {"ssp126", "ssp370", "ssp585"}
