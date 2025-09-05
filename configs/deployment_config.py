# configs/deployment_config.py

from dataclasses import dataclass

@dataclass
class DeploymentConfig:
    """
    Settings for inference and deployment.
    """
    # Inference
    device: str = "cuda"                   # or "cpu"
    quantize: bool = True                  # post-training quantization
    quantization_bits: int = 8             # 8-bit onnx or QAT

    # ONNX
    onnx_export_path: str = "outputs/onnx/"
    use_external_tensor_rt: bool = False

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    max_request_length: int = 32768         # allow up to 32K tokens

    # Logging & monitoring
    log_level: str = "INFO"
    metrics_enabled: bool = True
    metrics_endpoint: str = "/metrics"