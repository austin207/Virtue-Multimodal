# configs/distillation_config.py

from dataclasses import dataclass

@dataclass
class DistillationConfig:
    """
    Knowledge distillation settings for Virtue.
    """
    # Teacher model
    teacher_model_name: str = "google/gemma-3-4b-it"
    teacher_quantization: str = "4bit"      # Q4_0
    freeze_teacher: bool = True

    # Distillation losses
    temperature: float = 4.0
    alpha_kd: float = 0.8                   # weight for soft-target KD
    alpha_ce: float = 0.2                   # weight for hard CE loss

    # Vision distillation
    vision_distillation: bool = True
    alpha_vision: float = 0.2                # vision feature distillation weight

    # Data mixing
    text_to_vision_ratio: float = 0.3        # percent text-only examples
    multimodal_ratio: float = 0.7            # percent vision+text examples
