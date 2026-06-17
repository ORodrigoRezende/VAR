"""
Módulo de segmentação do campo de futebol.
Responsável pela detecção da área de jogo (grama).
"""

import cv2
import numpy as np


def get_field_mask(image: np.ndarray) -> np.ndarray:
    """
    Retorna máscara binária da área de jogo (grama verde).
    
    Args:
        image: Imagem BGR de entrada
        
    Returns:
        Máscara binária (0-255) onde 255 = campo, 0 = não-campo
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Verde da grama: matiz 25-90, saturação e valor mínimos para excluir sombras
    lower = np.array([25, 40, 40])
    upper = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Fecha pequenos buracos e remove ruído
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Mantém somente o maior componente conexo
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8) * 255

    return mask
