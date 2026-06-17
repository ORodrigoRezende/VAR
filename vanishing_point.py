"""
Módulo de detecção do ponto de fuga.
Responsável por detectar o ponto de convergência das linhas do campo.
"""

import cv2
import numpy as np
import math


def _line_intersection(l1: list, l2: list) -> tuple | None:
    """
    Calcula intersecção de dois segmentos de linha.
    
    Args:
        l1: Segmento [[x1, y1], [x2, y2]]
        l2: Segmento [[x3, y3], [x4, y4]]
        
    Returns:
        Tupla (x, y) da intersecção ou None se linhas paralelas
    """
    (x1, y1), (x2, y2) = l1
    (x3, y3), (x4, y4) = l2
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-8:
        return None
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    xi = x1 + t * (x2 - x1)
    yi = y1 + t * (y2 - y1)
    
    return (xi, yi)


def find_vanishing_point(
    image: np.ndarray,
    field_mask: np.ndarray,
    goal_side: str = "right"
) -> tuple:
    """
    Detecta o ponto de fuga usando as linhas laterais reais do campo.
    Calcula a intersecção das laterais para obter perspectiva correta.
    
    Args:
        image: Imagem BGR de entrada
        field_mask: Máscara binária do campo
        goal_side: 'left' ou 'right' — define o lado do gol atacado

    Returns:
        Tupla (vp_x, vp_y, lista_linhas_detectadas)
    """
    h, w = image.shape[:2]

    # Encontra o contorno do campo para extrair laterais
    contours, _ = cv2.findContours(field_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Fallback: VP estimado
        if goal_side == "right":
            vp = (-w * 0.8, h * 0.5)
        else:
            vp = (w * 1.8, h * 0.5)
        return vp, []
    
    # Usa o contorno maior (o campo)
    field_contour = max(contours, key=cv2.contourArea)
    
    # Aproxima para detectar os vértices principais
    epsilon = 0.01 * cv2.arcLength(field_contour, True)
    approx = cv2.approxPolyDP(field_contour, epsilon, True)
    
    pts = approx.reshape(-1, 2).astype(float)
    
    if len(pts) < 4:
        if goal_side == "right":
            vp = (-w * 0.8, h * 0.5)
        else:
            vp = (w * 1.8, h * 0.5)
        return vp, []
    
    # Ordena pontos por Y (topo vs base)
    sorted_by_y = pts[np.argsort(pts[:, 1])]
    
    # Pega top e bottom points, ordena por X para lado esquerdo/direito
    top_2 = sorted_by_y[:2]
    bottom_2 = sorted_by_y[-2:]
    
    # Lado esquerdo: mínimo X
    left_idx_top = np.argmin(top_2[:, 0])
    left_idx_bot = np.argmin(bottom_2[:, 0])
    
    left_top = top_2[left_idx_top]
    left_bottom = bottom_2[left_idx_bot]
    
    # Lado direito: máximo X
    right_idx_top = np.argmax(top_2[:, 0])
    right_idx_bot = np.argmax(bottom_2[:, 0])
    
    right_top = top_2[right_idx_top]
    right_bottom = bottom_2[right_idx_bot]
    
    # Constrói as duas linhas das laterais
    left_line = [tuple(left_top), tuple(left_bottom)]
    right_line = [tuple(right_top), tuple(right_bottom)]
    
    # Calcula intersecção (ponto de fuga)
    vp_candidate = _line_intersection(left_line, right_line)
    
    if vp_candidate is not None:
        vp_x, vp_y = vp_candidate
        
        # Limita VP a uma zona sensata (não muito longe)
        # Mas deixa livre para estar fora da imagem se necessário
        vp_x = np.clip(vp_x, -w * 2, w * 3)
        vp_y = np.clip(vp_y, -h, h * 2)
        
        return (vp_x, vp_y), [left_line, right_line]
    
    # Fallback
    if goal_side == "right":
        vp = (-w * 0.8, h * 0.5)
    else:
        vp = (w * 1.8, h * 0.5)
    
    return vp, []
