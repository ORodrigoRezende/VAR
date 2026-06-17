"""
Módulo de detecção de jogadores.
Responsável pela detecção de pessoas no campo usando HOG.
"""

import cv2
import numpy as np


def detect_players(image: np.ndarray, field_mask: np.ndarray) -> list:
    """
    Detecta jogadores usando múltiplos métodos (contornos + cor).
    Muito mais eficaz que HOG para imagens de futebol.
    
    Args:
        image: Imagem BGR de entrada
        field_mask: Máscara binária do campo
        
    Returns:
        Lista de bounding boxes (x, y, w, h)
    """
    # Tenta primeiro com detecção avançada (melhor para futebol)
    boxes = _detect_by_contours(image, field_mask)
    
    if len(boxes) >= 2:
        return boxes
    
    # Fallback: tenta detecção por cor
    boxes_color = _detect_by_color(image, field_mask)
    
    return boxes if len(boxes) > 0 else boxes_color


def _detect_by_contours(image: np.ndarray, field_mask: np.ndarray) -> list:
    """
    Detecta jogadores usando Canny edges + morphological dilation + clustering.
    Muito mais eficaz que adaptive threshold para detecção de pessoas.
    
    Args:
        image: Imagem BGR
        field_mask: Máscara do campo
        
    Returns:
        Lista de bounding boxes
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Usa Canny edge detection com thresholds baixos para capturar mais bordas
    edges = cv2.Canny(gray, 15, 50)
    edges_field = cv2.bitwise_and(edges, field_mask)
    
    # Dilate para conectar fragmentos de bordas
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    edges_dilated = cv2.dilate(edges_field, kernel_dilate, iterations=2)
    
    # Encontra contornos
    contours, _ = cv2.findContours(
        edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    boxes = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, bw, bh = cv2.boundingRect(cnt)
        
        # Filtra por área (relaxado para capturar mais variações)
        if area < 500 or area > 100000:
            continue
        
        # Filtra por aspect ratio (altura/largura)
        aspect_ratio = bh / (bw + 1e-6)
        if aspect_ratio < 0.15 or aspect_ratio > 5.0:
            continue
        
        # Verifica se está principalmente no campo
        roi_mask = field_mask[y:y+bh, x:x+bw]
        if roi_mask.size > 0 and roi_mask.mean() > 15:
            boxes.append((x, y, bw, bh))
    
    # Aplica NMS
    if boxes:
        boxes = _nms(np.array(boxes), overlap_thresh=0.3)
    
    # Agrupamento por proximidade: mescla boxes muito próximas
    if boxes:
        boxes = _cluster_nearby_boxes(np.array(boxes), min_distance=50)
    
    return boxes


def _cluster_nearby_boxes(boxes: np.ndarray, min_distance: int = 50) -> list:
    """
    Agrupa boxes que estão muito próximas (provavelmente mesma pessoa fragmentada).
    
    Args:
        boxes: Array de bounding boxes (x, y, w, h)
        min_distance: Distância mínima entre centros para manter separado
        
    Returns:
        Lista de boxes agrupadas
    """
    if len(boxes) == 0:
        return []
    
    # Calcula centros
    centers = boxes[:, :2] + boxes[:, 2:] / 2
    
    # Agrupa boxes próximas
    used = set()
    clustered = []
    
    for i in range(len(boxes)):
        if i in used:
            continue
        
        cluster = [boxes[i]]
        used.add(i)
        
        # Encontra outras boxes próximas
        for j in range(i + 1, len(boxes)):
            if j in used:
                continue
            
            # Distância entre centros
            dist = np.linalg.norm(centers[i] - centers[j])
            
            if dist < min_distance:
                cluster.append(boxes[j])
                used.add(j)
        
        # Mescla boxes do cluster
        if len(cluster) > 0:
            x_min = min(box[0] for box in cluster)
            y_min = min(box[1] for box in cluster)
            x_max = max(box[0] + box[2] for box in cluster)
            y_max = max(box[1] + box[3] for box in cluster)
            
            merged = (x_min, y_min, x_max - x_min, y_max - y_min)
            clustered.append(merged)
    
    return clustered


def _detect_by_color(image: np.ndarray, field_mask: np.ndarray) -> list:
    """
    Detecta jogadores por detecção de objetos não-verdes (complementar).
    
    Args:
        image: Imagem BGR
        field_mask: Máscara do campo
        
    Returns:
        Lista de bounding boxes
    """
    h, w = image.shape[:2]
    
    # Detecta pixels que NÃO são verdes (prováveis jogadores/árbitro)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    non_green = cv2.bitwise_not(green_mask)
    
    # Aplica máscara do campo
    non_green = cv2.bitwise_and(non_green, field_mask)
    
    # Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    non_green = cv2.morphologyEx(non_green, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Encontra contornos
    contours, _ = cv2.findContours(
        non_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    boxes = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, bw, bh = cv2.boundingRect(cnt)
        
        # Filtro de área
        if area < 1500 or area > 80000:
            continue
        
        # Filtro de aspect ratio
        aspect_ratio = bh / (bw + 1e-6)
        if aspect_ratio < 0.3 or aspect_ratio > 4.0:
            continue
        
        # Verifica proporção de pixeis não-verdes na ROI
        roi_non_green = non_green[y:y+bh, x:x+bw]
        if roi_non_green.size == 0:
            continue
        
        non_green_ratio = roi_non_green.mean() / 255
        if non_green_ratio > 0.3:
            boxes.append((x, y, bw, bh))
    
    # NMS
    if boxes:
        boxes = _nms(np.array(boxes), overlap_thresh=0.35)
    
    return boxes


def _nms(boxes: np.ndarray, overlap_thresh: float = 0.4) -> list:
    """
    Non-Maximum Suppression - remove caixas sobrepostas.
    
    Args:
        boxes: Array de bounding boxes (x, y, w, h)
        overlap_thresh: Limiar de overlap para remover caixas
        
    Returns:
        Lista de tuplas das caixas mantidas
    """
    if len(boxes) == 0:
        return []
    
    boxes = boxes.astype(float)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = areas.argsort()[::-1]
    keep = []
    
    while len(order):
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / (union + 1e-8)
        order = order[1:][iou <= overlap_thresh]
    
    return [tuple(boxes[i].astype(int)) for i in keep]


def player_foot_point(box: tuple) -> tuple:
    """
    Calcula o ponto do pé (centro-inferior) de uma bounding box.
    
    Args:
        box: Tupla (x, y, w, h)
        
    Returns:
        Tupla (x, y) do ponto do pé
    """
    x, y, w, h = box
    return (x + w // 2, y + h)


def player_key_point(box: tuple) -> tuple:
    """
    Calcula ponto representativo para cálculo angular (faixa do quadril, ~75% da altura).
    
    Args:
        box: Tupla (x, y, w, h)
        
    Returns:
        Tupla (x, y) do ponto chave
    """
    x, y, w, h = box
    return (x + w // 2, y + int(h * 0.75))
