"""
Módulo de classificação de times.
Responsável pela classificação de jogadores em times por cor de camisa.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans, DBSCAN


def extract_jersey_features(image: np.ndarray, box: tuple) -> np.ndarray:
    """
    Extrai histograma RGB da região do torso do jogador (ignorando fundo).
    
    Args:
        image: Imagem BGR de entrada
        box: Tupla (x, y, w, h) da bounding box
        
    Returns:
        Array de 96 features (histogramas RGB 32 bins cada)
    """
    x, y, w, h = box
    # Região do torso: 20-65% da altura da caixa
    y1 = y + int(h * 0.20)
    y2 = y + int(h * 0.65)
    x1 = x + int(w * 0.15)
    x2 = x + int(w * 0.85)
    
    roi = image[max(0, y1):y2, max(0, x1):x2]
    
    if roi.size == 0:
        return np.zeros(96)
    
    feats = []
    for ch in range(3):
        hist = cv2.calcHist([roi], [ch], None, [32], [0, 256])
        feats.append(hist.flatten())
    
    return np.concatenate(feats)


def classify_teams(image: np.ndarray, boxes: list) -> np.ndarray:
    """
    Classifica jogadores em Time A e Time B via KMeans (k=2) nos histogramas
    de camisa. Árbitros/goleiros são identificados via DBSCAN como ruído.

    Args:
        image: Imagem BGR de entrada
        boxes: Lista de bounding boxes (x, y, w, h)

    Returns:
        Array de labels onde 0=Time A, 1=Time B, -1=árbitro/goleiro/outlier
    """
    if len(boxes) < 3:
        return np.array([0] * len(boxes))

    feats = np.array(
        [extract_jersey_features(image, b) for b in boxes],
        dtype=np.float32
    )
    
    # Normaliza features
    feats_norm = feats / (feats.max(axis=1, keepdims=True) + 1e-6)

    # KMeans para os 2 times
    km = KMeans(n_clusters=2, n_init=10, random_state=42)
    km_labels = km.fit_predict(feats_norm)

    # DBSCAN para identificar outliers (árbitros, goleiros com cor diferente)
    # Mais liberal: eps=0.7 (era 0.5), min_samples=1 (era 2)
    db = DBSCAN(eps=0.7, min_samples=1)
    db_labels = db.fit_predict(feats_norm)

    # Se DBSCAN marcou como -1, é outlier → árbitro ou goleiro
    final = km_labels.copy()
    final[db_labels == -1] = -1

    return final
