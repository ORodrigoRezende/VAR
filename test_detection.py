#!/usr/bin/env python3
"""
Script de diagnóstico para testar diferentes métodos de detecção de jogadores.
"""

import cv2
import numpy as np
from segmentation import get_field_mask

def test_hog_detection(image_path):
    """Testa detecção HOG com diferentes parâmetros."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Imagem não encontrada: {image_path}")
        return
    
    h, w = image.shape[:2]
    print(f"\n📷 Imagem: {image_path} ({w}x{h})")
    print(f"{'='*60}")
    
    # ── Teste 1: HOG Padrão ──
    print("\n[1] HOG PADRÃO (Original)")
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    scale = min(1.0, 1280 / max(h, w))
    img_small = cv2.resize(image, (int(w * scale), int(h * scale)))
    
    boxes_raw, weights = hog.detectMultiScale(
        img_small,
        winStride=(8, 8),
        padding=(4, 4),
        scale=1.05
    )
    
    if len(boxes_raw) > 0:
        boxes_raw = (boxes_raw / scale).astype(int)
        print(f"    ✓ Detectados: {len(boxes_raw)} pessoas")
        print(f"    Confiança média: {weights.mean():.2f}")
    else:
        print(f"    ✗ Nenhuma pessoa detectada")
    
    # ── Teste 2: HOG com hitThreshold menor ──
    print("\n[2] HOG COM hitThreshold REDUZIDO")
    boxes_raw2, weights2 = hog.detectMultiScale(
        img_small,
        winStride=(8, 8),
        padding=(4, 4),
        scale=1.05,
        hitThreshold=0.4  # Reduzido de 1.0 (default)
    )
    
    if len(boxes_raw2) > 0:
        boxes_raw2 = (boxes_raw2 / scale).astype(int)
        print(f"    ✓ Detectados: {len(boxes_raw2)} pessoas")
        print(f"    Confiança média: {weights2.mean():.2f}")
    else:
        print(f"    ✗ Nenhuma pessoa detectada")
    
    # ── Teste 3: Detecção por Contornos ──
    print("\n[3] CONTORNOS + ÁREA")
    field_mask = get_field_mask(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Adaptative threshold para detectar pessoas
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 21, 3)
    
    # Aplica máscara do campo
    thresh_field = cv2.bitwise_and(thresh, field_mask)
    
    # Encontra contornos
    contours, _ = cv2.findContours(thresh_field, cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtra contornos por área (pessoas têm tamanho específico)
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Jogadores têm área típica
        if 2000 < area < 50000:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = h / (w + 1e-6)
            # Jogadores têm altura > largura (aspect ratio > 0.5)
            if 0.3 < aspect_ratio < 3.0:
                valid_contours.append(cnt)
    
    print(f"    ✓ Contornos válidos encontrados: {len(valid_contours)}")
    
    # ── Teste 4: Diferença de Frames (se houver múltiplas imagens) ──
    print("\n[4] DETECÇÃO POR COR (Verde vs Outros)")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Detecta pixels que NÃO são verdes (prováveis jogadores)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    non_green = cv2.bitwise_not(green_mask)
    
    # Fecha a imagem
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    non_green = cv2.morphologyEx(non_green, cv2.MORPH_CLOSE, kernel)
    
    contours_color, _ = cv2.findContours(non_green, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
    
    valid_color = []
    for cnt in contours_color:
        area = cv2.contourArea(cnt)
        if 2000 < area < 50000:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = h / (w + 1e-6)
            if 0.3 < aspect_ratio < 3.0:
                valid_color.append(cnt)
    
    print(f"    ✓ Objetos não-verdes: {len(valid_color)}")
    
    # ── Resumo ──
    print(f"\n{'='*60}")
    print(f"RESUMO:")
    print(f"  HOG (padrão): {len(boxes_raw)} jogadores")
    print(f"  HOG (liberal): {len(boxes_raw2)} jogadores")
    print(f"  Contornos: {len(valid_contours)} jogadores")
    print(f"  Cor: {len(valid_color)} objetos")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Testa com ambas as imagens
    test_hog_detection("0.jpg")
    test_hog_detection("1.png")
