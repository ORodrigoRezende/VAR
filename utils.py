"""
Módulo de utilidades e demonstração.
Funções auxiliares para criar imagens de teste.
"""

import cv2
import numpy as np


def create_demo_image(path: str = "/tmp/var_demo.jpg") -> str:
    """
    Gera uma imagem sintética de campo de futebol com jogadores posicionados
    para demonstrar a detecção de impedimento.

    Args:
        path: Caminho para salvar a imagem

    Returns:
        Caminho da imagem criada
    """
    h, w = 720, 1280
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Fundo de grama com bom contraste
    img[:] = (70, 140, 70)

    # Textura da grama (faixas)
    for i in range(0, h, 60):
        cv2.rectangle(img, (0, i), (w, i + 30), (55, 125, 55), -1)

    # Ruído leve para parecer realista
    noise = np.random.randint(-15, 15, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # ─── Linhas do campo em branco vibrante ────────────────────────────────────
    WHITE = (255, 255, 255)

    # Linha de meio-campo
    cv2.line(img, (w // 2, 0), (w // 2, h), WHITE, 4)
    cv2.circle(img, (w // 2, h // 2), 90, WHITE, 3)
    cv2.circle(img, (w // 2, h // 2), 6, WHITE, -1)

    # Linhas laterais
    cv2.line(img, (15, 0), (15, h), WHITE, 3)
    cv2.line(img, (w - 15, 0), (w - 15, h), WHITE, 3)
    cv2.line(img, (15, 0), (w - 15, 0), WHITE, 3)
    cv2.line(img, (15, h - 5), (w - 15, h - 5), WHITE, 3)

    # ─── Desenha jogadores com cores vibrantes ───
    def draw_player(cx, cy, color):
        """Desenha jogador com detalhes."""
        height, width = 95, 38
        # Corpo preenchido com cor vibrante
        cv2.rectangle(img, (cx - width // 2, cy - height), 
                      (cx + width // 2, cy - height // 2), color, -1)
        # Cabeça
        cv2.circle(img, (cx, cy - height - 20), 16, color, -1)
        # Pernas
        cv2.line(img, (cx - 8, cy - height // 2), (cx - 12, cy), color, 9)
        cv2.line(img, (cx + 8, cy - height // 2), (cx + 12, cy), color, 9)
        # Brilho na cabeça
        cv2.circle(img, (cx - 4, cy - height - 23), 3, (255, 255, 255), -1)

    # Cores muito vibrantes e contrastantes
    COLOR_A = (10, 60, 255)       # Vermelho vibrante
    COLOR_B = (0, 255, 100)       # Verde/Ciano vibrante
    COLOR_GK = (0, 165, 255)      # Laranja vibrante
    COLOR_REF = (150, 150, 150)   # Cinza médio

    # ─── Time A (atacante, esquerda) - 5 jogadores ───
    draw_player(250, 250, COLOR_A)
    draw_player(300, 350, COLOR_A)
    draw_player(200, 450, COLOR_A)
    draw_player(350, 500, COLOR_A)
    draw_player(280, 600, COLOR_A)

    # ─── Time B (defensor, direita) - 6 jogadores ───
    draw_player(950, 200, COLOR_B)
    draw_player(1000, 320, COLOR_B)  # 2º defensor (referência)
    draw_player(900, 420, COLOR_B)
    draw_player(1050, 480, COLOR_B)
    draw_player(920, 580, COLOR_B)
    # Goleiro (cor diferente)
    height_gk, width_gk = 105, 50
    cx_gk, cy_gk = 1150, 360
    cv2.rectangle(img, (cx_gk - width_gk // 2, cy_gk - height_gk), 
                  (cx_gk + width_gk // 2, cy_gk - height_gk // 2), COLOR_GK, -1)
    cv2.circle(img, (cx_gk, cy_gk - height_gk - 20), 17, COLOR_GK, -1)
    cv2.line(img, (cx_gk - 9, cy_gk - height_gk // 2), (cx_gk - 13, cy_gk), COLOR_GK, 10)
    cv2.line(img, (cx_gk + 9, cy_gk - height_gk // 2), (cx_gk + 13, cy_gk), COLOR_GK, 10)

    # ─── Árbitro ───
    draw_player(640, 180, COLOR_REF)

    cv2.imwrite(path, img)
    return path
