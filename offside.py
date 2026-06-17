"""
Módulo do algoritmo de impedimento.
Implementação do algoritmo de detecção de impedimento baseado em ângulo.
"""

import math
import numpy as np


def angular_position(
    vanishing_point: tuple,
    player_point: tuple,
    goal_side: str = "right"
) -> float:
    """
    Calcula o ângulo formado entre o ponto de referência horizontal,
    o ponto de fuga e o ponto do jogador.
    Ângulo maior → jogador mais próximo do gol atacado.

    Args:
        vanishing_point: Tupla (vp_x, vp_y) do ponto de fuga
        player_point: Tupla (x, y) do jogador
        goal_side: 'left' ou 'right'

    Returns:
        Ângulo em graus (pode ser negativo)
    """
    vp = np.array(vanishing_point, dtype=float)
    pp = np.array(player_point, dtype=float)

    # Ponto de referência: mesmo Y do VP, mas na borda oposta ao gol
    ref = np.array([0.0 if goal_side == "right" else 10000.0, vp[1]])

    va = ref - vp
    vb = pp - vp

    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    
    if norm_a < 1e-6 or norm_b < 1e-6:
        return 0.0

    cos_angle = np.clip(np.dot(va, vb) / (norm_a * norm_b), -1.0, 1.0)
    angle = math.degrees(math.acos(cos_angle))

    # Sinal: ângulo negativo se jogador estiver "atrás" do VP em x
    if goal_side == "right" and pp[0] < vp[0]:
        angle = -angle
    elif goal_side == "left" and pp[0] > vp[0]:
        angle = -angle

    return angle


def offside_algorithm(
    players: list,
    vanishing_point: tuple,
    goal_side: str = "right"
) -> tuple:
    """
    Implementação do algoritmo de detecção de impedimento.
    Baseia-se no ângulo do 2º defensor mais avançado.

    Args:
        players: Lista de dicts com 'box', 'team', 'attack'
        vanishing_point: Tupla (vp_x, vp_y) do ponto de fuga
        goal_side: 'left' ou 'right'

    Returns:
        Tupla (players_atualizado, offside_line_angle)
    """
    # Importa aqui para evitar circular import
    from detection import player_key_point

    # Calcula ângulo de cada jogador
    for p in players:
        kp = player_key_point(p['box'])
        p['angle'] = angular_position(vanishing_point, kp, goal_side)

    # Segundo-penúltimo defensor (índice 1 do time defensor ordenado por ângulo desc)
    defenders = [p for p in players if not p['attack'] and p['team'] != -1]
    
    if len(defenders) < 1:
        for p in players:
            p['offside'] = False
        return players, 0.0

    defenders_sorted = sorted(defenders, key=lambda p: p['angle'], reverse=True)
    # Segundo defensor mais avançado
    idx_second_last = min(1, len(defenders_sorted) - 1)
    min_defending_angle = defenders_sorted[idx_second_last]['angle']

    # Marca jogadores impedidos
    for p in players:
        if p['attack'] and p['team'] != -1:
            p['offside'] = p['angle'] > min_defending_angle
        else:
            p['offside'] = False

    return players, min_defending_angle
