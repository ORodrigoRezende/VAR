"""
Módulo de visualização.
Responsável pela renderização dos resultados na imagem.
"""

import cv2
import numpy as np
import math


COLORS = {
    "team_a":    (255, 100, 30),   # azul-laranja
    "team_b":    (30,  200, 255),  # ciano
    "referee":   (200, 200, 200),  # cinza
    "offside":   (0,   0,   220),  # vermelho
    "onside":    (0,   220, 0),    # verde
    "line":      (0,   220, 220),  # ciano
    "vp":        (0,   255, 0),    # verde-limão
}


def draw_offside_line(
    vis: np.ndarray,
    vanishing_point: tuple,
    offside_angle: float,
    goal_side: str,
    color: tuple = (0, 220, 220),
    thickness: int = 2
) -> None:
    """
    Desenha a linha de impedimento na imagem.

    Args:
        vis: Imagem de visualização (modificada in-place)
        vanishing_point: Tupla (vp_x, vp_y)
        offside_angle: Ângulo da linha de impedimento
        goal_side: 'left' ou 'right'
        color: Tupla BGR da cor
        thickness: Espessura da linha em pixels
    """
    h, w = vis.shape[:2]
    vp = np.array(vanishing_point, dtype=float)
    ref = np.array([0.0 if goal_side == "right" else w * 2.0, vp[1]])
    va = ref - vp

    angle_rad = math.radians(offside_angle)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    
    direction = np.array([
        va[0] * cos_a - va[1] * sin_a,
        va[0] * sin_a + va[1] * cos_a
    ])
    
    if np.linalg.norm(direction) < 1e-6:
        return

    # Dois pontos distantes ao longo dessa direção
    far = 5000
    norm_dir = direction / np.linalg.norm(direction)
    p1 = (int(vp[0] + norm_dir[0] * far),
          int(vp[1] + norm_dir[1] * far))
    p2 = (int(vp[0] - norm_dir[0] * far),
          int(vp[1] - norm_dir[1] * far))

    cv2.line(vis, p1, p2, color, thickness, cv2.LINE_AA)

    # Etiqueta
    label_pt = (max(5, min(w - 150, int((p1[0] + p2[0]) / 2))),
                max(20, min(h - 10, int((p1[1] + p2[1]) / 2))))
    cv2.putText(vis, "Linha de Impedimento", label_pt,
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def draw_players(
    vis: np.ndarray,
    players: list,
    attacking_team: int
) -> None:
    """
    Desenha caixas, etiquetas e indicadores de impedimento.

    Args:
        vis: Imagem de visualização (modificada in-place)
        players: Lista de dicts com 'box', 'team', 'offside', 'angle'
        attacking_team: Índice do time atacante
    """
    from detection import player_foot_point
    
    for p in players:
        x, y, bw, bh = p['box']
        team = p['team']
        is_off = p.get('offside', False)

        # Cor da caixa por time
        if team == -1:
            box_color = COLORS["referee"]
            label = "REF"
        elif team == attacking_team:
            box_color = COLORS["team_a"]
            label = f"ATK {'⛔' if is_off else '✓'}"
        else:
            box_color = COLORS["team_b"]
            label = "DEF"

        border = 3 if is_off else 2
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), box_color, border)

        # Ângulo no canto superior
        ang_txt = f"{p.get('angle', 0):.1f}°"
        cv2.putText(vis, ang_txt, (x + 2, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA)

        # Indicador de impedimento (círculo no pé)
        foot = player_foot_point(p['box'])
        indicator_color = COLORS["offside"] if is_off else COLORS["onside"]
        cv2.circle(vis, foot, 5, indicator_color, -1)

        # Label
        cv2.putText(vis, label, (x, y + bh + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1, cv2.LINE_AA)

        # Se impedido: linha vertical vermelha
        if is_off:
            cv2.line(vis, (foot[0], y), (foot[0], y + bh + 5),
                     COLORS["offside"], 2, cv2.LINE_AA)


def draw_verdict(
    vis: np.ndarray,
    any_offside: bool,
    offside_count: int
) -> None:
    """
    Desenha faixa de veredicto no topo da imagem.

    Args:
        vis: Imagem de visualização (modificada in-place)
        any_offside: Se há impedimento
        offside_count: Número de jogadores impedidos
    """
    h, w = vis.shape[:2]
    overlay = vis.copy()
    bar_h = 55
    color_bar = (0, 0, 180) if any_offside else (0, 140, 0)
    cv2.rectangle(overlay, (0, 0), (w, bar_h), color_bar, -1)
    cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)

    if any_offside:
        plural = 'es' if offside_count > 1 else ''
        text = f"⛔  IMPEDIMENTO  ({offside_count} jogador{plural})"
    else:
        text = "✓  SEM IMPEDIMENTO"

    cv2.putText(vis, text, (15, 38),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)


def draw_legend(vis: np.ndarray) -> None:
    """
    Desenha legenda no canto inferior esquerdo.

    Args:
        vis: Imagem de visualização (modificada in-place)
    """
    h, w = vis.shape[:2]
    items = [
        (COLORS["team_a"], "Time Atacante"),
        (COLORS["team_b"], "Time Defensor"),
        (COLORS["referee"], "Árbitro/Goleiro"),
        (COLORS["offside"], "Impedido"),
        (COLORS["onside"], "Em jogo"),
        (COLORS["line"], "Linha de impedimento"),
    ]
    
    x0, y0 = 10, h - 20 - 20 * len(items)
    overlay = vis.copy()
    cv2.rectangle(overlay, (x0 - 5, y0 - 5), (x0 + 230, h - 5), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)

    for i, (c, txt) in enumerate(items):
        y = y0 + i * 20
        cv2.rectangle(vis, (x0, y), (x0 + 15, y + 13), c, -1)
        cv2.putText(vis, txt, (x0 + 20, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (220, 220, 220), 1, cv2.LINE_AA)


def draw_debug_info(
    vis: np.ndarray,
    vp: tuple,
    vp_lines: list,
    goal_side: str,
    max_lines: int = 6
) -> None:
    """
    Desenha informações de debug (VP, linhas detectadas).

    Args:
        vis: Imagem de visualização (modificada in-place)
        vp: Tupla (vp_x, vp_y) do ponto de fuga
        vp_lines: Lista de segmentos de linha detectados
        goal_side: 'left' ou 'right'
        max_lines: Número máximo de linhas para desenhar
    """
    h, w = vis.shape[:2]
    
    # Desenha linhas de campo detectadas
    for seg in vp_lines[:max_lines]:
        cv2.line(vis, tuple(seg[0]), tuple(seg[1]), (60, 60, 200), 1, cv2.LINE_AA)
    
    # Ponto de fuga (pode estar fora da imagem)
    vp_screen = (max(0, min(w - 1, int(vp[0]))),
                 max(0, min(h - 1, int(vp[1]))))
    cv2.circle(vis, vp_screen, 8, COLORS["vp"], -1)
    cv2.putText(vis, "VP", (vp_screen[0] + 10, vp_screen[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["vp"], 1, cv2.LINE_AA)
    
    # Info overlay
    cv2.putText(vis, f"VP:({vp[0]:.0f},{vp[1]:.0f})  Gol:{goal_side}",
                (10, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
