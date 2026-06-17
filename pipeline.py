"""
Pipeline principal de detecção de impedimento.
"""

import cv2
import numpy as np

# Importa módulos
from segmentation import get_field_mask
from detection import detect_players, player_key_point
from classification import classify_teams
from vanishing_point import find_vanishing_point
from offside import offside_algorithm
from visualization import (
    draw_offside_line,
    draw_players,
    draw_verdict,
    draw_legend,
    draw_debug_info
)


def run_var(
    image_path: str,
    goal_side: str = "right",
    attacking_team: int = 0,
    output_path: str | None = None,
    show: bool = True,
    debug: bool = False
) -> dict:
    """
    Pipeline completo de detecção de impedimento.

    Args:
        image_path: Caminho para imagem de entrada (JPG/PNG)
        goal_side: Direção do gol atacado ('left' ou 'right')
        attacking_team: Qual cluster KMeans é o time atacante (0 ou 1)
        output_path: Caminho para salvar a imagem anotada (opcional)
        show: Se True, exibe a janela OpenCV
        debug: Se True, exibe informações de debug

    Returns:
        Dict com: {'offside': bool, 'count': int, 'image': np.ndarray}
    """
    print(f"\n{'='*60}")
    print(f"  VAR DE IMPEDIMENTO")
    print(f"{'='*60}")
    print(f"  Imagem   : {image_path}")
    print(f"  Gol lado : {goal_side}")
    print(f"{'='*60}\n")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    vis = image.copy()

    # ── Etapa 1: Máscara do campo ──────────────────────────────
    print("[1/5] Segmentando campo...")
    field_mask = get_field_mask(image)
    field_pct = field_mask.mean() / 255 * 100
    print(f"      Campo detectado: {field_pct:.1f}% da imagem")

    # ── Etapa 2: Detecção de jogadores ────────────────────────
    print("[2/5] Detectando jogadores (HOG)...")
    boxes = detect_players(image, field_mask)
    print(f"      Jogadores detectados: {len(boxes)}")

    if len(boxes) < 2:
        print("\n⚠  Menos de 2 jogadores detectados.")
        print("   Tente uma imagem com jogadores mais visíveis.\n")
        cv2.putText(vis, "POUCOS JOGADORES DETECTADOS", (20, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
        if output_path:
            cv2.imwrite(output_path, vis)
        if show:
            cv2.imshow("VAR - Impedimento", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return {"offside": False, "count": 0, "image": vis}

    # ── Etapa 3: Classificação de times ───────────────────────
    print("[3/5] Classificando times por cor de camisa...")
    team_labels = classify_teams(image, boxes)
    count_a = (team_labels == 0).sum()
    count_b = (team_labels == 1).sum()
    count_other = (team_labels == -1).sum()
    print(f"      Time A: {count_a}  Time B: {count_b}  Outros: {count_other}")

    # ── Etapa 4: Ponto de fuga ────────────────────────────────
    print("[4/5] Detectando ponto de fuga (linhas do campo)...")
    vp, vp_lines = find_vanishing_point(image, field_mask, goal_side)
    print(f"      Ponto de fuga: ({vp[0]:.0f}, {vp[1]:.0f})")

    # ── Etapa 5: Algoritmo de Impedimento ─────────────────────
    print("[5/5] Aplicando algoritmo de impedimento...")
    players = []
    for i, box in enumerate(boxes):
        players.append({
            'box': box,
            'team': int(team_labels[i]),
            'attack': int(team_labels[i]) == attacking_team,
        })

    players, offside_line_angle = offside_algorithm(players, vp, goal_side)

    offside_players = [p for p in players if p.get('offside', False)]
    any_offside = len(offside_players) > 0
    offside_count = len(offside_players)

    print(f"\n  Ângulo da linha de impedimento: {offside_line_angle:.2f}°")
    for p in players:
        status = "⛔ IMPEDIDO" if p.get('offside') else "✓ em jogo"
        if p['team'] == -1:
            team_str = "ARB"
        elif p['team'] == attacking_team:
            team_str = f"A (ATK)"
        else:
            team_str = f"B (DEF)"
        print(f"  [Time {team_str}] ângulo={p.get('angle', 0):.1f}°  {status}")

    # ── Visualização ──────────────────────────────────────────
    if debug:
        draw_debug_info(vis, vp, vp_lines, goal_side)
    
    draw_offside_line(vis, vp, offside_line_angle, goal_side,
                      color=(0, 220, 220), thickness=2)
    draw_players(vis, players, attacking_team)
    draw_verdict(vis, any_offside, offside_count)
    draw_legend(vis)

    print(f"\n{'='*60}")
    print(f"  VEREDICTO: {'⛔ IMPEDIMENTO' if any_offside else '✓ SEM IMPEDIMENTO'}")
    if any_offside:
        print(f"  Jogadores impedidos: {offside_count}")
    print(f"{'='*60}\n")

    if output_path:
        cv2.imwrite(output_path, vis)
        print(f"  Imagem salva em: {output_path}\n")

    if show:
        cv2.namedWindow("VAR - Impedimento", cv2.WINDOW_NORMAL)
        h, w = vis.shape[:2]
        cv2.resizeWindow("VAR - Impedimento", min(w, 1280), min(h, 720))
        cv2.imshow("VAR - Impedimento", vis)
        print("  Pressione qualquer tecla para fechar.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {"offside": any_offside, "count": offside_count, "image": vis}
