"""
Script de teste modular do pipeline de detecção de impedimento.
Execute com:  python test_pipeline.py

Cada etapa pode ser testada individualmente. Se uma etapa falhar,
as seguintes são puladas e o erro é mostrado com clareza.
"""

import cv2
import numpy as np
import sys
import os

IMAGE_PATH = r'Imagens Var\Imagens Var\jogo6_offside.PNG'

# ─────────────────────────────────────────────
# ETAPA 1 — Carregar imagem
# ─────────────────────────────────────────────
def test_carrega_imagem():
    print("\n[ETAPA 1] Carregando imagem...")
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"  ERRO: imagem não encontrada em '{IMAGE_PATH}'")
        return None
    print(f"  OK — shape: {img.shape}")
    return img


# ─────────────────────────────────────────────
# ETAPA 2 — Segmentação do campo (PlayerSegmentationUtils)
# ─────────────────────────────────────────────
def test_segmentacao(img):
    print("\n[ETAPA 2] Segmentação do campo...")
    try:
        from PlayerSegmentationUtils import get_contours, get_boxes, non_max
        contours, mask_aud = get_contours(img.copy())
        boxes, scores = get_boxes(img.copy(), contours)
        if len(boxes) == 0:
            print("  AVISO: nenhum bounding box detectado.")
            return img, []
        final_boxes = non_max(boxes, np.array(scores), 0.1)
        print(f"  OK — {len(final_boxes)} bounding boxes após NMS")

        # Visualização
        vis = img.copy()
        for box in final_boxes:
            cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.imwrite('resultado_segmentacao.jpg', vis)
        print("  Salvo: resultado_segmentacao.jpg")
        return img, final_boxes
    except Exception as e:
        print(f"  ERRO: {e}")
        return img, []


# ─────────────────────────────────────────────
# ETAPA 3 — Ponto de fuga (VanishingPointUtils)
# ─────────────────────────────────────────────
def test_ponto_de_fuga(img):
    print("\n[ETAPA 3] Calculando ponto de fuga...")
    try:
        from VanishingPointUtils import get_vertical_vanishing_point, get_horizontal_vanishing_point
        vvp = get_vertical_vanishing_point(img.copy(), 'left')
        hvp = get_horizontal_vanishing_point(img.copy())
        print(f"  OK — VVP: ({vvp[0]:.1f}, {vvp[1]:.1f})")
        print(f"  OK — HVP: ({hvp[0]:.1f}, {hvp[1]:.1f})")

        vis = img.copy()
        cv2.circle(vis, (int(vvp[0]), int(vvp[1])), 10, (0, 0, 255), -1)
        cv2.circle(vis, (int(hvp[0]), int(hvp[1])), 10, (255, 0, 0), -1)
        cv2.imwrite('resultado_vanishing.jpg', vis)
        print("  Salvo: resultado_vanishing.jpg")
        return vvp, hvp
    except Exception as e:
        print(f"  ERRO: {e}")
        return None, None


# ─────────────────────────────────────────────
# ETAPA 4 — Estimativa de pose (requer modelo carregado)
# ─────────────────────────────────────────────
def test_pose(img, boxes):
    print("\n[ETAPA 4] Estimativa de pose...")
    if len(boxes) == 0:
        print("  PULADO: sem bounding boxes da etapa anterior.")
        return []

    # Tenta importar o modelo de pose real do repositório
    try:
        from PoseEstimationUtils import get_pose_estimations
        pose_estimations = get_pose_estimations(img.copy())
        print(f"  OK — {len(pose_estimations)} poses detectadas")
        return pose_estimations
    except NotImplementedError:
        print("  AVISO: get_info() não implementada ainda. Gerando poses simuladas para continuar o teste...")
    except Exception as e:
        print(f"  AVISO: erro na pose real ({e}). Gerando poses simuladas...")

    # ── Poses simuladas para não travar o restante do pipeline ──
    # Cada pose tem o formato esperado pelo restante do código:
    # [id, team_label_placeholder, {keypoints}, leftmost_point]
    simulated = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        cx = (x1 + x2) // 2
        h  = y2 - y1
        # keypoints aproximados baseados na proporção do bounding box
        kp = {
            'leftShoulder':  {'x': cx - 15, 'y': y1 + int(h * 0.25)},
            'rightShoulder': {'x': cx + 15, 'y': y1 + int(h * 0.25)},
            'leftHip':       {'x': cx - 10, 'y': y1 + int(h * 0.50)},
            'rightHip':      {'x': cx + 10, 'y': y1 + int(h * 0.50)},
            'leftKnee':      {'x': cx - 10, 'y': y1 + int(h * 0.70)},
            'rightKnee':     {'x': cx + 10, 'y': y1 + int(h * 0.70)},
            'leftAnkle':     {'x': cx - 8,  'y': y2 - 5},
            'rightAnkle':    {'x': cx + 8,  'y': y2 - 5},
        }
        leftmost = [x1, y1 + int(h * 0.5)]
        simulated.append([i, None, kp, leftmost])

    print(f"  {len(simulated)} poses simuladas criadas.")
    return simulated


# ─────────────────────────────────────────────
# ETAPA 5 — Classificação de times (KMeans + DBSCAN)
# ─────────────────────────────────────────────
def test_classificacao(img, pose_estimations):
    print("\n[ETAPA 5] Classificação de times (KMeans + DBSCAN)...")
    if not pose_estimations:
        print("  PULADO: sem poses.")
        return pose_estimations

    try:
        from TeamClassificationUtils import get_team_classifications
        pose_estimations = get_team_classifications(pose_estimations, img)

        contagem = {}
        for pose in pose_estimations:
            label = pose[-1]
            contagem[label] = contagem.get(label, 0) + 1

        print("  OK — distribuição de labels:")
        for label, n in sorted(contagem.items()):
            nome = {0: 'Time 0', 1: 'Time 1', 2: 'Goleiro/Árbitro'}.get(label, f'Label {label}')
            print(f"    {nome}: {n} jogador(es)")

        # Visualização
        cores = {0: (0, 100, 255), 1: (255, 100, 0), 2: (0, 255, 0), -1: (128, 128, 128)}
        vis = img.copy()
        for pose in pose_estimations:
            kp = pose[2]
            label = pose[-1]
            cor = cores.get(label, (200, 200, 200))
            if 'leftAnkle' in kp:
                pt = (int(kp['leftAnkle']['x']), int(kp['leftAnkle']['y']))
                cv2.circle(vis, pt, 8, cor, -1)
                cv2.putText(vis, str(label), (pt[0]+5, pt[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)
        cv2.imwrite('resultado_classificacao.jpg', vis)
        print("  Salvo: resultado_classificacao.jpg")

    except Exception as e:
        print(f"  ERRO: {e}")

    return pose_estimations


# ─────────────────────────────────────────────
# ETAPA 6 — Algoritmo de impedimento
# ─────────────────────────────────────────────
def test_impedimento(img, pose_estimations, vvp):
    print("\n[ETAPA 6] Algoritmo de impedimento...")
    if not pose_estimations or vvp is None:
        print("  PULADO: dependências anteriores falharam.")
        return

    try:
        # Atualiza ângulos no ponto de fuga
        from VanishingPointUtils import get_angle
        for pose in pose_estimations:
            lm = pose[-1]  # leftmost point [x, y]
            angle = get_angle(vvp, (lm[0], lm[1]), img, 'left')
            pose.append(angle)

        from CoreOffsideUtils import get_offside_decision
        # Assume time 0 como atacante e time 1 como defensor para o teste
        result, last_def = get_offside_decision(
            pose_estimations,
            vvp,
            attackingTeamId=0,
            defendingTeamId=1,
            isKeeperFound=False
        )

        impedidos = [p for p in result if p[-1] == 'off']
        nao_impedidos = [p for p in result if p[-1] == 'on']
        print(f"  OK — Impedidos: {len(impedidos)} | Não impedidos: {len(nao_impedidos)}")
        print(f"  Último defensor: jogador ID {last_def}")

        # Visualização final
        vis = img.copy()
        for pose in result:
            kp = pose[2]
            decisao = pose[-1]
            if 'leftAnkle' in kp:
                pt = (int(kp['leftAnkle']['x']), int(kp['leftAnkle']['y']))
                if decisao == 'off':
                    cor = (0, 0, 255)    # vermelho = impedido
                    texto = 'OFF'
                elif decisao == 'on':
                    cor = (0, 255, 0)    # verde = não impedido
                    texto = 'ON'
                else:
                    cor = (200, 200, 200)
                    texto = decisao
                cv2.circle(vis, pt, 10, cor, -1)
                cv2.putText(vis, texto, (pt[0]+5, pt[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)

        cv2.imwrite('resultado_impedimento.jpg', vis)
        print("  Salvo: resultado_impedimento.jpg")

    except Exception as e:
        print(f"  ERRO: {e}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("  TESTE DO PIPELINE — DETECÇÃO DE IMPEDIMENTO")
    print("=" * 55)

    img = test_carrega_imagem()
    if img is None:
        sys.exit(1)

    img, boxes        = test_segmentacao(img)
    vvp, hvp          = test_ponto_de_fuga(img)
    poses             = test_pose(img, boxes)
    poses             = test_classificacao(img, poses)
    test_impedimento(img, poses, vvp)

    print("\n" + "=" * 55)
    print("  TESTE CONCLUÍDO")
    print("  Imagens salvas: resultado_segmentacao.jpg,")
    print("                  resultado_vanishing.jpg,")
    print("                  resultado_classificacao.jpg,")
    print("                  resultado_impedimento.jpg")
    print("=" * 55)
