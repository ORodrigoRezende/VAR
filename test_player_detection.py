#!/usr/bin/env python3
"""
Script de teste para detecção de jogadores.
Testa apenas a etapa de detecção sem classificação ou impedimento.

Uso:
    python test_player_detection.py --image caminho/para/imagem.jpg [--output saida.jpg] [--show]
"""

import argparse
import sys
import cv2
import numpy as np
from pathlib import Path

# Importa módulos do projeto
from segmentation import get_field_mask
from detection import detect_players, player_foot_point


def draw_detection_results(image: np.ndarray, boxes: list, field_mask: np.ndarray) -> np.ndarray:
    """
    Desenha as caixas de detecção na imagem.
    
    Args:
        image: Imagem BGR
        boxes: Lista de bounding boxes (x, y, w, h)
        field_mask: Máscara do campo
        
    Returns:
        Imagem com anotações
    """
    result = image.copy()
    h, w = image.shape[:2]
    
    # Desenha a máscara do campo em transparência
    overlay = result.copy()
    overlay[field_mask == 0] = [50, 50, 50]  # Campo escuro
    cv2.addWeighted(overlay, 0.2, result, 0.8, 0, result)
    
    # Cores para visualização
    box_color = (0, 255, 0)  # Verde
    text_color = (255, 255, 255)  # Branco
    circle_color = (0, 165, 255)  # Laranja
    
    for i, (x, y, bw, bh) in enumerate(boxes):
        # Desenha bounding box
        cv2.rectangle(result, (x, y), (x + bw, y + bh), box_color, 2)
        
        # Desenha área (informação técnica)
        area = bw * bh
        cv2.putText(result, f"A:{area:.0f}", (x + 3, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
        
        # Desenha índice
        cv2.putText(result, f"{i+1}", (x + 3, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
        # Desenha ponto nos pés (referência)
        foot = (x + bw // 2, y + bh)
        cv2.circle(result, foot, 4, circle_color, -1)
        
        # Desenha linha do centro até os pés
        center = (x + bw // 2, y + bh // 2)
        cv2.line(result, center, foot, circle_color, 1)
    
    # Estatísticas no topo
    stats_text = f"Jogadores Detectados: {len(boxes)}"
    cv2.putText(result, stats_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    # Legenda
    legend_y = 60
    cv2.putText(result, "Verde = Bounding box", (10, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 1)
    cv2.putText(result, "Laranja = Posicao dos pes", (10, legend_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, circle_color, 1)
    
    return result


def print_detection_stats(boxes: list, image_shape: tuple) -> None:
    """
    Imprime estatísticas de detecção.
    
    Args:
        boxes: Lista de bounding boxes
        image_shape: Forma da imagem (h, w, c)
    """
    h, w = image_shape[:2]
    
    print("\n" + "=" * 60)
    print("  TESTE DE DETECÇÃO DE JOGADORES")
    print("=" * 60)
    print(f"\nTotal detectado: {len(boxes)} jogadores")
    print(f"Dimensões da imagem: {w}x{h} pixels")
    
    if len(boxes) > 0:
        print("\nDetalhes individuais:")
        print(f"{'#':<3} {'X':<6} {'Y':<6} {'W':<6} {'H':<6} {'Área':<8} {'Aspecto':<8}")
        print("-" * 60)
        
        areas = []
        aspects = []
        
        for i, (x, y, bw, bh) in enumerate(boxes):
            area = bw * bh
            aspect = bh / (bw + 1e-6)
            
            areas.append(area)
            aspects.append(aspect)
            
            print(f"{i+1:<3} {x:<6} {y:<6} {bw:<6} {bh:<6} {area:<8.0f} {aspect:<8.2f}")
        
        # Estatísticas agregadas
        print("\n" + "-" * 60)
        print("Estatísticas Agregadas:")
        print(f"  Área mínima: {min(areas):.0f} px")
        print(f"  Área máxima: {max(areas):.0f} px")
        print(f"  Área média: {np.mean(areas):.0f} px")
        print(f"  Aspecto mín: {min(aspects):.2f}")
        print(f"  Aspecto máx: {max(aspects):.2f}")
        print(f"  Aspecto médio: {np.mean(aspects):.2f}")
    
    print("\n" + "=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Teste de detecção de jogadores"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="Caminho para a imagem de teste"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Caminho para salvar imagem anotada (opcional)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Exibir imagem em janela (pressione qualquer tecla para sair)"
    )
    
    args = parser.parse_args()
    
    # Carrega imagem
    if not Path(args.image).exists():
        print(f"❌ Erro: Arquivo '{args.image}' não encontrado")
        sys.exit(2)
    
    image = cv2.imread(args.image)
    if image is None:
        print(f"❌ Erro: Não foi possível ler a imagem '{args.image}'")
        sys.exit(2)
    
    print(f"✓ Imagem carregada: {args.image} ({image.shape[1]}x{image.shape[0]})")
    
    # Segmenta o campo
    print("→ Segmentando campo...")
    field_mask = get_field_mask(image)
    field_coverage = (field_mask > 0).sum() / field_mask.size * 100
    print(f"✓ Campo detectado: {field_coverage:.1f}% da imagem")
    
    # Detecta jogadores
    print("→ Detectando jogadores...")
    boxes = detect_players(image, field_mask)
    print(f"✓ {len(boxes)} jogadores detectados")
    
    # Imprime estatísticas
    print_detection_stats(boxes, image.shape)
    
    # Desenha resultados
    print("→ Gerando visualização...")
    result_image = draw_detection_results(image, boxes, field_mask)
    
    # Salva resultado
    if args.output:
        cv2.imwrite(args.output, result_image)
        print(f"✓ Imagem salva em: {args.output}")
    
    # Exibe resultado
    if args.show:
        cv2.imshow("Teste de Detecção de Jogadores", result_image)
        print("\n(Pressione qualquer tecla para sair)")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("✓ Teste concluído com sucesso!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
