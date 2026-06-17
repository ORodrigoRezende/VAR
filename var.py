#!/usr/bin/env python3
"""
VAR de Impedimento no Futebol
=============================
Sistema de detecção automática de impedimento usando Visão Computacional.

Uso:
  python var.py --image caminho/para/imagem.jpg [--goal left|right]
  python var.py --demo          # gera imagem sintética de teste
  python var.py --demo --debug  # modo demo com informações de debug

Dependências:
  pip install opencv-python scikit-learn scipy numpy
"""

import argparse
import sys
from utils import create_demo_image
from pipeline import run_var


def main():
    """Entry point da aplicação."""
    parser = argparse.ArgumentParser(
        description="VAR de Impedimento — detecção automática via Visão Computacional"
    )
    parser.add_argument("--image", type=str,
                        help="Caminho para a imagem de entrada")
    parser.add_argument("--goal", type=str, default="right",
                        choices=["left", "right"],
                        help="Lado do gol atacado (default: right)")
    parser.add_argument("--attack", type=int, default=0,
                        choices=[0, 1],
                        help="Índice do time atacante no KMeans (0 ou 1, default: 0)")
    parser.add_argument("--output", type=str, default=None,
                        help="Caminho para salvar imagem anotada")
    parser.add_argument("--demo", action="store_true",
                        help="Gera e analisa imagem sintética de demonstração")
    parser.add_argument("--no-show", action="store_true",
                        help="Não exibe a janela OpenCV (apenas salva)")
    parser.add_argument("--debug", action="store_true",
                        help="Modo debug com informações adicionais")
    
    args = parser.parse_args()

    try:
        if args.demo:
            demo_path = "/tmp/var_demo.jpg"
            print("  Gerando imagem de demonstração...\n")
            create_demo_image(demo_path)
            result = run_var(
                image_path=demo_path,
                goal_side=args.goal,
                attacking_team=args.attack,
                output_path=args.output or "/tmp/var_resultado_demo.jpg",
                show=not args.no_show,
                debug=args.debug
            )
        elif args.image:
            result = run_var(
                image_path=args.image,
                goal_side=args.goal,
                attacking_team=args.attack,
                output_path=args.output,
                show=not args.no_show,
                debug=args.debug
            )
        else:
            parser.print_help()
            sys.exit(0)

        # Código de saída: 1 = impedimento, 0 = sem impedimento
        sys.exit(1 if result["offside"] else 0)

    except FileNotFoundError as e:
        print(f"\n❌ Erro: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()