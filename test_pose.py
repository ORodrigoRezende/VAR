import cv2
import sys
# Adiciona a pasta lib ao caminho do Python para ele encontrar os arquivos .so que compilamos
sys.path.append('./lib')

# Importamos o utilitário do repositório original
import PoseEstimationUtils as pose_utils

def test_neural_network(image_path):
    print("Iniciando a Rede Neural. Carregando os pesos (isso pode demorar uns segundos na primeira vez)...")
    
    # 1. Carrega a imagem
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erro: Não encontrei a imagem em {image_path}")
        return

    # 2. Chama a função de predição do repositório original
    # Essa função faz a ponte com o C++ e com o modelo treinado
    try:
        # A função process_image_for_pose geralmente retorna a imagem com os esqueletos desenhados
        # e a lista de coordenadas brutas (depende de como o autor escreveu o return)
        result_img, coordinates = pose_utils.get_pose_estimations(img)
        
        print("Sucesso! A rede neural processou a imagem.")
        print(f"Jogadores detectados: {len(coordinates)}")
        
        # Mostra o resultado na tela
        cv2.imshow("Estimativa de Pose", result_img)
        print("Pressione qualquer tecla na janela da imagem para fechar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Ocorreu um erro ao rodar a inferência: {e}")

if __name__ == "__main__":
    # Coloque o caminho de uma imagem do seu dataset aqui
    caminho = '0.jpg' 
    test_neural_network(caminho)