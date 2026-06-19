import cv2
import numpy as np

def segment_pitch(image_path):
    """
    Função para ler uma imagem de futebol, isolar o gramado verde
    e retornar a máscara do campo e o maior contorno (polígono).
    """
    # 1. Carregar a imagem
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erro: Não foi possível carregar a imagem em {image_path}")
        return None, None
    
    # Redimensionar para facilitar o processamento e visualização na tela
    img = cv2.resize(img, (800, 600))
    
    # 2. Converter de BGR (padrão OpenCV) para HSV
    # O espaço HSV separa a cor (Hue) da iluminação (Value), ótimo para lidar com sombras no campo.
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 3. Definir o intervalo da cor verde no espaço HSV
    # Esses valores podem precisar de ajustes finos dependendo da iluminação da foto
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    # 4. Criar a máscara inicial (Tudo que for verde vira branco (255), o resto vira preto (0))
    mask = cv2.inRange(hsv_img, lower_green, upper_green)
    
    # 5. Morfologia Matemática (Limpando a imagem)
    kernel = np.ones((5, 5), np.uint8)
    
    # Fechamento (Dilatação seguida de Erosão): Preenche os "buracos" pretos dentro do campo (ex: as linhas brancas)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Abertura (Erosão seguida de Dilatação): Remove "ruídos" brancos fora do campo (ex: camisa verde na torcida)
    mask_cleaned = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 6. Encontrar os contornos na máscara limpa
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Nenhum campo detectado.")
        return img, mask_cleaned
        
    # Assumimos que o maior contorno verde da imagem é o gramado
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Desenhando o polígono do campo na imagem original para visualização (em vermelho)
    cv2.drawContours(img, [largest_contour], -1, (0, 0, 255), 3)
    
    return img, mask_cleaned

if __name__ == "__main__":
    # Testando o script
    # Substitua 'teste_campo.jpg' pelo caminho de uma imagem do dataset baixado
    caminho_imagem = '1.png' 
    
    imagem_resultado, mascara_final = segment_pitch(caminho_imagem)
    
    if imagem_resultado is not None:
        # Mostrando os resultados nas janelas do sistema
        cv2.imshow("Imagem Original com Poligono", imagem_resultado)
        cv2.imshow("Mascara do Gramado (PDI)", mascara_final)
        
        print("Pressione 'q' ou qualquer tecla na janela da imagem para fechar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        #conda activate offside-detection