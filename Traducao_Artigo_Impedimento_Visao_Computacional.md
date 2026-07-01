# Um Dataset e Metodologia para Detecção de Impedimento no Futebol Baseada em Visão Computacional

**Neeraj Panse** — Pune Institute of Computer Technology (neerajpanse9@gmail.com)  
**Ameya Mahabaleshwarkar** — MIT College of Engineering, Pune (ameyasm1154@gmail.com)

> *Ambos os autores contribuíram igualmente para esta pesquisa.*

---

## RESUMO

As decisões de impedimento são parte integrante de toda partida de futebol. Nos últimos tempos, a tomada de decisão nas partidas de futebol, incluindo as decisões de impedimento, tem sido fortemente influenciada pela tecnologia. No entanto, apesar do uso do Árbitro Assistente de Vídeo (VAR), as decisões de impedimento continuam sendo marcadas por inconsistências. Os dois principais pontos de crítica ao VAR têm sido os extensos atrasos na entrega das decisões finais e as decisões imprecisas decorrentes de erros humanos. A natureza visual da tomada de decisão de impedimento torna as técnicas de Visão Computacional uma opção viável para enfrentar esses problemas, automatizando aspectos apropriados do processo.

No entanto, a falta de um algoritmo computacional que capture todos os aspectos da regra de impedimento, a ausência de uma metodologia estabelecida para representar computacionalmente as cenas de partidas de futebol de forma utilizável por tal algoritmo, e a inexistência de um conjunto de dados diverso e abrangente para testar esses métodos têm impedido os esforços de pesquisa nesse problema. Este artigo aborda com precisão cada um desses obstáculos, com o objetivo de facilitar futuras pesquisas nessa área. O artigo apresenta um algoritmo computacional de decisão de impedimento para imagens de partidas de futebol. A metodologia para criar uma representação quantitativa de imagens de partidas de futebol para esse algoritmo também foi apresentada como um pipeline de tarefas de Visão Computacional.

Um conjunto de dados inédito para a avaliação dessa metodologia foi apresentado, contendo uma seleção curada de cenas de partidas de futebol que representam os diversos desafios que podem ser enfrentados por um sistema que visa auxiliar ou automatizar a tarefa de tomar decisões de impedimento. Por fim, o artigo detalha o desempenho de um conjunto específico de tarefas de Visão Computacional utilizadas no pipeline apresentado, no dataset fornecido. O sistema proposto alcança um **F1 score de 0,85** no dataset. As limitações e áreas de melhoria para esses métodos também foram discutidas em uma tentativa de direcionar pesquisas futuras nessa tarefa. O dataset apresentado e o código de implementação do pipeline estão disponíveis em: https://github.com/Neerajj9/Computer-Vision-based-Offside-Detection-in-Soccer

---

## PALAVRAS-CHAVE

Detecção de Impedimento, Visão Computacional, Estimativa de Pose, Classificação de Times, Aprendizado Profundo

---

## 1. INTRODUÇÃO

A arbitragem no futebol é um processo complexo devido a muitos fatores. Na última década, a arbitragem assistida por tecnologia proliferou no futebol. Por exemplo, a Tecnologia de Linha de Gol praticamente eliminou todos os erros nas decisões de gol/não-gol. O Árbitro Assistente de Vídeo (VAR) é uma adição recente ao conjunto de ferramentas de tomada de decisão assistida por tecnologia. O VAR opera com a participação de uma equipe de árbitros externos que revisam todas as decisões durante o jogo, categorizadas em: Gol/Não-Gol, Decisões de Pênalti, Incidentes de Cartão Vermelho Direto e Identidade Equivocada [1]. Esperava-se que o VAR aumentasse a precisão da arbitragem de 82% para 96% [2]. O VAR introduziu um atraso significativo na entrega das decisões. O uso do VAR ainda seria justificado se tivesse demonstrado o aumento esperado na precisão. No entanto, o VAR falhou em múltiplas ocasiões, mesmo para decisões claras e óbvias que não são subjetivas por natureza, como a decisão de impedimento.

A regra de impedimento, conforme definida pela Federação Internacional de Futebol Associado (FIFA), estabelece que um jogador está em posição de impedimento se sua parte do corpo jogável estiver mais próxima da linha de gol adversária do que tanto a bola quanto a penúltima parte do corpo jogável do adversário. A regra também estabelece que, para ser considerada uma infração, o jogador deve não apenas estar em posição de impedimento, mas também estar ativamente envolvido no jogo em andamento. As decisões de impedimento tomadas pelos árbitros em campo apresentam uma taxa de erro de 20% a 26% [3][4][5]. Assim, embora o paradigma de usar tecnologia para auxiliar tais decisões não possa ser questionado, as ineficiências desses métodos certamente precisam ser abordadas. Este trabalho visa apoiar os esforços que têm como alvo essas ineficiências com o auxílio da tecnologia.

Por meio deste trabalho, foi apresentado um algoritmo computacional que representa integralmente a regra de impedimento para avaliar jogadores em imagens de partidas de futebol. O algoritmo computacional requer uma forma quantitativa de representar os diversos aspectos importantes de uma cena de jogo de futebol — as posições relativas dos jogadores, seus times e projeções de partes específicas do corpo. Um pipeline baseado em Visão Computacional foi apresentado, que pode ser usado para converter imagens representando cenas de uma partida de futebol em uma representação computacional que pode ser fornecida ao mencionado algoritmo de impedimento.

Este trabalho também apresenta um dataset de 500 imagens que cobrem vários casos em uma partida de futebol, como um conjunto adequado de dados de validação para testar o pipeline, bem como uma base para julgar esforços futuros sobre esse problema. Até onde sabemos, este é o primeiro dataset que trata dessa tarefa específica. O artigo também estabelece um forte baseline no dataset fornecido, adaptando algoritmos e metodologias de Visão Computacional às diversas tarefas do pipeline e avaliando os resultados usando o algoritmo de impedimento. O dataset e as anotações de verdade fundamental serão disponibilizados como código aberto.

A próxima seção fornece uma visão geral dos trabalhos anteriores. O pipeline de Visão Computacional é descrito na Seção 3. A Seção 4 descreve a modificação proposta da infraestrutura existente. A Seção 5 descreve o dataset. A Seção 6 fornece os resultados e analisa os casos de falha. As Seções 7 e 8 resumem o trabalho e delineiam as direções futuras, respectivamente.

---

## 2. TRABALHOS RELACIONADOS

Os erros nas decisões de impedimento têm sido estudados há muito tempo em uma tentativa de melhorar sua precisão [6]. Já foram feitos esforços para analisar e compreender os problemas que os árbitros em campo enfrentam ao tomar decisões de impedimento [7]. Essas iniciativas revelaram problemas comuns como obstruções na linha de visão do árbitro, oclusões e imagens residuais [8]. A introdução do VAR demonstrou uma melhoria na precisão ao lidar com esses problemas usando múltiplas câmeras e transmissões; no entanto, gerou problemas igualmente críticos, como atrasos de tempo. O VAR tem sido fortemente criticado pelo tempo gasto para anunciar as decisões, chegando a 5-6 minutos em certas partidas [9][10]. A precisão das decisões de impedimento por revisão do VAR também foi questionada devido a erros na elaboração das projeções das partes do corpo dos jogadores [11].

Junto com esses esforços para compreender as dificuldades na tomada de decisão de impedimento, várias tentativas também foram feitas para enfrentar esses problemas e automatizar o processo até certo ponto. Problemas decorrentes de oclusão na visão do árbitro em campo foram abordados usando transmissão de vídeo de múltiplas câmeras [12][13]. No entanto, esses trabalhos não fornecem detalhes suficientes sobre como as diversas subtarefas necessárias — encontrar posições das partes do corpo dos jogadores, rastrear jogadores junto com a bola e classificar jogadores em times — podem ser realizadas em dados de imagem.

K. Muthuraman et al. utilizaram técnicas de Visão Computacional e Processamento de Imagens para marcar dinamicamente uma linha de impedimento [14]. Neste trabalho, eles fornecem um método consideravelmente robusto para rastrear jogadores usando o rastreador Kanade-Lucas-Tomasi e um método para obter e usar um ponto de fuga para determinar a posição relativa dos jogadores. No entanto, seu trabalho não aborda a **condição de parte do corpo jogável** — cláusula importante na regra de impedimento que estabelece que apenas a posição relativa das partes do corpo jogáveis dos atacantes (cabeça, pé, joelho e ombros) pode ser usada para julgar o impedimento.

Jagjeet et al. utilizaram detecção de jogadores baseada em aprendizado profundo para obter posições e tomar decisões de impedimento [15], mas igualmente não consideram a cláusula de parte do corpo jogável nem o efeito de distorções de câmeras. Parth et al. utilizaram rastreamento de bola e jogadores para detectar passes e determinar impedimentos usando posições relativas [16], adicionando o contexto do passe, mas usando apenas posições aproximadas dos pés.

A tarefa de detecção automática de impedimento consiste em várias subtarefas: detecção de jogadores, estimativa de pose e classificação de times. Para estimativa de pose, técnicas genéricas como [17] e [18] podem ser usadas, mas trabalhos específicos para esportes também existem: para o futebol, [19] usa um sistema com múltiplas câmeras para converter estimativas de pose 2D em um mapa de pose 3D; [20] usa uma CNN profunda para detectar a pose do jogador e do taco no hóquei. Para classificação de times, abordagens de clustering e correspondência de características têm sido usadas [21][22]. A classificação do goleiro ainda não foi resolvida satisfatoriamente — [23] usa rastreamento manual próximo ao goleiro.

---

## 3. PIPELINE DE TAREFAS DE VISÃO COMPUTACIONAL

Neste trabalho, um método para gerar representações computacionais de imagens de cenas de partidas de futebol foi apresentado como um pipeline de tarefas de Visão Computacional e Processamento de Imagens a serem realizadas em sequência específica. O pipeline é projetado com base em certas suposições: as imagens são capturadas de uma câmera única cobrindo metade do campo, e as linhas de demarcação no campo são visíveis nas imagens. O dataset apresentado neste artigo adere a essas suposições.

O primeiro passo é encontrar pontos de fuga para contrarrestar o efeito das distorções da câmera ao determinar posições relativas dos jogadores e partes do corpo. Os pontos de fuga também são usados para obter o plano do campo, utilizado posteriormente para projeção de partes do corpo fora do plano do solo. Cada jogador na imagem é representado por uma estrutura de dados com as informações necessárias para tomar uma decisão de impedimento, consistindo em três entidades:

1. **Estimativa de pose** para o jogador e posições das partes do corpo 'chave'
2. **ID do time** para o jogador, indicando se pertence ao time atacante ou defensor
3. A **projeção mais avançada** das partes do corpo 'chave' do jogador no plano do campo

Essa representação computacional de cada jogador é obtida combinando a saída das subtarefas individuais detalhadas nas seções seguintes. O algoritmo central de impedimento processa os dados de cada jogador usando uma implementação computacional da regra oficial e dá a decisão final.

### 3.1. Algoritmo de Detecção de Ponto de Fuga

Determinar as posições relativas dos jogadores no campo é crucial na detecção de impedimento. Ao resolver as posições relativas usando imagens 2D de vista lateral, é necessário contrarrestar as distorções introduzidas pelo posicionamento da câmera. O trabalho de K. Muthuraman et al. usa o conceito de Ponto de Fuga para determinar as posições reais, estabelecendo-o pela interseção das linhas dos padrões de grama no gramado [11]. No entanto, nem todo gramado tem padrões bem definidos com linhas horizontais precisas. Portanto, neste trabalho são utilizadas as **linhas de demarcação oficiais** no campo (linha do meio-campo, limite da área do pênalti) para calcular o ponto de fuga.

O algoritmo de detecção de bordas **Canny** é usado para detectar bordas, e o algoritmo **Houghline** é aplicado para extrair linhas da imagem. A interseção dessas linhas, o ponto de fuga, é calculada em um plano estendido. Para medir as posições relativas dos jogadores, considera-se o ângulo entre a linha que conecta uma parte do corpo chave ao ponto de fuga e o limite horizontal da imagem.

Para determinar conclusivamente o plano do campo, um 'ponto de fuga vertical' e um 'ponto de fuga horizontal' são calculados usando linhas perpendiculares entre si. Para tornar o cálculo independente da posição da câmera, um intervalo angular é considerado ao selecionar as linhas no campo, em vez de valores fixos. Os valores máximos e mínimos do ângulo são determinados empiricamente.

### 3.2. Detecção de Jogadores

O primeiro passo para a detecção bem-sucedida de impedimento é a localização precisa dos jogadores. Cada jogador precisa ser detectado e classificado em categorias: impedido, não impedido, último defensor ou defensor. Simplesmente detectar jogadores como caixas delimitadoras não se mostra útil para determinar a linha mais avançada do jogador, pois as caixas podem cobrir mais área do que o corpo real do jogador. Além disso, a regra de impedimento verifica a **parte do corpo mais avançada permitida**, o que não pode ser representado com precisão pelas quatro coordenadas da caixa delimitadora.

A estimativa de pose de [20] é usada para detectar partes individuais do corpo de cada jogador — **cabeça, tornozelos, joelhos, quadris e ombros** — representadas por um único ponto no sistema de coordenadas x-y. As linhas que passam pelo ponto de fuga e pelos pontos do corpo são usadas para determinar a parte do corpo mais avançada permitida.

Como os quadros de entrada também contêm detecções indesejadas (substitutos, meninos de bola, plateia), é necessário segmentar a área de jogo. Isso é feito em duas etapas: primeiro, uma **máscara de cor** é aplicada para segmentar o gramado; segundo, **transformações morfológicas** são aplicadas à imagem mascarada para segmentar a plateia da área de jogo. Quaisquer previsões fora do campo são descartadas. O método baseado em estimativa de pose alcança uma **precisão de 94,24%** no dataset.

### 3.3. Classificação de Times

A classificação de times é a tarefa de determinar o time ao qual cada jogador pertence. Essa tarefa é realizada de forma **não supervisionada** usando algoritmos de clustering. Com as localizações das partes do corpo, o time de cada jogador é identificado com base nas características extraídas da cor de sua camisa e shorts. No entanto, toda a região da camisa e shorts contém muito ruído (números de identificação, nomes de jogadores, logos de patrocinadores). Para superar isso, usa-se uma porção específica: o polígono representado pelos dois pontos dos joelhos e pelos dois pontos que ficam na metade da distância entre os pontos do ombro e do quadril.

Essa região de interesse para cada jogador é usada para extrair características: todos os pixels no e dentro do polígono são convertidos em **três histogramas (r/g/b)** concatenados para formar um vetor de características para os algoritmos de clustering.

O objetivo não é apenas atribuir times, mas também identificar **árbitros e goleiros**: árbitros são excluídos das tarefas posteriores, e goleiros são tratados separadamente no algoritmo central. Como condição obrigatória no futebol, a cor do uniforme dos goleiros é significativamente diferente de seus respectivos times, assim como a dos árbitros.

Para explorar a presença de pontos de dados ruidosos (goleiros e árbitros) e um número conhecido de clusters ideais (dois times), foi usado um **ensemble de dois algoritmos de clustering**: KMeans e DBSCAN.

- O **DBSCAN** é conhecido por identificar pontos de dados ruidosos e separá-los do processo de clustering. No entanto, produz um número variável de clusters — propriedade indesejada quando o número necessário é conhecido.
- O **KMeans** garante exatamente dois clusters para os dados não ruidosos.

Um ponto de dado foi classificado como entidade estranha (goleiro ou árbitro) se classificado como ruído pelo DBSCAN; caso contrário, recebeu o rótulo atribuído pelo KMeans.

Testar essa abordagem ensemble no dataset revela uma **precisão de 95,75%** para classificar corretamente os jogadores em dois times e marcar árbitros e goleiros separadamente. Um problema persistente foi observado para goleiros com uniformes fluorescentes quando os times usavam uniformes predominantemente brancos. Para demonstrar o efeito desse problema, o pipeline também foi aplicado ao dataset incorporando manualmente os goleiros — uma comparação quantitativa é apresentada na Seção 6.

### 3.4. Projeção de Partes do Corpo

Cada parte do corpo permitida do jogador tem papel significativo na decisão de impedimento. Se a parte do corpo mais avançada do jogador não estiver no plano do solo (como no caso de um ombro), ela precisa ser **projetada no plano do solo**.

Para conseguir isso, a parte do corpo é projetada na linha que conecta o ponto de fuga e o tornozelo do mesmo lado, pois essa linha ficará no plano do solo. Em alguns casos, a estimativa de pose baseada em aprendizado profundo falha ao localizar o tornozelo com confiabilidade suficiente. Nesse caso, a equação do plano do solo é calculada usando duas linhas que se intersectam no solo, e a parte do corpo é projetada aproximando sua elevação usando um método heurístico.

Por meio de revisão da literatura sobre anatomia humana [24], a razão do corpo humano inferior para o superior (até os ombros) foi encontrada como **1,6**. Usando esse valor e a distância entre o quadril e o ombro, a elevação é estimada e a parte do corpo é projetada no plano. A altura do ombro é dada por:

> **heights = 2,6 × √((xs − xh)² + (ys − yh)²)**

onde `heights` é a altura do ombro do solo, `(xs, ys)` são as coordenadas do ombro e `(xh, yh)` são as coordenadas do quadril.

Para lidar com jogadores cujo corpo não é perpendicular ao plano do solo, primeiro o ângulo de inclinação θ é calculado usando um vetor do quadril ao ombro, e a fórmula é ajustada com cos θ:

> **heights = 2,6 × √((xs − xh)² + (ys − yh)²) × cos θ**

### 3.5. Algoritmo Central de Impedimento

Com a ajuda de todos os módulos anteriores, a pose, o time e as projeções de cada jogador são obtidos. Esta seção converte a regra oficial de impedimento em uma função computacional.

O primeiro passo é encontrar o **último defensor** no campo. O jogador do time defensor que forma o **ângulo máximo com o ponto de fuga** é considerado o último defensor. Esse ângulo é comparado com os ângulos dos jogadores do time atacante. Os atacantes com ângulos de ponto de fuga maiores do que o do último defensor são declarados impedidos.

**Algoritmo 1: Algoritmo de Detecção de Impedimento**

```
Entrada → Jogadores, IdTimeAtacante, IdTimeDefensor
Saída  → Decisões de Impedimento

// Encontrar a posição do 'Penúltimo Defensor' em relação ao ponto de fuga
PosAngularDefesa = []
para cada Jogador em Jogadores:
    se Jogador[id] == IdTimeDefensor:
        PosAngularDefesa.append(Jogador[AnguloNoPontoDeFuga])

Ordenar(PosAngularDefesa, Decrescente)
PosAngularMinima = PosAngularDefesa[1]

// Verificar jogadores do time atacante para impedimento
para cada Jogador em Jogadores:
    se Jogador[id] == IdTimeAtacante:
        se Jogador[AnguloNoPontoDeFuga] < PosAngularMinima:
            Jogador[DecisaoDeImpedimento] = IMPEDIDO
        senão:
            Jogador[DecisaoDeImpedimento] = NAO_IMPEDIDO
```

---

## 4. MODIFICAÇÃO DA INFRAESTRUTURA EXISTENTE

A infraestrutura existente consiste em um árbitro externo monitorando o jogo e revisando as decisões do árbitro em campo. Nosso sistema pode ser adotado como **módulo assistivo ao árbitro externo**. Usando este módulo, o árbitro pode obter decisões precisas por quadro e por jogador em questão de segundos, eliminando o tempo gasto para comparar manualmente as posições de cada jogador. Os erros decorrentes do julgamento humano também são drasticamente reduzidos, pois o sistema computacional de ponta a ponta toma decisões matematicamente, introduzindo pouca ou nenhuma possibilidade de erros decorrentes de cálculos errados nas projeções.

Em uma partida real, quando o árbitro externo é solicitado a dar uma decisão, ele pode identificar o quadro no momento em que o passe foi feito pelo jogador que está realizando o passe e fornecer o timestamp correspondente ao nosso módulo. O sistema receberá o quadro como entrada e produzirá uma imagem com todos os jogadores detectados e classificados como impedidos ou não impedidos.

---

## 5. DESCRIÇÃO DO DATASET

O dataset consiste em **500 imagens** coletadas de partidas de futebol do mundo real, extraídas de câmeras presentes em estádios de futebol para transmissão. Todas as restrições para as imagens foram seguidas. A visão obtida dessas câmeras é a visão disponível para o VAR. As imagens foram coletadas de forma a apresentar uma coleção diversa de eventos que podem ocorrer em um jogo, incluindo vários cenários de ambos os lados do campo e situações de bola parada como cobranças de falta. Como a regra de impedimento não se aplica durante escanteios e arremessos laterais, tais casos não estão presentes no dataset.

Cada pessoa no campo é rotulada a partir do seguinte conjunto: `{ árbitro, goleiroA, goleiroB, timeA, timeB }`. Além disso, cada jogador do timeA e timeB é rotulado como impedido ou não impedido, considerando-se o timeA como atacante e o timeB como defensor e vice-versa. Isso fornece um total de **1000 pontos de dados** para avaliar o desempenho do algoritmo.

O dataset também inclui dados de **verdade fundamental** além das imagens. Cada ponto de dado nessa verdade fundamental é uma representação quantitativa de um jogador na imagem, contendo:

- **Posições das partes do corpo chave** do jogador, como coordenadas na imagem
- O **time real** ao qual o jogador pertence (denotado por 1 ou 0)
- O **ID do time defensor e atacante** atual (1 para atacante, 0 para defensor)
- Uma **decisão de impedimento** se o jogador pertence ao time atacante

As características extraídas do pipeline de Visão Computacional podem ser comparadas com a verdade fundamental usando um script disponibilizado com o dataset. O script usa um método de atribuição de soma linear para mapear jogadores nas previsões para jogadores na verdade fundamental usando suas coordenadas de partes do corpo chave. Este script calcula a precisão de cada tarefa individual e a precisão geral da decisão de impedimento.

---

## 6. RESULTADOS E AVALIAÇÃO

Nesta seção, testamos nossa metodologia em todas as 500 imagens do dataset e fornecemos uma análise dos resultados, discutindo as métricas corretas para avaliar a tarefa de detecção de impedimento. Também fornecemos as precisões das tarefas individuais de detecção de jogadores e classificação de times, e comentamos sobre as deficiências da abordagem.

Identificar o goleiro do time defensor é crucial para detectar jogadores em posição de impedimento. Como detectar o goleiro automaticamente é difícil e pode necessitar de rastreamento manual, fornecemos uma comparação entre o desempenho do pipeline usando o método de rastreamento manual e a abordagem DBSCAN.

### 6.1. Métricas para Avaliação

Para detecção de jogadores, calculamos a precisão em termos de porcentagem do total de jogadores corretamente detectados no campo. Como os dados são altamente assimétricos (em média, 10-12 jogadores por imagem, mas apenas 0-4 em posição de impedimento), o Verdadeiro Negativo seria altamente enganoso, tornando a especificidade uma métrica inadequada. É necessário considerar **precisão, recall e F score**:

> **Precisão** = Decisões Corretas de Impedimento / Total de Decisões de Impedimento
>
> **Recall** = Decisões Corretas de Impedimento / Total de Decisões Corretas de Impedimento
>
> **F1 Score** = 2 × (Precisão × Recall) / (Precisão + Recall)

A precisão do algoritmo de classificação de times é **95,75%** e do módulo de detecção de jogadores é **94,24%**. A Tabela 1 apresenta os resultados do sistema com e sem rastreamento manual do goleiro.

**Tabela 1: Comparação de desempenho com atribuição automática (DBSCAN) e manual da posição do goleiro (GK)**

| Atribuição GK | Precisão      | Recall        | F1 Score      |
|---------------|---------------|---------------|---------------|
| DBSCAN        | 0,87 ± 0,05   | 0,91 ± 0,02   | 0,85 ± 0,03   |
| Manual        | 0,96 ± 0,02   | 0,96 ± 0,02   | 0,98 ± 0,02   |

Ao avaliar os resultados, as imprecisões foram identificadas em certas deficiências específicas do sistema:
1. Detecção incorreta do goleiro
2. Classificação incorreta de time
3. Falha na detecção do jogador devido à oclusão
4. Falha na detecção do jogador devido ao desfoque causado por movimento

A detecção incorreta do goleiro resultou em considerável perda de precisão. O pipeline apresentado alcança um **F1 score de 0,85**. No entanto, se a atribuição automática de goleiros for tratada com maior precisão, os resultados mostrarão um aumento significativo — fornecendo manualmente a posição do goleiro, o F1 score sobe para **0,98** nesse dataset. Resolver qualquer uma das deficiências mencionadas pode contribuir para o aumento da precisão geral do sistema.

---

## 7. CONCLUSÃO

Este trabalho apresentou com sucesso um framework de Visão Computacional para melhorar as decisões de impedimento nas partidas de futebol. Ele também fornece um dataset abrangente para testar a abordagem e pode ser considerado como um ponto de partida para pesquisas futuras relacionadas à automação da tomada de decisão no futebol usando técnicas de Visão Computacional.

Um desempenho de baseline do framework no novo dataset de detecção de impedimento foi apresentado. As decisões obtidas deste sistema são instantâneas e mostram um **F1 score de 0,85**. Um algoritmo para a regra de impedimento do futebol foi implementado, permitindo a tomada de decisão computacional para impedimentos. As deficiências do sistema também foram discutidas.

O artigo detalha as tarefas individuais de Visão Computacional que podem ser combinadas para gerar uma representação de imagens de partidas de futebol, sobre a qual o algoritmo de impedimento funciona. Ele também coloca o sistema proposto em contexto com o cenário do mundo real, discutindo como ele pode se encaixar perfeitamente no processo existente do VAR para auxiliar o árbitro externo a tomar a decisão correta.

---

## 8. TRABALHOS FUTUROS

Trabalhos futuros sobre esse problema podem tomar múltiplas direções:

- **Melhoria das tarefas fundamentais de Visão Computacional**: estimativa de pose e classificação de times têm espaço para aumento de precisão e completa automação.
- **Oclusão e desfoque**: a estimativa de pose baseada em aprendizado profundo é suscetível a problemas como oclusão de jogadores e imagens desfocadas. O rastreamento pode ser usado para lidar com oclusão e movimentos repentinos de jogadores.
- **Detecção automática do goleiro**: melhorar a precisão dessa tarefa é crucial, dado seu impacto evidente nos resultados gerais.
- **Sistema completamente autônomo**: realizar pesquisas sobre classificação de eventos e ações dos jogadores no futebol. A classificação precisa de ações como o passe da bola e o envolvimento ativo no jogo removeria a necessidade de intervenção humana, pois o sistema poderia ser automaticamente acionado para sequências de vídeo correspondentes a essas ações.

---

## REFERÊNCIAS

[1] FIFA. Video assistant referees (VAR): How does video assistant referee improve the game? fifa.com, 2018.

[2] Dale Johnson. The ultimate guide to VAR in the premier league — all your questions answered. Sony ESPN FC, 2019.

[3] Peter Catteeuw, Bart Gilis, Arne Jaspers e Johan Wagemans. Training of perceptual-cognitive skills in offside decision making. *J Sport Exerc Psychol.* 32:845–861, 2010.

[4] Werner F. Helsen, Bart Gilis e Matthew Weston. Errors in judging "offside" in association football: Test of the optical error versus the perceptual flash-lag hypothesis. *J Sports Sci.* 24:1–8, 2006.

[5] Raôul R. D. Oudejans e Frank C. Bakker. How position and motion of expert assistant referees in soccer relate to the quality of their offside judgements during actual match play. *Int J Sport Psychol.* 36:3–21, 2005.

[6] Bart Gilis, Werner Helsen, Peter Catteeuw e Johan Wagemans. Offside decisions by expert assistant referees in association football: Perception and recall of spatial positions in complex dynamic events. *Journal of Experimental Psychology: Applied*, 14(1), 21–35, 2008.

[7] Sirimamayvadee Siratanita, Kosin Chamnongthai e Mistusji Muneyasu. Saliency-based football offside detection. *17th International Symposium on Communications and Information Technologies (ISCIT)*, Cairns, QLD, 2017.

[8] Michael McKinney. The persistence of vision. vision.org, 2005.

[9] Telegraph Sport. VAR in the premier league explained: How does it work, what decisions can be changed and why is it controversial? www.telegraph.co.uk, 2019.

[10] Chris Nee. Is VAR causing more problems than it solves? thesetpieces.com, 2019.

[11] Elliott Jackson. Chris Kamara says Roberto Firmino offside decision 'was made up' as VAR controversy grows over Liverpool incident. www.liverpoolecho.co.uk, 2019.

[12] Sirimamayvadee Siratanita, Kosin Chamnongthai e Mistusji Muneyasu. A method of saliency-based football-offside detection using six cameras. *Global Wireless Summit (GWS)*, Cape Town, pp. 127-131, 2017.

[13] Sirimamayvadee Siratanita, Kosin Chamnongthai e Mistusji Muneyasu. A method of football-offside detection using multiple cameras for an automatic linesman assistance system. *Wireless Personal Communications*, 2019.

[14] Karthik Muthuraman, Pranav Joshi e Suraj K. Raman. Vision based dynamic offside line marker for soccer games. *arXiv preprint arXiv:1804.06438*, 2018.

[15] Jagjeet Singh. Vision based method for offside detection in soccer. github.com/jagjeet-singh/Vision-based-method-for-offside-detection-in-soccer, 2017.

[16] Parthasarathi Khirwadkar, Sriram Yenamandra, Rahul Chanduka e Ishank Juneja. Offside detection system for football. github.com/kparth98/ITSP-Project, 2017.

[17] Eldar Insafutdinov et al. Deepercut: A deeper, stronger, and faster multi-person pose estimation model. *European Conference on Computer Vision (ECCV)*, 2017.

[18] Ke Sun, Bin Xiao, Dong Liu e Jingdong Wang. Deep high-resolution representation learning for human pose estimation. *CoRR*, abs/1902.09212, 2019.

[19] Helmut Neher, Kanav Vats, Alexander Wong e David A. Clausi. Hyperstacknet: A hyper stacked hourglass deep convolutional neural network architecture for joint player and stick pose estimation in hockey. pp. 313–320, 2018.

[20] Bridgeman Lewis, Volino Marco, Guillemaut Jean-Yves e Hilton Adrian. Multi-person 3D pose estimation and tracking in sports. Junho de 2019.

[21] Theagarajan Rajkumar, Pala Federico, Zhang Xiu e Bhanu Bir. Soccer: Who has the ball? Generating visual analytics and player statistics. Junho de 2018.

[22] Paolo Spagnolo, Nicola Mosca, Massimiliano Nitti e Arcangelo Distante. An unsupervised approach for segmentation and clustering of soccer players. pp. 133–142, 2007.

[23] Yasushi Akiyama, Rodolfo Garcia Barrantes e Tyson Hynes. Video scene extraction tool for soccer goalkeeper performance data analysis. 2019.

[24] Evon M. O. Abu-Taieh e Hamed S. Al-Bdour. A human body mathematical model biometric using golden ratio: A new algorithm. *Machine Learning and Biometrics*, 2018.
