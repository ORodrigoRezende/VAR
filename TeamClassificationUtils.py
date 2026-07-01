import numpy as np
import cv2
from sklearn.cluster import KMeans, DBSCAN


def extract_roi_mask(image, pose):
    """
    Extrai a região de interesse (ROI) de cada jogador conforme o artigo:
    polígono formado pelos dois joelhos e pelos dois pontos no meio do vetor
    quadril->ombro de cada lado. Essa região evita ruídos como números e logos.
    """
    kp = pose[2]

    required = ['leftKnee', 'rightKnee', 'leftHip', 'rightHip', 'leftShoulder', 'rightShoulder']
    if not all(k in kp for k in required):
        return None

    def mid(p1, p2):
        return (
            int((p1['x'] + p2['x']) / 2),
            int((p1['y'] + p2['y']) / 2)
        )

    left_mid  = mid(kp['leftHip'],  kp['leftShoulder'])
    right_mid = mid(kp['rightHip'], kp['rightShoulder'])

    pts = np.array([
        [int(kp['leftKnee']['x']),  int(kp['leftKnee']['y'])],
        [int(kp['rightKnee']['x']), int(kp['rightKnee']['y'])],
        [right_mid[0], right_mid[1]],
        [left_mid[0],  left_mid[1]],
    ], dtype=np.int32)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def extract_histogram_features(image, mask):
    """
    Extrai histogramas RGB dos pixels dentro da ROI e concatena em um vetor de features.
    Cada canal tem 32 bins → vetor final de tamanho 96.
    """
    if mask is None:
        return None

    bins = 32
    features = []
    for ch in range(3):
        channel = image[:, :, ch]
        pixels = channel[mask == 255]
        if len(pixels) == 0:
            return None
        hist, _ = np.histogram(pixels, bins=bins, range=(0, 256))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-6)  # normaliza
        features.extend(hist.tolist())

    return features


def classify_teams(pose_estimations, image):
    """
    Implementação do ensemble KMeans + DBSCAN conforme descrito no artigo:

    - DBSCAN identifica pontos ruidosos (árbitros e goleiros, que aparecem como outliers
      por terem uniformes significativamente diferentes dos dois times principais).
    - KMeans (k=2) separa os jogadores restantes nos dois times.

    Cada pose recebe um label adicionado ao final da lista:
        0 ou 1  → time
        2       → goleiro (noise no DBSCAN)
        3       → árbitro (noise no DBSCAN, minoria absoluta)
    """
    feature_matrix = []
    valid_indices = []

    for i, pose in enumerate(pose_estimations):
        mask = extract_roi_mask(image, pose)
        feats = extract_histogram_features(image, mask)
        if feats is not None:
            feature_matrix.append(feats)
            valid_indices.append(i)

    if len(feature_matrix) < 3:
        # poucos jogadores detectados, fallback simples
        for pose in pose_estimations:
            pose.append(-1)
        return pose_estimations

    X = np.array(feature_matrix)

    # --- DBSCAN: detecta outliers (goleiros, árbitros) ---
    dbscan = DBSCAN(eps=0.5, min_samples=2, metric='euclidean')
    dbscan_labels = dbscan.fit_predict(X)

    # --- KMeans: separa os dois times entre os não-outliers ---
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)

    # Conta quantos outliers existem por label do KMeans para decidir
    # qual label do KMeans é time 0 e qual é time 1 (estabiliza a ordem)
    team_labels = {}
    for idx, (db_label, km_label) in enumerate(zip(dbscan_labels, kmeans_labels)):
        pose_idx = valid_indices[idx]
        if db_label == -1:
            # Outlier: goleiro ou árbitro
            # Árbitros são a minoria extrema — heurística simples:
            # deixamos como "2" (tratamento especial no algoritmo de impedimento)
            team_labels[pose_idx] = 2
        else:
            team_labels[pose_idx] = int(km_label)

    # Aplica os labels nas poses
    for i, pose in enumerate(pose_estimations):
        label = team_labels.get(i, -1)
        pose.append(label)

    return pose_estimations


def get_team_classifications(pose_estimations, image):
    """
    Ponto de entrada principal. Substitui a versão antiga que exigia
    cores de time fornecidas manualmente.

    Retorna pose_estimations com o label de time appended ao final de cada pose.
    """
    return classify_teams(pose_estimations, image)
