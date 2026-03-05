import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from torch.nn.utils.rnn import pack_padded_sequence

# Importamos las piezas clave de tu archivo original
# Reemplaza 'training_newstructure' con el nombre real de tu archivo .py
from training_newstructure import Config, PatientRiskModel, PatientDataset, collate_fn, test_loader, train_loader, val_loader, train_model


def get_embeddings(model, loader):
    model.eval()
    embeddings = []
    labels = []
    
    print("Extraiendo embeddings del conjunto de test...")
    with torch.no_grad():
        for seq_batch, demo_batch, y_batch, lengths in loader:
            # Replicamos el flujo del forward() hasta la concatenación
            packed_x = pack_padded_sequence(seq_batch, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, (h_n, _) = model.lstm(packed_x)
            seq_embedding = h_n[-1] # Estado final de la LSTM
            
            demo_embedding = model.demo_encoder(demo_batch) # Salida del DNN demo
            
            # Combinamos ambos como hace tu modelo original
            combined = torch.cat((seq_embedding, demo_embedding), dim=1)
            
            embeddings.append(combined.cpu().numpy())
            labels.append(y_batch.cpu().numpy())
            
    return np.vstack(embeddings), np.concatenate(labels)

def run_visualisation(trained_model, data_loader):
    # 1. Obtener datos de alta dimensión
    emb, y = get_embeddings(trained_model, data_loader)

    # 2. Reducir a 2D con PCA para poder plotear
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(emb)

    # 3. Crear el "Meshgrid" (la rejilla de fondo)
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))

    # 4. Entrenar un clasificador Proxy (SVM) para dibujar la frontera
    # Esto imita cómo tu modelo LSTM-DNN separa las clases en 2D
    proxy_clf = SVC(kernel='rbf', C=1.0)
    proxy_clf.fit(X_pca, y)

    # 5. Predicción en la rejilla
    Z = proxy_clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 6. Generar el gráfico
    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu') # Fondo de color
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, s=30, 
                        edgecolor='k', cmap='RdYlBu', alpha=0.7)
    
    plt.title("Frontera de Decisión T2D (Proyección PCA de Embeddings)")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.colorbar(scatter, label="Riesgo (0: Sano, 1: T2D)")
    plt.show()

# --- BLOQUE PRINCIPAL ---
if __name__ == "__main__":
    # 1. Instanciar la configuración
    config = Config()
    
    # 2. Inicializar el modelo
    model = PatientRiskModel(config)
    
    # 3. Entrenar el modelo para obtener 'trained_model'
    # Nota: Esto usará los DataLoaders importados de training_newstructure
    print("Iniciando entrenamiento del modelo...")
    trained_model = train_model(model, train_loader, val_loader, config)
    
    # 4. Ejecutar la visualización
    print("Generando visualización...")
    run_visualisation(trained_model, test_loader)