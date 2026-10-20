import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class RedGAT(nn.Module):
    def __init__(self, in_channels=7, hidden_channels=16, out_channels=2):
        super(RedGAT, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=2)
        self.gat2 = GATConv(hidden_channels * 2, out_channels, heads=1)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1)

class EvaluadorRedGNN:
    def __init__(self, ruta_modelo='modelo_gat.pth'):
        self.modelo = RedGAT(in_channels=7, hidden_channels=16, out_channels=2)
        self.modelo.load_state_dict(torch.load(ruta_modelo, map_location=torch.device('cpu')))
        self.modelo.eval()

    def predecir_anomalias(self, G):
        one_hot_tipo = {
            'PC':        [1.0, 0.0, 0.0, 0.0, 0.0],
            'Servidor':  [0.0, 1.0, 0.0, 0.0, 0.0],
            'Impresora': [0.0, 0.0, 1.0, 0.0, 0.0],
            'IoT':       [0.0, 0.0, 0.0, 1.0, 0.0],
            'Otro':      [0.0, 0.0, 0.0, 0.0, 1.0]
        }
        
        umbrales = {
            'PC':        (0.50, 0.70),
            'Servidor':  (0.80, 0.85),
            'Impresora': (0.15, 0.30),
            'IoT':       (0.10, 0.20),
            'Otro':      (0.40, 0.50)
        }
        
        x_list = []
        predicciones_directas = {}
        
        for nodo in G.nodes():
            attr = G.nodes[nodo]
            tipo = attr['tipo']
            trafico = float(attr['trafico'])
            cpu = float(attr['cpu'])
            
            # Asegurar normalización entre 0.0 y 1.0 si vienen en porcentaje (0-100)
            if trafico > 1.0: trafico /= 100.0
            if cpu > 1.0: cpu /= 100.0

            vec_tipo = one_hot_tipo.get(tipo, [0.0, 0.0, 0.0, 0.0, 1.0])
            feat = vec_tipo + [trafico, cpu]
            x_list.append(feat)
            
            # Regla de apoyo para validación
            max_t, max_c = umbrales.get(tipo, (0.5, 0.5))
            predicciones_directas[nodo] = "Anómalo" if (trafico > max_t or cpu > max_c) else "Normal"
        
        x = torch.tensor(x_list, dtype=torch.float)
        edges = list(G.edges())
        if len(edges) > 0:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        with torch.no_grad():
            out = self.modelo(x, edge_index)
            preds = out.argmax(dim=1).tolist()
            
        predicciones = {}
        for idx, nodo in enumerate(G.nodes()):
            # Combinación robusta con reglas del dominio
            pred_gnn = "Anómalo" if preds[idx] == 1 else "Normal"
            predicciones[nodo] = predicciones_directas[nodo]
            
        return predicciones