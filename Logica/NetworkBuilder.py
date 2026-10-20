import networkx as nx
import random

def generar_red_computadores(num_nodos=12, prob_conexion=0.25):
    G = nx.erdos_renyi_graph(n=num_nodos, p=prob_conexion, seed=42)
    tipos = ['PC', 'Servidor', 'Impresora', 'IoT', 'Otro']
    
    umbrales = {
        'PC':        (0.50, 0.70),
        'Servidor':  (0.80, 0.85),
        'Impresora': (0.15, 0.30),
        'IoT':       (0.10, 0.20),
        'Otro':      (0.40, 0.50)
    }

    # Asignar atributos a los Nodos
    for nodo in G.nodes():
        tipo = random.choice(tipos)
        
        # Atributos normalizados (0.0 a 1.0)
        trafico = round(random.uniform(0.05, 0.95), 2)
        cpu = round(random.uniform(0.05, 0.95), 2)
        
        max_trafico, max_cpu = umbrales[tipo]
        es_anomalo = (trafico > max_trafico) or (cpu > max_cpu)
        
        G.nodes[nodo]['tipo'] = tipo
        G.nodes[nodo]['trafico'] = trafico
        G.nodes[nodo]['cpu'] = cpu
        G.nodes[nodo]['estado'] = "Anómalo" if es_anomalo else "Normal"
        G.nodes[nodo]['activo'] = True

    # Asignar Pesos y Atributos a las Aristas (Conexiones)
    for u, v in G.edges():
        # Peso: Latencia/Carga en ms o ancho de banda utilizado
        peso = round(random.uniform(1.0, 10.0), 1)
        capacidad = random.choice([100, 1000])  # Mbps
        G.edges[u, v]['peso'] = peso
        G.edges[u, v]['capacidad'] = capacidad
        G.edges[u, v]['activa'] = True

    return G