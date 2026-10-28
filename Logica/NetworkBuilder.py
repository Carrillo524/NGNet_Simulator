import networkx as nx
import random
import math

def generar_red_computadores(num_nodos=12, prob_conexion=0.40):
    G = nx.Graph()
    radio = 240.0

    for i in range(num_nodos):
        angulo = (2 * math.pi * i) / num_nodos
        pos_x = round(radio * math.cos(angulo), 2)
        pos_y = round(radio * math.sin(angulo), 2)
        G.add_node(i, x=pos_x, y=pos_y)

    for i in range(num_nodos):
        G.add_edge(i, (i + 1) % num_nodos)

    for i in range(num_nodos):
        j3 = (i + 3) % num_nodos
        if not G.has_edge(i, j3) and random.random() < prob_conexion:
            G.add_edge(i, j3)
        j4 = (i + 4) % num_nodos
        if not G.has_edge(i, j4) and random.random() < prob_conexion:
            G.add_edge(i, j4)

    for i in range(num_nodos):
        if G.degree[i] < 3:
            candidatos = [(i + 3) % num_nodos, (i + 4) % num_nodos, (i + 5) % num_nodos]
            for c in candidatos:
                if not G.has_edge(i, c):
                    G.add_edge(i, c)
                    break

    tipos = ['PC', 'Servidor', 'Impresora', 'IoT', 'Otro']
    umbrales = {
        'PC':        (0.50, 0.70),
        'Servidor':  (0.80, 0.85),
        'Impresora': (0.15, 0.30),
        'IoT':       (0.10, 0.20),
        'Otro':      (0.40, 0.50)
    }

    for nodo in G.nodes():
        tipo = random.choice(tipos)
        trafico = round(random.uniform(0.05, 0.95), 2)
        cpu = round(random.uniform(0.05, 0.95), 2)
        max_t, max_c = umbrales[tipo]
        
        G.nodes[nodo]['tipo'] = tipo
        G.nodes[nodo]['trafico'] = trafico
        G.nodes[nodo]['cpu'] = cpu
        G.nodes[nodo]['estado'] = "Anómalo" if (trafico > max_t or cpu > max_c) else "Normal"
        G.nodes[nodo]['activo'] = True

    for u, v in G.edges():
        G.edges[u, v]['peso'] = round(random.uniform(1.0, 10.0), 1)
        G.edges[u, v]['capacidad'] = random.choice([100, 1000])
        G.edges[u, v]['activa'] = True

    return G
