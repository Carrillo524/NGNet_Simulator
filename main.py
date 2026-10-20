import os
from Lógica.NetworkBuilder import generar_red_computadores
from Lógica.AutomataNetwork import AutomataComunicacion
from Lógica.GnnInference import EvaluadorRedGNN

def ejecutar_simulacion():
    print("=" * 50)
    print(" 🌐 INICIANDO SISTEMA DE MONITOREO DE RED (GNN + AFD) ")
    print("=" * 50)

    # 1. Generar la red de prueba
    print("\n[1/3] Generando topología de red simulada...")
    red = generar_red_computadores(num_nodos=10, prob_conexion=0.25)
    print(f"✔ Red generada con {red.number_of_nodes()} nodos y {red.number_of_edges()} conexiones.")

    # 2. Inferencia con la GNN
    print("\n[2/3] Evaluando estado de nodos con la GNN (modelo_gat.pth)...")
    ruta_modelo = os.path.join(os.path.dirname(__file__), 'modelo_gat.pth')
    
    # Si el modelo está dentro de Lógica, ajusta a: os.path.join(os.path.dirname(__file__), 'Lógica', 'modelo_gat.pth')
    if not os.path.exists(ruta_modelo):
        ruta_modelo = os.path.join(os.path.dirname(__file__), 'Lógica', 'modelo_gat.pth')

    evaluador = EvaluadorRedGNN(ruta_modelo)
    predicciones = evaluador.predecir_anomalias(red)

    print("\n--- Clasificación de Nodos por la GNN ---")
    nodos_anomalos = []
    for nodo, estado_pred in predicciones.items():
        tipo = red.nodes[nodo]['tipo']
        trafico = red.nodes[nodo]['trafico']
        cpu = red.nodes[nodo]['cpu']
        print(f"Nodo {nodo:2d} | Tipo: {tipo:10s} | Tráfico: {trafico:.2f} | CPU: {cpu:.2f} | GNN: {estado_pred}")
        if estado_pred == "Anómalo":
            nodos_anomalos.append(nodo)

    if nodos_anomalos:
        print(f"\n⚠️ Alerta: Se detectaron {len(nodos_anomalos)} nodos anómalos: {nodos_anomalos}")
    else:
        print("\n🟢 Todos los nodos operan dentro de los parámetros normales.")

    # 3. Validación de comunicaciones con el Autómata
    print("\n[3/3] Probando ciclo de vida de comunicaciones con el Autómata (AFD)...")
    automata = AutomataComunicacion()

    secuencias_prueba = [
        ("PWSR", "Secuencia normal de envío y reposo"),
        ("PWER", "Secuencia con interrupción/error"),
        ("PR",   "Secuencia inválida o directa")
    ]

    for cad, desc in secuencias_prueba:
        valida, estado_fin = automata.procesar_cadena(cad)
        if 'E' in cad:
            resultado_str = "Finalizada con ERROR (Recuperado)"
        elif valida:
            resultado_str = "Aceptada (Éxito)"
        else:
            resultado_str = "Rechazada (Transición Inválida)"
            
        print(f" Cadena: '{cad}' ({desc}) -> {resultado_str} | Estado final: {estado_fin}")

    print("\n" + "=" * 50)
    print(" ✔ INTEGRACIÓN DE MÓDULOS COMPLETADA CON ÉXITO")
    print("=" * 50)

if __name__ == "__main__":
    ejecutar_simulacion()