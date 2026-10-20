class AutomataComunicacion:
    def __init__(self):
        self.estado_inicial = 'q0'
        # q0 es estado final (reposo / ciclo completado tras R)
        self.estados_finales = {'q0', 'qf', 'qe'}
        
        # Tabla de transiciones δ(q, σ)
        self.transiciones = {
            'q0': {'P': 'q1', 'R': 'q0'},
            'q1': {'W': 'q2', 'E': 'qe'},
            'q2': {'S': 'qf', 'E': 'qe'},
            'qf': {'R': 'q0', 'P': 'q1'}, # Volver a reposo R o iniciar nuevo paquete P
            'qe': {'R': 'q0', 'P': 'q1'}  # Volver a reposo R o reintentar paquete P
        }

    def procesar_cadena(self, cadena_eventos):
        estado_actual = self.estado_inicial
        traza = [estado_actual]
        
        for simbolo in cadena_eventos:
            if simbolo in self.transiciones.get(estado_actual, {}):
                estado_actual = self.transiciones[estado_actual][simbolo]
                traza.append(estado_actual)
            else:
                return False, estado_actual, traza
        
        es_valida = estado_actual in self.estados_finales
        return es_valida, estado_actual, traza