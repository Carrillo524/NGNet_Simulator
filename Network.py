import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
import os
import tempfile
import networkx as nx
import json
import random

from Logica.NetworkBuilder import generar_red_computadores
from Logica.AutomataNetwork import AutomataComunicacion
from Logica.GnnInference import EvaluadorRedGNN

st.set_page_config(page_title="NGNet Live Simulator", layout="wide")

# Estilos CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #1f2937; padding: 10px; border-radius: 8px; border: 1px solid #374151; }
    </style>
""", unsafe_allow_html=True)

st.title(" NGNet Simulator: Solución técnologica para la empresa SoftCast")
st.caption("Simulador interactivo con detección GNN, tráfico en tiempo real y módulo de validación formal por AFD")

# Estado de Sesión
if 'red' not in st.session_state:
    st.session_state.red = generar_red_computadores(num_nodos=12, prob_conexion=0.25)
    # Asignar nodos bloqueados aleatorios al iniciar
    nodos_init = list(st.session_state.red.nodes())
    st.session_state.nodos_bloqueados = set(random.sample(nodos_init, k=min(2, len(nodos_init))))
if 'nodos_bloqueados' not in st.session_state:
    st.session_state.nodos_bloqueados = set()
if 'conexiones_bloqueadas' not in st.session_state:
    st.session_state.conexiones_bloqueadas = set()
if 'anim_id' not in st.session_state:
    st.session_state.anim_id = 0

# Cargar GNN
ruta_modelo = os.path.join(os.path.dirname(__file__), 'modelo_gat.pth')
if not os.path.exists(ruta_modelo):
    ruta_modelo = os.path.join(os.path.dirname(__file__), 'Logica', 'modelo_gat.pth')

evaluador = EvaluadorRedGNN(ruta_modelo)

# Cálculo de Carga de Canal
def calcular_carga_acumulada_arista(G, u, v):
    traf_u = float(G.nodes[u]['trafico'])
    traf_v = float(G.nodes[v]['trafico'])
    capacidad = float(G.edges[u, v].get('capacidad', 1000))
    
    # Mayor sensibilidad en enlaces de menor capacidad
    factor_capacidad = 1.4 if capacidad == 100 else 1.1
    
    vecinos_u = [k for k in G.neighbors(u) if k != v]
    vecinos_v = [j for j in G.neighbors(v) if j != u]
    
    traf_agregado = sum(float(G.nodes[k]['trafico']) for k in vecinos_u + vecinos_v)
    
    # Carga base escalada + aporte de congestión vecina
    carga_base = ((traf_u + traf_v) / 2.0) * factor_capacidad
    carga_total = carga_base + 0.08 * traf_agregado
    
    return round(carga_total, 2)

# --- PANEL LATERAL ---
st.sidebar.header("🕹️ Controles de Red")

if st.button("🔄 Regenerar Red Aleatoria", use_container_width=True):
        st.session_state.red = generar_red_computadores()
        st.session_state.nodos_bloqueados = set(random.sample(list(st.session_state.red.nodes()), 2))
        st.session_state.conexiones_bloqueadas = set()
        st.session_state.anim_id += 1
        
        # Elimina las claves cacheadas de los sliders para que lean los nuevos valores aleatorios
        for k in list(st.session_state.keys()):
            if "trafico" in k or "cpu" in k or "slider" in k:
                del st.session_state[k]
                
        st.rerun()

st.sidebar.divider()

st.sidebar.subheader("🎛️ Ajuste Dinámico de Nodo")
nodo_sel = st.sidebar.selectbox("Seleccionar Nodo:", list(st.session_state.red.nodes()))

if nodo_sel is not None:
    attr_sel = st.session_state.red.nodes[nodo_sel]
    st.sidebar.caption(f"Tipo: **{attr_sel['tipo']}**")
    nuevo_trafico = st.sidebar.slider("Tráfico:", 0.0, 1.0, float(attr_sel['trafico']), 0.05, key=f"traf_{nodo_sel}")
    nueva_cpu = st.sidebar.slider("CPU:", 0.0, 1.0, float(attr_sel['cpu']), 0.05, key=f"cpu_{nodo_sel}")
    st.session_state.red.nodes[nodo_sel]['trafico'] = nuevo_trafico
    st.session_state.red.nodes[nodo_sel]['cpu'] = nueva_cpu

st.sidebar.divider()

st.sidebar.subheader("🔒 Aislamiento & Bloqueo")
predicciones = evaluador.predecir_anomalias(st.session_state.red)
nodos_anomalos_detectados = [n for n, est in predicciones.items() if est == "Anómalo"]

nodo_bloquear_sel = st.sidebar.selectbox("Seleccionar Nodo a Bloquear/Aislar:", list(st.session_state.red.nodes()), key="sel_nodo_block")
if nodo_bloquear_sel is not None:
    es_bloqueado_actual = nodo_bloquear_sel in st.session_state.nodos_bloqueados
    btn_label = "🔓 Desbloquear Nodo" if es_bloqueado_actual else "🔒 Bloquear/Aislar Nodo"
    if st.sidebar.button(btn_label):
        if es_bloqueado_actual:
            st.session_state.nodos_bloqueados.remove(nodo_bloquear_sel)
        else:
            st.session_state.nodos_bloqueados.add(nodo_bloquear_sel)
        st.session_state.anim_id += 1
        st.rerun()

st.sidebar.caption("--- Acciones Masivas ---")
col_b1, col_b2 = st.sidebar.columns(2)
if col_b1.button("Bloquear Anómalos"):
    for n in nodos_anomalos_detectados:
        st.session_state.nodos_bloqueados.add(n)
    st.session_state.anim_id += 1
    st.rerun()

if col_b2.button("Restablecer Todo"):
    st.session_state.nodos_bloqueados = set()
    st.session_state.conexiones_bloqueadas = set()
    st.session_state.anim_id += 1
    st.rerun()

st.sidebar.divider()

lista_aristas = [f"{u} <-> {v}" for u, v in st.session_state.red.edges()]
if lista_aristas:
    arista_sel = st.sidebar.selectbox("Conexión a Cortar/Saturar:", lista_aristas)
    if st.sidebar.button("Bloquear/Cortar Enlace"):
        u_str, v_str = arista_sel.split(" <-> ")
        st.session_state.conexiones_bloqueadas.add((int(u_str), int(v_str)))
        st.session_state.conexiones_bloqueadas.add((int(v_str), int(u_str)))
        st.session_state.anim_id += 1
        st.rerun()

# --- MÉTRICAS (Lógica de Conjuntos Excluyentes) ---
nodos_bloqueados_set = st.session_state.nodos_bloqueados
num_bloqueados = len(nodos_bloqueados_set)

anomalias_activas = [n for n in nodos_anomalos_detectados if n not in nodos_bloqueados_set]
num_anomalias_activas = len(anomalias_activas)

nodos_normales = [
    n for n in st.session_state.red.nodes() 
    if n not in nodos_bloqueados_set and n not in nodos_anomalos_detectados
]
num_normales = len(nodos_normales)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Nodos 🖥️", st.session_state.red.number_of_nodes())
col2.metric("Nodos Normales 🟢", num_normales)
col3.metric("Anomalías GNN 🔴", num_anomalias_activas)
col4.metric("Nodos Bloqueados 🔒", num_bloqueados)

st.divider()

# --- SEPARACIÓN POR PESTAÑAS ---
tab_red, tab_automata = st.tabs(["🌐 1. Simulador de Tráfico de Red (Ping / PDU)", "🔤 2. Laboratorio del Autómata (AFD)"])

# ==============================================================================
# PESTAÑA 1: SIMULADOR DE RED
# ==============================================================================
with tab_red:
    col_left, col_right = st.columns([2, 1])

    ruta_renderizada = []
    exito_transmision = False
    mensaje_sim = ""
    obstaculo_tipo = "ninguno"
    nodo_destino_enlace = None

    with col_right:
        st.subheader("📡 Inyección y Simulación de Tráfico")
        st.caption("Prueba el envío de un paquete PDU de datos entre dos nodos evaluando la infraestructura física.")
        
        col_o, col_d = st.columns(2)
        origen = col_o.selectbox("Origen 📍:", list(st.session_state.red.nodes()), index=0, key="ping_origen")
        destino = col_d.selectbox("Destino 🎯:", list(st.session_state.red.nodes()), index=min(3, len(st.session_state.red)-1), key="ping_destino")
        
        cadena_predeterminada = "PWSR"
        automata_net = AutomataComunicacion()
        
        if st.button("🚀 Transmitir Paquete", use_container_width=True):
            st.session_state.anim_id += 1
            
            # 1. EVALUAR ESTADO DEL ORIGEN Y DESTINO
            origen_bloqueado = origen in st.session_state.nodos_bloqueados
            origen_anomalo = (predicciones[origen] == "Anómalo")
            destino_bloqueado = destino in st.session_state.nodos_bloqueados
            destino_anomalo = (predicciones[destino] == "Anómalo")

            # 2. CONSTRUIR SUBGRAFO LIMPIO (Umbral estricto < 1.00)
            G_limpio = nx.Graph()
            for u, attr in st.session_state.red.nodes(data=True):
                es_malo = (predicciones[u] == "Anómalo") or (u in st.session_state.nodos_bloqueados)
                if not es_malo:
                    G_limpio.add_node(u, **attr)

            for u, v, attr in st.session_state.red.edges(data=True):
                if u in G_limpio and v in G_limpio:
                    conexion_cortada = (u, v) in st.session_state.conexiones_bloqueadas or (v, u) in st.session_state.conexiones_bloqueadas
                    carga_enlace = calcular_carga_acumulada_arista(st.session_state.red, u, v)
                    if not conexion_cortada and carga_enlace < 1.00:
                        G_limpio.add_edge(u, v, weight=attr['peso'] * (1.0 + carga_enlace))

            # 3. VERIFICAR SI EXISTE CAMINO LIMPIO
            hay_ruta_exitosa = (
                not origen_bloqueado and not origen_anomalo and
                not destino_bloqueado and not destino_anomalo and
                origen in G_limpio and destino in G_limpio and
                nx.has_path(G_limpio, origen, destino)
            )

            obstaculo_tipo = "ninguno"
            nodo_destino_enlace = None

            if hay_ruta_exitosa:
                ruta_renderizada = nx.shortest_path(G_limpio, origen, destino, weight='weight')
                exito_transmision = True
                mensaje_sim = f"✅ Transmisión Exitosa (ACK) vía {' -> '.join(map(str, ruta_renderizada))}"
            else:
                exito_transmision = False

                if origen_bloqueado or origen_anomalo:
                    causa_obstaculo = "Nodo de Origen Bloqueado 🔒" if origen_bloqueado else "Nodo de Origen Anómalo 🔴"
                    ruta_renderizada = [origen]
                    obstaculo_tipo = "origen"
                    mensaje_sim = f"⚠️ Transmisión Abortada: {causa_obstaculo} N{origen}. No puede emitir paquetes."
                elif nx.has_path(st.session_state.red, origen, destino):
                    ruta_topologica = nx.shortest_path(st.session_state.red, origen, destino)
                    ruta_alcanzada = [origen]
                    causa_obstaculo = "Falla de red"

                    for idx in range(1, len(ruta_topologica)):
                        nodo_prev = ruta_topologica[idx - 1]
                        nodo_act = ruta_topologica[idx]

                        conexion_cortada = (nodo_prev, nodo_act) in st.session_state.conexiones_bloqueadas or (nodo_act, nodo_prev) in st.session_state.conexiones_bloqueadas
                        carga_enlace = calcular_carga_acumulada_arista(st.session_state.red, nodo_prev, nodo_act)
                        es_bloqueado = nodo_act in st.session_state.nodos_bloqueados
                        es_anomalo = (predicciones[nodo_act] == "Anómalo")

                        # PRIORIDAD 1: EL CANAL FÍSICO (El paquete no puede pasar si el enlace falla)
                        if conexion_cortada:
                            causa_obstaculo = f"Enlace Cortado ✂️ entre N{nodo_prev} y N{nodo_act}"
                            obstaculo_tipo = "enlace"
                            nodo_destino_enlace = nodo_act
                            break
                        elif carga_enlace >= 1.00:
                            causa_obstaculo = f"Enlace Saturado ⚠️ ({int(carga_enlace*100)}%) entre N{nodo_prev} y N{nodo_act}"
                            obstaculo_tipo = "enlace"
                            nodo_destino_enlace = nodo_act
                            break

                        # PRIORIDAD 2: SI EL ENLACE ESTÁ SANO, EL PAQUETE ALCANZA EL NODO
                        ruta_alcanzada.append(nodo_act)

                        # PRIORIDAD 3: EVALUAR EL NODO ALCANZADO
                        if es_bloqueado:
                            causa_obstaculo = f"Nodo Bloqueado 🔒 N{nodo_act}"
                            obstaculo_tipo = "nodo"
                            break
                        elif es_anomalo:
                            causa_obstaculo = f"Nodo Anómalo 🔴 N{nodo_act}"
                            obstaculo_tipo = "nodo"
                            break

                    ruta_renderizada = ruta_alcanzada
                    mensaje_sim = f"⚠️ Paquete Obstruido: {causa_obstaculo}. Rebotando a Origen."
                else:
                    ruta_renderizada = [origen]
                    mensaje_sim = "❌ Sin ruta física conectada entre Origen y Destino."

            if exito_transmision:
                st.success(mensaje_sim)
            else:
                st.error(mensaje_sim)

        st.divider()
        st.write("📊 **Estado Instantáneo de Dispositivos:**")
        for n in st.session_state.red.nodes():
            attr = st.session_state.red.nodes[n]
            est = predicciones[n]
            esta_bloq = n in st.session_state.nodos_bloqueados
            badge = "🔒" if esta_bloq else ("🔴" if est == "Anómalo" else "🟢")
            st.write(f"{badge} **N{n} ({attr['tipo']})** | Traf: `{attr['trafico']:.2f}` | CPU: `{attr['cpu']:.2f}`")

    with col_left:
        st.subheader("🕸️ Topología Interactiva de Red")
        
        net = Network(height="530px", width="100%", bgcolor="#111827", font_color="white", directed=False)
        net.toggle_physics(False)
        net.set_options("""
        {
          "interaction": {
            "dragNodes": true,
            "dragView": true,
            "zoomView": true,
            "hover": true
          }
        }
        """)

        for nodo in st.session_state.red.nodes():
            attr = st.session_state.red.nodes[nodo]
            estado_gnn = predicciones[nodo]
            esta_bloqueado = nodo in st.session_state.nodos_bloqueados
            
            color = "#6B7280" if esta_bloqueado else ("#EF4444" if estado_gnn == "Anómalo" else "#10B981")

            net.add_node(
                nodo, 
                label=f"N{nodo} ({attr['tipo']})", 
                color=color, 
                size=22,
                x=attr.get('x'),
                y=attr.get('y'),
                physics=False
            )

        for source, target, attr_e in st.session_state.red.edges(data=True):
            conexion_cortada = (source, target) in st.session_state.conexiones_bloqueadas or (target, source) in st.session_state.conexiones_bloqueadas
            
            if not conexion_cortada:
                carga_acumulada = calcular_carga_acumulada_arista(st.session_state.red, source, target)
                peso_val = attr_e.get('peso', 1.0)
                capacidad_val = attr_e.get('capacidad', 100)
                color_edge = "#EF4444" if carga_acumulada >= 1.00 else ("#F59E0B" if carga_acumulada >= 0.75 else "#4B5563")

                # Texto plano limpio con saltos de línea nativos
                info_tooltip = (
                    f"Enlace N{source} ⟷ N{target}\n"
                    f"• Capacidad: {capacidad_val} Mbps\n"
                    f"• Latencia base: {peso_val:.1f} ms\n"
                    f"• Carga acumulada: {carga_acumulada:.2f} ({int(carga_acumulada*100)}%)"
                )

                net.add_edge(
                    source, 
                    target, 
                    # Se elimina 'value=peso_val' para evitar que Vis.js agrande la fuente
                    label=f"{peso_val:.1f}ms", 
                    title=info_tooltip,
                    width=2, 
                    color=color_edge,
                    font={
                        'size': 10, 
                        'color': '#E5E7EB', 
                        'background': '#111827', 
                        'strokeWidth': 0,
                        'align': 'horizontal'
                    }
                )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmpfile:
            net.save_graph(tmpfile.name)
            with open(tmpfile.name, 'r', encoding='utf-8') as f:
                html_content = f.read()

        js_animacion = f"""
        <!-- ID de Animación Dinámico: {st.session_state.anim_id} -->
        <style>
            div.vis-tooltip {{
                position: absolute;
                background-color: #1f2937 !important;
                color: #f9fafb !important;
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
                font-size: 12px !important;
                line-height: 1.5 !important;
                padding: 8px 12px !important;
                border: 1px solid #374151 !important;
                border-radius: 8px !important;
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.6) !important;
                white-space: pre-line !important;
                z-index: 99999 !important;
                pointer-events: none !important;
            }}
        </style>
        <script>
        (function() {{
            var initialPositions = {{}};

            function configurarRetornoYAnimacion() {{
                if (typeof network === 'undefined' || !network || typeof nodes === 'undefined' || !nodes) {{
                    setTimeout(configurarRetornoYAnimacion, 150);
                    return;
                }}

                var canvasPositions = network.getPositions();
                for (var id in canvasPositions) {{
                    initialPositions[id] = {{ x: canvasPositions[id].x, y: canvasPositions[id].y }};
                }}

                network.on("dragEnd", function(params) {{
                    if (params.nodes && params.nodes.length > 0) {{
                        var draggedId = params.nodes[0];
                        var target = initialPositions[draggedId];
                        if (target) {{
                            var current = network.getPositions([draggedId])[draggedId];
                            if (current) {{
                                var step = 0;
                                var maxSteps = 12;
                                var startX = current.x;
                                var startY = current.y;
                                var timer = setInterval(function() {{
                                    step++;
                                    var p = step / maxSteps;
                                    nodes.update({{
                                        id: draggedId,
                                        x: startX + (target.x - startX) * p,
                                        y: startY + (target.y - startY) * p
                                    }});
                                    network.redraw();

                                    if (step >= maxSteps) {{
                                        clearInterval(timer);
                                        nodes.update({{ id: draggedId, x: target.x, y: target.y }});
                                        network.redraw();
                                    }}
                                }}, 16);
                            }}
                        }}
                    }}
                }});

                var rutaCorta = {json.dumps(ruta_renderizada)};
                var esExito = {json.dumps(exito_transmision)};
                var obstaculoTipo = {json.dumps(obstaculo_tipo)};
                var nodoDestinoEnlace = {json.dumps(nodo_destino_enlace)};

                if (!rutaCorta || (rutaCorta.length < 2 && obstaculoTipo !== "enlace")) return;

                var steps = [];

                if (esExito === true) {{
                    for (var i = 0; i < rutaCorta.length - 1; i++) {{
                        steps.push({{ from: rutaCorta[i], to: rutaCorta[i+1], startP: 0.0, endP: 1.0, color: '#00F0FF' }});
                    }}
                    for (var i = rutaCorta.length - 1; i > 0; i--) {{
                        steps.push({{ from: rutaCorta[i], to: rutaCorta[i-1], startP: 0.0, endP: 1.0, color: '#32FA99' }});
                    }}
                }} else if (obstaculoTipo === "nodo") {{
                    for (var i = 0; i < rutaCorta.length - 1; i++) {{
                        steps.push({{ from: rutaCorta[i], to: rutaCorta[i+1], startP: 0.0, endP: 1.0, color: '#00F0FF' }});
                    }}
                    for (var i = rutaCorta.length - 1; i > 0; i--) {{
                        steps.push({{ from: rutaCorta[i], to: rutaCorta[i-1], startP: 0.0, endP: 1.0, color: '#FF0055' }});
                    }}
                }} else if (obstaculoTipo === "enlace" && nodoDestinoEnlace !== null) {{
                    var prevNode = rutaCorta[rutaCorta.length - 1];
                    for (var i = 0; i < rutaCorta.length - 1; i++) {{
                        steps.push({{ from: rutaCorta[i], to: rutaCorta[i+1], startP: 0.0, endP: 1.0, color: '#00F0FF' }});
                    }}
                    steps.push({{ from: prevNode, to: nodoDestinoEnlace, startP: 0.0, endP: 0.22, color: '#00F0FF' }});
                    steps.push({{ from: prevNode, to: nodoDestinoEnlace, startP: 0.22, endP: 0.0, color: '#FF0055' }});
                    for (var i = rutaCorta.length - 1; i > 0; i--) {{
                        steps.push({{ from: rutaCorta[i], to: rutaCorta[i-1], startP: 0.0, endP: 1.0, color: '#FF0055' }});
                    }}
                }}

                if (steps.length === 0) return;

                var currentStepIndex = 0;
                var t = 0.0;
                var animacionFinalizada = false;
                var lastTimestamp = performance.now();
                var animFramePending = false;

                network.on("afterDrawing", function(ctx) {{
                    if (currentStepIndex < steps.length) {{
                        var now = performance.now();
                        var dt = (now - lastTimestamp) / 1000;
                        lastTimestamp = now;

                        if (dt > 0.05) dt = 0.05;

                        var step = steps[currentStepIndex];
                        var posA = network.getPositions([step.from])[step.from];
                        var posB = network.getPositions([step.to])[step.to];

                        if (posA && posB) {{
                            var p = step.startP + (step.endP - step.startP) * t;
                            var posX = posA.x + (posB.x - posA.x) * p;
                            var posY = posA.y + (posB.y - posA.y) * p;

                            ctx.save();
                            ctx.beginPath();
                            ctx.arc(posX, posY, 9, 0, 2 * Math.PI, false);
                            ctx.fillStyle = step.color;
                            ctx.shadowColor = step.color;
                            ctx.shadowBlur = 18;
                            ctx.fill();
                            ctx.strokeStyle = '#FFFFFF';
                            ctx.lineWidth = 2;
                            ctx.stroke();
                            ctx.restore();

                            var dist = Math.abs(step.endP - step.startP);
                            var duration = dist < 0.5 ? 0.45 : 1.10;
                            t += dt / duration;

                            if (t >= 1.0) {{
                                t = 0.0;
                                currentStepIndex++;
                            }}

                            if (!animFramePending && currentStepIndex < steps.length) {{
                                animFramePending = true;
                                requestAnimationFrame(function() {{
                                    animFramePending = false;
                                    network.redraw();
                                }});
                            }}
                        }}
                    }} else if (!animacionFinalizada) {{
                        animacionFinalizada = true;
                        network.setOptions({{ physics: {{ enabled: true }} }});
                        network.redraw();
                    }}
                }});

                network.redraw();
            }}

            setTimeout(configurarRetornoYAnimacion, 400);
        }})();
        </script>
        """
        
        html_animado = html_content.replace("</body>", js_animacion)
        components.html(html_animado, height=550)


# ==============================================================================
# PESTAÑA 2: LABORATORIO AUTÓNOMO DEL AUTÓMATA (AFD)
# ==============================================================================
with tab_automata:
    st.header("🔤 Laboratorio Formal del Autómata (AFD)")
    st.caption("Validador interactivo con animación reactiva letra a letra en tiempo real.")

    col_afd_left, col_afd_right = st.columns([1, 1])

    with col_afd_left:
        st.subheader("📌 Especificación Formal $M = (Q, \\Sigma, \\delta, q_0, F)$")
        st.markdown("""
        * **$\\Sigma$ (Alfabeto):** `{R, P, W, S, E}`
        * **$Q$ (Estados):** `{q0, q1, q2, qf, qe}`
        * **$q_0$ (Inicial):** `q0` (Reposo)
        * **$F$ (Finales):** `{q0, qf, qe}`
        """)

        st.subheader("🔀 Expresión Regular ($ER$)")
        st.latex(r"ER = \Big( R \;\big|\; P\,W\,S \;\big|\; P\,E \;\big|\; P\,W\,E \Big)^*")

        st.subheader("📑 Tabla de Transición $\\delta(q, \\sigma)$")
        st.markdown("""
        | Estado ($q$) | $P$ | $W$ | $S$ | $E$ | $R$ |
        | :---: | :---: | :---: | :---: | :---: | :---: |
        | **$q_0$** *(Final)* | $q_1$ | - | - | - | $q_0$ |
        | **$q_1$** | - | $q_2$ | - | $q_e$ | - |
        | **$q_2$** | - | - | $q_f$ | $q_e$ | - |
        | **$q_f$** *(Final)* | $q_1$ | - | - | - | $q_0$ |
        | **$q_e$** *(Final)* | $q_1$ | - | - | - | $q_0$ |
        """)

    with col_afd_right:
        st.subheader("⚡ Grafo Reactivo e Interactivo")
        st.caption("Escribe directamente en la caja para ver el flujo animado en tiempo real sin presionar Enter.")

        html_automata_reactivo = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { background-color: #111827; color: #ffffff; font-family: system-ui, sans-serif; margin:0; padding:10px; }
                .input-box {
                    width: 100%; padding: 12px; font-size: 18px; font-weight: bold;
                    background: #1F2937; border: 2px solid #3B82F6; border-radius: 8px;
                    color: #00F0FF; text-align: center; outline: none; box-sizing: border-box;
                    margin-bottom: 12px; text-transform: uppercase; letter-spacing: 2px;
                }
                .status-badge {
                    padding: 10px; border-radius: 6px; font-weight: bold; text-align: center;
                    font-size: 16px; margin-bottom: 12px; transition: all 0.3s;
                }
                .badge-ok { background-color: #065F46; color: #34D399; border: 1px solid #059669; }
                .badge-fail { background-color: #991B1B; color: #FCA5A5; border: 1px solid #DC2626; }
                svg { width: 100%; height: 350px; background: #0b0f19; border-radius: 8px; border: 1px solid #374151; }
                
                .node { fill: #1F2937; stroke: #3B82F6; stroke-width: 2px; transition: all 0.3s ease; }
                .node-final { stroke: #10B981; stroke-width: 3px; stroke-dasharray: 4 2; }
                .node.visited { fill: #D97706; stroke: #F59E0B; }
                .node.active-ok { fill: #10B981; stroke: #34D399; filter: drop-shadow(0 0 8px #34D399); }
                .node.active-fail { fill: #EF4444; stroke: #FCA5A5; filter: drop-shadow(0 0 8px #EF4444); }
                
                text { fill: #FFFFFF; font-family: monospace; font-size: 13px; font-weight: bold; pointer-events: none; }
                .edge { stroke: #4B5563; stroke-width: 2px; transition: stroke 0.3s; }
                .edge.active { stroke: #00F0FF; stroke-width: 3px; filter: drop-shadow(0 0 5px #00F0FF); }
                .edge-label { fill: #9CA3AF; font-size: 11px; }
            </style>
        </head>
        <body>
            <input type="text" id="tape" class="input-box" value="PWSR" placeholder="Escribe P, W, S, E, R..." autocomplete="off">
            <div id="status" class="status-badge badge-ok">🟢 Pertenece al autómata</div>

            <svg viewBox="0 0 600 300">
                <defs>
                    <marker id="arrow" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                        <path d="M 0 0 L 10 5 L 0 10 z" fill="#6B7280"/>
                    </marker>
                    <marker id="arrow-active" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                        <path d="M 0 0 L 10 5 L 0 10 z" fill="#00F0FF"/>
                    </marker>
                </defs>

                <!-- ARISTAS (Transiciones) -->
                <line x1="80" y1="150" x2="200" y2="70" class="edge" id="e-q0-q1" marker-end="url(#arrow)"/>
                <path d="M 80 130 A 20 20 0 1 1 80 170" fill="none" class="edge" id="e-q0-q0" marker-end="url(#arrow)"/>
                <line x1="200" y1="70" x2="340" y2="70" class="edge" id="e-q1-q2" marker-end="url(#arrow)"/>
                <line x1="200" y1="70" x2="480" y2="230" class="edge" id="e-q1-qe" marker-end="url(#arrow)"/>
                <line x1="340" y1="70" x2="480" y2="150" class="edge" id="e-q2-qf" marker-end="url(#arrow)"/>
                <line x1="340" y1="70" x2="480" y2="230" class="edge" id="e-q2-qe" marker-end="url(#arrow)"/>
                <line x1="480" y1="150" x2="80" y2="150" class="edge" id="e-qf-q0" marker-end="url(#arrow)"/>
                <path d="M 480 150 Q 300 10 200 70" fill="none" class="edge" id="e-qf-q1" marker-end="url(#arrow)"/>
                <line x1="480" y1="230" x2="80" y2="150" class="edge" id="e-qe-q0" marker-end="url(#arrow)"/>
                <path d="M 480 230 Q 250 250 200 70" fill="none" class="edge" id="e-qe-q1" marker-end="url(#arrow)"/>

                <!-- ETIQUETAS DE ARISTAS -->
                <text x="130" y="100" class="edge-label">P</text>
                <text x="40" y="150" class="edge-label">R</text>
                <text x="270" y="60" class="edge-label">W</text>
                <text x="310" y="170" class="edge-label">E</text>
                <text x="400" y="100" class="edge-label">S</text>
                <text x="420" y="140" class="edge-label">E</text>
                <text x="280" y="140" class="edge-label">R</text>
                <text x="320" y="30" class="edge-label">P</text>
                <text x="280" y="210" class="edge-label">R</text>
                <text x="330" y="240" class="edge-label">P</text>

                <!-- NODOS -->
                <g id="n-q0"><circle cx="80" cy="150" r="24" class="node node-final"/><text x="80" y="154" text-anchor="middle">q0</text></g>
                <g id="n-q1"><circle cx="200" cy="70" r="24" class="node"/><text x="200" y="74" text-anchor="middle">q1</text></g>
                <g id="n-q2"><circle cx="340" cy="70" r="24" class="node"/><text x="340" y="74" text-anchor="middle">q2</text></g>
                <g id="n-qf"><circle cx="480" cy="150" r="24" class="node node-final"/><text x="480" y="154" text-anchor="middle">qf</text></g>
                <g id="n-qe"><circle cx="480" cy="230" r="24" class="node node-final"/><text x="480" y="234" text-anchor="middle">qe</text></g>
            </svg>

            <script>
                const afd = {
                    initial: 'q0',
                    finals: new Set(['q0', 'qf', 'qe']),
                    transitions: {
                        'q0': { 'P': 'q1', 'R': 'q0' },
                        'q1': { 'W': 'q2', 'E': 'qe' },
                        'q2': { 'S': 'qf', 'E': 'qe' },
                        'qf': { 'R': 'q0', 'P': 'q1' },
                        'qe': { 'R': 'q0', 'P': 'q1' }
                    }
                };

                const tapeInput = document.getElementById('tape');
                const statusDiv = document.getElementById('status');

                function updateAutomaton() {
                    const inputStr = tapeInput.value.toUpperCase().trim();
                    
                    document.querySelectorAll('circle').forEach(c => {
                        c.classList.remove('visited', 'active-ok', 'active-fail');
                    });
                    document.querySelectorAll('.edge').forEach(e => {
                        e.classList.remove('active');
                        e.setAttribute('marker-end', 'url(#arrow)');
                    });

                    let currentState = afd.initial;
                    let trace = [currentState];
                    let isValid = true;
                    let activeEdges = [];

                    for (let sim of inputStr) {
                        if (afd.transitions[currentState] && afd.transitions[currentState][sim]) {
                            let nextState = afd.transitions[currentState][sim];
                            activeEdges.push(`e-${currentState}-${nextState}`);
                            currentState = nextState;
                            trace.push(currentState);
                        } else {
                            isValid = false;
                            break;
                        }
                    }

                    if (isValid) {
                        isValid = afd.finals.has(currentState);
                    }

                    trace.forEach((st, idx) => {
                        const circle = document.querySelector(`#n-${st} circle`);
                        if (circle) {
                            if (idx === trace.length - 1) {
                                circle.classList.add(isValid ? 'active-ok' : 'active-fail');
                            } else {
                                circle.classList.add('visited');
                            }
                        }
                    });

                    activeEdges.forEach(eId => {
                        const edgeEl = document.getElementById(eId);
                        if (edgeEl) {
                            edgeEl.classList.add('active');
                            edgeEl.setAttribute('marker-end', 'url(#arrow-active)');
                        }
                    });

                    if (isValid) {
                        statusDiv.className = "status-badge badge-ok";
                        statusDiv.innerHTML = `🟢 Pertenece al autómata <br><small style="font-weight:normal;">Estado Final: <b>${currentState}</b> | Traza: ${trace.join(' → ')}</small>`;
                    } else {
                        statusDiv.className = "status-badge badge-fail";
                        statusDiv.innerHTML = `🔴 No pertenece al autómata <br><small style="font-weight:normal;">Rechazado en: <b>${currentState}</b> | Traza: ${trace.join(' → ')}</small>`;
                    }
                }

                tapeInput.addEventListener('input', updateAutomaton);
                updateAutomaton();
            </script>
        </body>
        </html>
        """

        components.html(html_automata_reactivo, height=520)

        components.html(html_automata_reactivo, height=520)