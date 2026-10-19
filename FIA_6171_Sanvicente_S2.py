"""
Actividad Parcial 2 - Sección 2: Problemas de búsqueda y espacios de búsqueda
Alumno: Fernando Yair Sanvicente Valencia | Grupo 171
Asignatura: Fundamentos de Inteligencia Artificial (UP Atlautla)

Se modelan DOS problemas como problemas de búsqueda
(estado inicial, acciones, modelo de transición, costo, prueba de meta):
  1) Problema de las jarras (4 L y 3 L, meta: 2 L en la jarra grande)
     -> grafo dirigido y PONDERADO (costo = litros movidos)
  2) Misioneros y caníbales (3 y 3, bote de capacidad 2)
     -> grafo dirigido con costo uniforme (1 por cruce)
Cada espacio de estados se genera con un generador (yield) y se
representa con NetworkX (DiGraph).
"""
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================================================================
# PROBLEMA 1: JARRAS DE AGUA
# Estado = (a, b): litros en la jarra A (cap. 4) y en la jarra B (cap. 3)
# ======================================================================
CAP_A, CAP_B = 4, 3
INICIO_J = (0, 0)

def meta_jarras(s):
    """Prueba de meta: la jarra A contiene exactamente 2 litros."""
    return s[0] == 2

def sucesores_jarras(s):
    """Generador de estados: produce (accion, nuevo_estado, costo).
    El costo es la cantidad de litros movidos (llenar/vaciar/verter)."""
    a, b = s
    # Llenar A o B hasta el tope (costo = litros agregados)
    if a < CAP_A: yield ("Llenar A", (CAP_A, b), CAP_A - a)
    if b < CAP_B: yield ("Llenar B", (a, CAP_B), CAP_B - b)
    # Vaciar A o B (costo = litros desechados)
    if a > 0: yield ("Vaciar A", (0, b), a)
    if b > 0: yield ("Vaciar B", (a, 0), b)
    # Verter A -> B
    t = min(a, CAP_B - b)
    if t > 0: yield ("Verter A->B", (a - t, b + t), t)
    # Verter B -> A
    t = min(b, CAP_A - a)
    if t > 0: yield ("Verter B->A", (a + t, b - t), t)

# ======================================================================
# PROBLEMA 2: MISIONEROS Y CANÍBALES
# Estado = (m, c, lado): misioneros y caníbales en la orilla IZQUIERDA;
# lado = 0 si el bote está a la izquierda, 1 si está a la derecha.
# ======================================================================
N = 3
INICIO_M = (N, N, 0)
META_M = (0, 0, 1)

def valido_mc(m, c):
    """Un estado es válido si en ninguna orilla hay más caníbales que
    misioneros (cuando hay al menos un misionero en esa orilla)."""
    if not (0 <= m <= N and 0 <= c <= N):
        return False
    mr, cr = N - m, N - c
    if m > 0 and c > m: return False
    if mr > 0 and cr > mr: return False
    return True

def sucesores_mc(s):
    """Generador de estados: el bote lleva 1 o 2 personas; costo = 1."""
    m, c, lado = s
    for dm, dc in [(1,0),(2,0),(0,1),(0,2),(1,1)]:
        if lado == 0:   # el bote sale de la izquierda
            nm, nc = m - dm, c - dc
        else:           # el bote regresa de la derecha
            nm, nc = m + dm, c + dc
        if valido_mc(nm, nc):
            yield (f"Cruzan {dm}M {dc}C", (nm, nc, 1 - lado), 1)

# ======================================================================
# Construcción del grafo (exploración completa del espacio de estados)
# ======================================================================
def construir_grafo(inicio, sucesores):
    G = nx.DiGraph()
    G.add_node(inicio)
    visitados, frontera = {inicio}, [inicio]
    while frontera:
        s = frontera.pop(0)
        for accion, nuevo, costo in sucesores(s):
            G.add_edge(s, nuevo, accion=accion, weight=costo)
            if nuevo not in visitados:
                visitados.add(nuevo)
                frontera.append(nuevo)
    return G

def resolver(G, inicio, es_meta):
    """Camino de menor costo con Dijkstra (BFS si todos los costos son 1)."""
    metas = [n for n in G.nodes if es_meta(n)]
    mejor = min(((nx.dijkstra_path_length(G, inicio, m), m) for m in metas
                 if nx.has_path(G, inicio, m)), default=None)
    if mejor is None:
        return None, None
    costo, meta = mejor
    return nx.dijkstra_path(G, inicio, meta), costo

def dibujar(G, camino, inicio, es_meta, titulo, archivo, etiquetas_peso=True, size=(11, 8), layout="kk"):
    pos = nx.kamada_kawai_layout(G) if layout == "kk" else nx.spring_layout(G, seed=7, k=1.2)
    en_camino = set(zip(camino, camino[1:])) if camino else set()
    colores = []
    for n in G.nodes:
        colores.append("#2E86C1" if n == inicio else "#27AE60" if es_meta(n) else "#F5B041" if n in camino else "#D5DBDB")
    ancho = [3.0 if e in en_camino else 0.8 for e in G.edges]
    ecol = ["#C0392B" if e in en_camino else "#7F8C8D" for e in G.edges]
    plt.figure(figsize=size)
    nx.draw_networkx_nodes(G, pos, node_color=colores, node_size=1100, edgecolors="black")
    nx.draw_networkx_labels(G, pos, labels={n: str(n).replace(" ", "") for n in G.nodes}, font_size=7)
    nx.draw_networkx_edges(G, pos, width=ancho, edge_color=ecol, arrows=True, arrowsize=14,
                           connectionstyle="arc3,rad=0.12", node_size=1100)
    if etiquetas_peso:
        pesos = {e: G.edges[e]["weight"] for e in G.edges if e in en_camino}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=pesos, font_size=8, font_color="#C0392B",
                                     label_pos=0.5, bbox=dict(alpha=0.0))
    plt.title(titulo, fontsize=12, fontweight="bold")
    plt.axis("off"); plt.tight_layout()
    plt.savefig(archivo, dpi=170); plt.close()

def reporte(nombre, G, inicio, es_meta, camino, costo):
    print("=" * 62); print(nombre); print("=" * 62)
    print(f"Estados alcanzables |V| = {G.number_of_nodes()}")
    print(f"Transiciones         |E| = {G.number_of_edges()}")
    print(f"Factor de ramificación promedio b = {G.number_of_edges()/G.number_of_nodes():.2f}")
    print(f"Factor de ramificación máximo     = {max(d for _, d in G.out_degree())}")
    if camino:
        print(f"Solución óptima: {len(camino)-1} acciones, costo total = {costo}")
        for u, v in zip(camino, camino[1:]):
            print(f"   {u} --[{G.edges[u, v]['accion']} | costo {G.edges[u, v]['weight']}]--> {v}")
    print()

if __name__ == "__main__":
    # ---- Problema 1 ----
    G1 = construir_grafo(INICIO_J, sucesores_jarras)
    c1, k1 = resolver(G1, INICIO_J, meta_jarras)
    reporte("PROBLEMA 1: JARRAS (4 L y 3 L, meta 2 L en A)", G1, INICIO_J, meta_jarras, c1, k1)
    dibujar(G1, c1, INICIO_J, meta_jarras, "Grafo del problema de las jarras (dirigido y ponderado)", "grafo_jarras.png")

    # ---- Problema 2 ----
    G2 = construir_grafo(INICIO_M, sucesores_mc)
    c2, k2 = resolver(G2, INICIO_M, lambda s: s == META_M)
    reporte("PROBLEMA 2: MISIONEROS Y CANIBALES (3 y 3)", G2, INICIO_M, lambda s: s == META_M, c2, k2)
    dibujar(G2, c2, INICIO_M, lambda s: s == META_M, "Grafo de misioneros y caníbales (dirigido, costo 1 por cruce)",
            "grafo_misioneros.png", etiquetas_peso=False, size=(10, 7))

    # Cantidad total de estados posibles vs. alcanzables (para el análisis)
    print("Jarras: estados posibles (CAP_A+1)*(CAP_B+1) =", (CAP_A + 1) * (CAP_B + 1))
    posibles_mc = sum(1 for m in range(N+1) for c in range(N+1) for l in (0, 1) if valido_mc(m, c))
    print("M&C: estados válidos (incluyendo ambos lados del bote) =", posibles_mc)
