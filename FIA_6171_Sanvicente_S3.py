"""
Actividad Parcial 2 - Sección 3: Búsqueda local y optimización
Alumno: Fernando Yair Sanvicente Valencia | Grupo 171
Asignatura: Fundamentos de Inteligencia Artificial (UP Atlautla)

Problema: ruta mínima de reparto (TSP simétrico) con 30 puntos.
Se implementan y comparan:
  - Hill Climbing (ascenso de colina, aceptación solo de mejoras)
  - Simulated Annealing (recocido simulado, enfriamiento geométrico)
Ambos usan la MISMA vecindad (2-opt), el MISMO estado inicial por semilla
y el MISMO presupuesto de evaluaciones, para que la comparación sea justa.
Como validación, en una versión pequeña de 15 puntos se calcula el óptimo
exacto con programación dinámica (Held-Karp) y se compara contra ambos algoritmos.
"""

import math, random, time, itertools, statistics
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 1. DATOS DEL PROBLEMA (coordenadas sintéticas reproducibles, plano 100x100)
# ----------------------------------------------------------------------
def generar_instancia(n, semilla=2026):
    """Crea n puntos aleatorios reproducibles y la matriz de distancias."""
    global N_CIUDADES, CIUDADES, DIST
    N_CIUDADES = n
    rng_datos = random.Random(semilla)
    CIUDADES = [(rng_datos.uniform(0, 100), rng_datos.uniform(0, 100)) for _ in range(n)]
    DIST = [[math.dist(a, b) for b in CIUDADES] for a in CIUDADES]


generar_instancia(30)


# ----------------------------------------------------------------------
# 2. FUNCIÓN OBJETIVO Y VECINDAD
# ----------------------------------------------------------------------
def costo(ruta):
    """Función objetivo: longitud total del circuito cerrado (a minimizar)."""
    return sum(DIST[ruta[i]][ruta[(i + 1) % len(ruta)]] for i in range(len(ruta)))


def vecino_2opt(ruta, rng):
    """Vecindad 2-opt: se invierte un tramo [i, j] de la ruta.
    Reemplaza dos aristas por otras dos y siempre produce una ruta válida."""
    i, j = sorted(rng.sample(range(len(ruta)), 2))
    return ruta[:i] + ruta[i : j + 1][::-1] + ruta[j + 1 :]


# ----------------------------------------------------------------------
# 3. ALGORITMOS
# ----------------------------------------------------------------------
def hill_climbing(inicial, iteraciones, rng):
    """Hill Climbing estocástico: genera un vecino al azar y lo acepta
    solo si mejora (costo menor). Se detiene en un óptimo local."""
    actual, c_act = inicial[:], costo(inicial)
    historial = [c_act]
    for _ in range(iteraciones):
        cand = vecino_2opt(actual, rng)
        c_cand = costo(cand)
        if c_cand < c_act:
            actual, c_act = cand, c_cand
        historial.append(c_act)
    return actual, c_act, historial


def simulated_annealing(inicial, iteraciones, rng, T0=50.0, eta=0.9995):
    """Simulated Annealing: acepta empeoramientos con probabilidad
    exp(-delta/T). La temperatura baja geométricamente: T <- eta * T."""
    actual, c_act = inicial[:], costo(inicial)
    mejor, c_mejor = actual[:], c_act
    T = T0
    historial = [c_mejor]
    for _ in range(iteraciones):
        cand = vecino_2opt(actual, rng)
        c_cand = costo(cand)
        delta = c_cand - c_act
        if delta < 0 or rng.random() < math.exp(-delta / T):
            actual, c_act = cand, c_cand
            if c_act < c_mejor:
                mejor, c_mejor = actual[:], c_act
        T *= eta
        historial.append(c_mejor)
    return mejor, c_mejor, historial


def optimo_exacto():
    """Held-Karp (programación dinámica): costo mínimo exacto del TSP."""
    n = N_CIUDADES
    INF = float("inf")
    dp = {}
    for k in range(1, n):
        dp[(1 << k, k)] = DIST[0][k]
    for tam in range(2, n):
        for sub in itertools.combinations(range(1, n), tam):
            mask = sum(1 << k for k in sub)
            for k in sub:
                prev = mask & ~(1 << k)
                dp[(mask, k)] = min(dp[(prev, m)] + DIST[m][k] for m in sub if m != k)
    full = sum(1 << k for k in range(1, n))
    return min(dp[(full, k)] + DIST[k][0] for k in range(1, n))


# ----------------------------------------------------------------------
# 4. EXPERIMENTO: 30 corridas independientes por algoritmo
# ----------------------------------------------------------------------
ITER, CORRIDAS = 20000, 30


def experimento(T0=50.0, eta=0.9995):
    hc, sa = [], []
    for semilla in range(CORRIDAS):
        rng0 = random.Random(semilla)
        inicial = list(range(N_CIUDADES))
        rng0.shuffle(inicial)
        r1 = hill_climbing(inicial, ITER, random.Random(1000 + semilla))
        r2 = simulated_annealing(inicial, ITER, random.Random(2000 + semilla), T0, eta)
        hc.append(r1)
        sa.append(r2)
    return hc, sa


def resumen(nombre, res, optimo):
    """Imprime mejor, media, desviación, peor, brecha vs referencia y veces que se igualó."""
    cs = [r[1] for r in res]
    gap = 100 * (statistics.mean(cs) - optimo) / optimo
    hits = sum(1 for c in cs if c <= optimo * 1.0001)
    print(
        f"{nombre:20s} mejor={min(cs):7.2f}  media={statistics.mean(cs):7.2f}  "
        f"desv={statistics.pstdev(cs):5.2f}  peor={max(cs):7.2f}  "
        f"brecha media={gap:5.2f}%  igualó la referencia: {hits}/{len(cs)}"
    )
    return cs


def dibujar_ruta(ax, ruta, titulo):
    xs = [CIUDADES[i][0] for i in ruta + [ruta[0]]]
    ys = [CIUDADES[i][1] for i in ruta + [ruta[0]]]
    ax.plot(xs, ys, "-o", color="#1F3864", lw=1.6, ms=6)
    for pos, i in enumerate(ruta):
        ax.annotate(str(i), CIUDADES[i], textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_title(titulo, fontsize=9)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)


def vecino_mas_cercano():
    ruta, libres = [0], set(range(1, N_CIUDADES))
    while libres:
        s_ = min(libres, key=lambda j: DIST[ruta[-1]][j])
        ruta.append(s_)
        libres.remove(s_)
    return ruta


if __name__ == "__main__":
    t0 = time.time()
    # ---------- A) Validación en una instancia pequeña (n = 15) ----------
    generar_instancia(15, 2026)
    opt15 = optimo_exacto()
    print(f"[Validación n=15] Óptimo exacto (Held-Karp): {opt15:.2f}")
    hc15, sa15 = experimento()
    resumen("HC  (n=15)", hc15, opt15)
    resumen("SA  (n=15)", sa15, opt15)

    # ---------- B) Instancia principal (n = 30) ----------
    generar_instancia(30, 2026)
    ruta_nn = vecino_mas_cercano()
    print(f"\n[Principal n=30] Heurística del vecino más cercano: {costo(ruta_nn):.2f}")
    hc, sa = experimento()
    # Sin óptimo exacto: se usa como referencia el mejor costo hallado en todas las corridas
    sens, todos = [], [r[1] for r in hc] + [r[1] for r in sa]
    for T0, eta in [(10, 0.9995), (50, 0.9995), (200, 0.9995), (50, 0.999), (50, 0.9999)]:
        if (T0, eta) == (50, 0.9995):
            s_ = sa  # ya calculado arriba
        else:
            _, s_ = experimento(T0, eta)
        cs = [r[1] for r in s_]
        todos += cs
        sens.append((T0, eta, T0 * eta**ITER, statistics.mean(cs), min(cs), statistics.pstdev(cs)))
    ref = min(todos)
    print(f"Mejor costo conocido (referencia) = {ref:.2f}")
    print(f"\nResultados tras {CORRIDAS} corridas de {ITER} iteraciones:")
    c_hc = resumen("Hill Climbing", hc, ref)
    c_sa = resumen("Simulated Annealing", sa, ref)
    gana_sa = sum(1 for a, b in zip(c_sa, c_hc) if a < b)
    empates = sum(1 for a, b in zip(c_sa, c_hc) if a == b)
    print(
        f"Comparación pareada (mismo estado inicial): SA mejor en {gana_sa}/{CORRIDAS}, "
        f"empates {empates}, HC mejor en {CORRIDAS - gana_sa - empates}"
    )
    print("\nSensibilidad de Simulated Annealing (T0, eta):")
    for T0, eta, Tf, med, mej, sd in sens:
        print(
            f"T0={T0:5} eta={eta:<7} T_final={Tf:9.4f}  media={med:7.2f}  mejor={mej:7.2f}  desv={sd:5.2f}"
        )

    # --- Figuras ---
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for h in hc[:8]:
        ax.plot(h[2], color="#C0392B", alpha=0.18, lw=0.8)
    for h in sa[:8]:
        ax.plot(h[2], color="#2E86C1", alpha=0.18, lw=0.8)
    ax.plot(
        np.mean([h[2] for h in hc], axis=0),
        color="#C0392B",
        lw=2.4,
        label="Hill Climbing (media de 30 corridas)",
    )
    ax.plot(
        np.mean([h[2] for h in sa], axis=0),
        color="#2E86C1",
        lw=2.4,
        label="Simulated Annealing (media de 30 corridas)",
    )
    ax.axhline(ref, color="#27AE60", ls="--", lw=1.5, label=f"Mejor costo conocido = {ref:.1f}")
    ax.set_xlabel("Iteración (evaluaciones de vecinos)")
    ax.set_ylabel("Mejor costo encontrado")
    ax.set_title("Convergencia de Hill Climbing vs Simulated Annealing (30 puntos)")
    ax.set_xscale("symlog", linthresh=100)
    ax.set_xlim(0, ITER)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("convergencia.png", dpi=170)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(11, 3.9))
    dibujar_ruta(
        axs[0],
        hc[min(range(CORRIDAS), key=lambda i: hc[i][1])][0],
        f"Mejor Hill Climbing ({min(c_hc):.1f})",
    )
    dibujar_ruta(
        axs[1],
        sa[min(range(CORRIDAS), key=lambda i: sa[i][1])][0],
        f"Mejor Simulated Annealing ({min(c_sa):.1f})",
    )
    dibujar_ruta(axs[2], list(ruta_nn), f"Vecino más cercano ({costo(ruta_nn):.1f})")
    fig.tight_layout()
    fig.savefig("rutas.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot([c_hc, c_sa], tick_labels=["Hill Climbing", "Simulated\nAnnealing"], patch_artist=True)
    for p, c in zip(bp["boxes"], ["#F1948A", "#85C1E9"]):
        p.set_facecolor(c)
    ax.axhline(ref, color="#27AE60", ls="--", label=f"Mejor conocido = {ref:.1f}")
    ax.set_ylabel("Costo final")
    ax.set_title("Distribución del costo final (30 corridas)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("cajas.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    xs, ys = zip(*CIUDADES)
    ax.scatter(xs, ys, c="#1F3864", s=45)
    for i, (x, y) in enumerate(CIUDADES):
        ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_title("Los 30 puntos de reparto (coordenadas sintéticas)", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("puntos.png", dpi=170)
    plt.close(fig)
    print(f"\nTiempo total de ejecución: {time.time() - t0:.1f} s")
