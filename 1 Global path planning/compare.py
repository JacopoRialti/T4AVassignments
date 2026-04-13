import osmnx as ox
import random
import os
import csv
import matplotlib.pyplot as plt

import Dijkstra
import Astar

def load_graph(place_name):
    G = ox.graph_from_place(place_name, network_type="drive")
    G = ox.truncate.largest_component(G, strongly=True)
    for edge in G.edges:
        maxspeed = 40
        if "maxspeed" in G.edges[edge]:
            maxspeed = G.edges[edge]["maxspeed"]
            if type(maxspeed) == list:
                speeds = [int(speed) if speed != "walk" else 1 for speed in maxspeed]
                maxspeed = min(speeds)
            elif type(maxspeed) == str:
                if maxspeed == "walk": 
                    maxspeed = 1
                else:
                    maxspeed = int(maxspeed.strip(" mph"))
        G.edges[edge]["maxspeed"] = maxspeed
        # Convert maxspeed from km/h to m/s to get cost in real seconds
        speed_ms = maxspeed / 3.6
        G.edges[edge]["weight"] = G.edges[edge]["length"] / speed_ms
    return G

if __name__ == "__main__":
    #place_name = "Aosta, Aosta, Italy"
    place_name = "Turin, Piedmont, Italy"


    # IMPOSTAZIONE SEED PER RIPRODUCIBILITA'
    random.seed(42)
    num_runs = 10

    print("=" * 100)
    print(f"Loading graph: {place_name}")
    G = load_graph(place_name)
    print(f"Nodes: {len(G.nodes)}, Edges: {len(G.edges)}")
    print("=" * 100)

    # Condividi il grafo con entrambi i moduli
    Dijkstra.G = G
    Astar.G = G

    # Generazione coppie esattamente identica agli altri script
    nodes_list = sorted(list(G.nodes))
    pairs = []
    while len(pairs) < num_runs:
        s = random.choice(nodes_list)
        e = random.choice(nodes_list)
        if s != e:
            pairs.append((s, e))

    # Velocita' massima globale in m/s x l'euristica A*
    max_speed = max(G.edges[edge]["maxspeed"] for edge in G.edges) / 3.6

    # Definiamo gli algoritmi (entrambe le versioni di A*)
    algorithms = {
        "Dijkstra":       lambda s, e: Dijkstra.dijkstra(s, e),
        "A* Man (Metri)": lambda s, e: Astar.astar(s, e, lambda n, d: Astar.manhattan(n, d, convert_to_meters=True), max_speed),
        "A* Euc (Metri)": lambda s, e: Astar.astar(s, e, lambda n, d: Astar.euclidean(n, d, convert_to_meters=True), max_speed),
        "A* Hav (Metri)": lambda s, e: Astar.astar(s, e, Astar.haversine, max_speed),
        "A* Man (Gradi)": lambda s, e: Astar.astar(s, e, lambda n, d: Astar.manhattan(n, d, convert_to_meters=False), max_speed),
        "A* Euc (Gradi)": lambda s, e: Astar.astar(s, e, lambda n, d: Astar.euclidean(n, d, convert_to_meters=False), max_speed),
    }

    results = {name: [] for name in algorithms}
    for i, (start, end) in enumerate(pairs):
        for name, algo_func in algorithms.items():
            iters = algo_func(start, end)
            results[name].append(iters)

    # Stampa la tabella a terminale
    header = f"{'Run':<6}"
    for name in algorithms:
        header += f"{name:>16}"
    print()
    print(f"{place_name} ({len(G.nodes)} nodi, {len(G.edges)} archi)")
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for i in range(num_runs):
        row = f"{i+1:<6}"
        for name in algorithms:
            val = results[name][i]
            if val is not None:
                row += f"{val:>16}"
            else:
                row += f"{'no path':>16}"
        print(row)

    print("-" * len(header))
    avg_row = f"{'Media':<6}"
    for name in algorithms:
        valid = [v for v in results[name] if v is not None]
        avg = sum(valid) / len(valid) if valid else 0
        avg_row += f"{avg:>16.1f}"
    print(avg_row)
    print()

    # Salvataggio CSV
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/img", exist_ok=True)
    safe_place = place_name.replace(", ", "_").replace(" ", "_").lower()
    csv_path = f"results/tables/{safe_place}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Run"] + list(algorithms.keys()))
        for i in range(num_runs):
            row = [i + 1]
            for name in algorithms:
                val = results[name][i]
                row.append(val if val is not None else "no path")
            writer.writerow(row)
        avg_row_csv = ["Media"]
        for name in algorithms:
            valid = [v for v in results[name] if v is not None]
            avg = sum(valid) / len(valid) if valid else 0
            avg_row_csv.append(f"{avg:.1f}")
        writer.writerow(avg_row_csv)
    print(f"Saved {csv_path}")

    # Plot (salva TUTTE le immagini per avere la prova visiva)
    last_start, last_end = pairs[-1]
    plot_algorithms = {
        "Dijkstra":               lambda s, e: Dijkstra.dijkstra(s, e),
        "Astar_Manhattan_Metri":  lambda s, e: Astar.astar(s, e, lambda n, d: Astar.manhattan(n, d, True), max_speed),
        "Astar_Euclidean_Metri":  lambda s, e: Astar.astar(s, e, lambda n, d: Astar.euclidean(n, d, True), max_speed),
        "Astar_Haversine_Metri":  lambda s, e: Astar.astar(s, e, Astar.haversine, max_speed),
        "Astar_Manhattan_Gradi":  lambda s, e: Astar.astar(s, e, lambda n, d: Astar.manhattan(n, d, False), max_speed),
        "Astar_Euclidean_Gradi":  lambda s, e: Astar.astar(s, e, lambda n, d: Astar.euclidean(n, d, False), max_speed),
    }

    for name, algo_func in plot_algorithms.items():
        algo_func(last_start, last_end)
        if name == "Dijkstra":
            Dijkstra.reconstruct_path(last_start, last_end, algorithm="dijkstra")
        else:
            Astar.reconstruct_path(last_start, last_end, algorithm="astar")

        safe_name = name.replace("*", "star").replace(" ", "_")
        fig, ax = ox.plot_graph(
            G,
            node_size=[G.nodes[node]["size"] for node in G.nodes],
            edge_color=[G.edges[edge]["color"] for edge in G.edges],
            edge_alpha=[G.edges[edge]["alpha"] for edge in G.edges],
            edge_linewidth=[G.edges[edge]["linewidth"] for edge in G.edges],
            node_color="white",
            bgcolor="black",
            show=False,
            close=False
        )
        ax.set_title(f"{place_name} - {name.replace('_', ' ')}", color="white", fontsize=16)
        filepath = f"results/img/{safe_place}_{safe_name}.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="black")
        plt.close(fig)
        print(f"Saved {filepath}")

    print("All plots saved.")