# The script loads a driving network from OpenStreetMap (using osmnx), sets each edge's travel time as its 
# weight (length / maxspeed), runs a Dijkstra shortest-path search (using heapq) between two randomly chosen nodes, 
# styles edges/nodes for visualization while the algorithm runs, reconstructs the found path, increments an 
# edge usage counter (dijkstra_uses) for the path, and finally draws a graph visualization showing visited/active/path 
# edges. The graph object G is a global osmnx MultiDiGraph.

import osmnx as ox
import random
import heapq

def style_unvisited_edge(edge):        
    G.edges[edge]["color"] = "gray"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 0.2

def style_visited_edge(edge):
    G.edges[edge]["color"] = "green"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 1

def style_active_edge(edge):
    G.edges[edge]["color"] = "red"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 1

def style_path_edge(edge):
    G.edges[edge]["color"] = "white"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 5

def plot_graph():
    ox.plot_graph(
        G,
        node_size =  [ G.nodes[node]["size"] for node in G.nodes ],
        edge_color = [ G.edges[edge]["color"] for edge in G.edges ],
        edge_alpha = [ G.edges[edge]["alpha"] for edge in G.edges ],
        edge_linewidth = [ G.edges[edge]["linewidth"] for edge in G.edges ],
        node_color = "white",
        bgcolor = "black"
    )

def dijkstra(orig, dest, plot=False):
    for node in G.nodes:
        G.nodes[node]["visited"] = False
        G.nodes[node]["distance"] = float("inf")
        G.nodes[node]["previous"] = None
        G.nodes[node]["size"] = 0
    for edge in G.edges:
        style_unvisited_edge(edge)
    G.nodes[orig]["distance"] = 0
    G.nodes[orig]["size"] = 50
    G.nodes[dest]["size"] = 50
    pq = [(0, orig)]
    step = 0
    while pq:
        _, node = heapq.heappop(pq)
        if node == dest:
            return step
        if G.nodes[node]["visited"]: continue
        G.nodes[node]["visited"] = True
        for edge in G.out_edges(node):
            style_visited_edge((edge[0], edge[1], 0))
            neighbor = edge[1]
            weight = G.edges[(edge[0], edge[1], 0)]["weight"]
            if G.nodes[neighbor]["distance"] > G.nodes[node]["distance"] + weight:
                G.nodes[neighbor]["distance"] = G.nodes[node]["distance"] + weight
                G.nodes[neighbor]["previous"] = node
                heapq.heappush(pq, (G.nodes[neighbor]["distance"], neighbor))
                for edge2 in G.out_edges(neighbor):
                    style_active_edge((edge2[0], edge2[1], 0))
        step += 1

def reconstruct_path(orig, dest, plot=False, algorithm=None):
    dist = 0
    speeds = []
    curr = dest
    while curr != orig:
        prev = G.nodes[curr]["previous"]
        if prev is None:
            break
        dist += G.edges[(prev, curr, 0)]["length"]
        speeds.append(G.edges[(prev, curr, 0)]["maxspeed"])
        style_path_edge((prev, curr, 0))
        if algorithm:
            G.edges[(prev, curr, 0)][f"{algorithm}_uses"] = G.edges[(prev, curr, 0)].get(f"{algorithm}_uses", 0) + 1
        curr = prev
    dist /= 1000

if __name__ == "__main__":
    place_name = "Aosta, Aosta, Italy"
    #place_name = "Turin, Piedmont, Italy"

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
        # Convert maxspeed (km/h) to m/s
        speed_ms = maxspeed / 3.6
        G.edges[edge]["weight"] = G.edges[edge]["length"] / speed_ms
        G.edges[edge]["dijkstra_uses"] = 0

    # Set seed for reproducibility
    random.seed(42)
    num_runs = 10
    
    # Sort nodes to ensure identical randomization across files
    nodes_list = sorted(list(G.nodes))
    pairs = []
    while len(pairs) < num_runs:
        s = random.choice(nodes_list)
        e = random.choice(nodes_list)
        if s != e:
            pairs.append((s, e))

    print("Running Dijkstra")
    print("Place:", place_name)
    print("Nodes:", len(G.nodes))
    print("Edges:", len(G.edges))
    print()

    all_iterations = []
    for i, (start, end) in enumerate(pairs):
        iters = dijkstra(start, end)
        if iters is not None:
            all_iterations.append(iters)
            print(f"Run {i+1}: {iters} iterations")
            reconstruct_path(start, end, algorithm="dijkstra")
        else:
            print(f"Run {i+1}: no path found")

    avg_iterations = sum(all_iterations) / len(all_iterations) if all_iterations else 0
    print()
    print(f"Average iterations over {len(all_iterations)} runs: {avg_iterations:.1f}")

    # Plot only the last run
    plot_graph()