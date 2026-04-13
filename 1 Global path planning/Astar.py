
import osmnx as ox
import random
import heapq
import math

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

def manhattan(node, dest, convert_to_meters=True):
    x1, y1 = G.nodes[node]["x"], G.nodes[node]["y"]
    x2, y2 = G.nodes[dest]["x"], G.nodes[dest]["y"]
    
    if convert_to_meters:
        avg_lat = math.radians((y1 + y2) / 2)
        dx = abs(x1 - x2) * 111320 * math.cos(avg_lat)
        dy = abs(y1 - y2) * 111320
    else:
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        
    return dx + dy

def euclidean(node, dest, convert_to_meters=True):
    x1, y1 = G.nodes[node]["x"], G.nodes[node]["y"]
    x2, y2 = G.nodes[dest]["x"], G.nodes[dest]["y"]
    
    if convert_to_meters:
        avg_lat = math.radians((y1 + y2) / 2)
        dx = (x1 - x2) * 111320 * math.cos(avg_lat)
        dy = (y1 - y2) * 111320
    else:
        dx = (x1 - x2)
        dy = (y1 - y2)
        
    return math.sqrt(dx**2 + dy**2)

def haversine(node, dest):
    x1, y1 = G.nodes[node]["x"], G.nodes[node]["y"]
    x2, y2 = G.nodes[dest]["x"], G.nodes[dest]["y"]
    phi1 = math.radians(y1)
    phi2 = math.radians(y2)
    delta_phi = math.radians(y1 - y2)
    delta_lambda = math.radians(x1 - x2)
    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    R = 6371000 
    return R * c

def astar(orig, dest, heuristic, max_speed=1, plot=False):
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
    pq = [(heuristic(orig, dest) / max_speed, orig)]
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
                f_score = G.nodes[neighbor]["distance"] + heuristic(neighbor, dest) / max_speed
                heapq.heappush(pq, (f_score, neighbor))
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
        # Convert maxspeed (km/h) to m/s for travel time in seconds
        speed_ms = maxspeed / 3.6
        G.edges[edge]["weight"] = G.edges[edge]["length"] / speed_ms

    # Global max speed in m/s for A* heuristic
    max_speed = max(G.edges[edge]["maxspeed"] for edge in G.edges) / 3.6

    heuristics_converted = {
        "Manhattan (in Metri)": lambda n, d: manhattan(n, d, convert_to_meters=True),
        "Euclidean (in Metri)": lambda n, d: euclidean(n, d, convert_to_meters=True),
        "Haversine (in Metri)": haversine 
    }
    
    heuristics_raw = {
        "Manhattan (Raw degrees)": lambda n, d: manhattan(n, d, convert_to_meters=False),
        "Euclidean (Raw degrees)": lambda n, d: euclidean(n, d, convert_to_meters=False)
    }

    # Set seed for reproducibility
    random.seed(42)
    num_runs = 10
    
    nodes_list = sorted(list(G.nodes))
    pairs = []
    while len(pairs) < num_runs:
        s = random.choice(nodes_list)
        e = random.choice(nodes_list)
        if s != e:
            pairs.append((s, e))

    print("Running A*")
    print("Place:", place_name)
    print("Nodes:", len(G.nodes))
    print("Edges:", len(G.edges))

    print("\n" + "="*50)
    print("TEST 1: HEURISTICS CONVERTED TO METERS")
    print("="*50)
    for h_name, h_func in heuristics_converted.items():
        all_iterations = []
        print(f"\n--- Heuristic: {h_name} ---")
        for i, (start, end) in enumerate(pairs):
            iters = astar(start, end, h_func, max_speed)
            if iters is not None:
                all_iterations.append(iters)
                reconstruct_path(start, end, algorithm="astar")
            else:
                print(f"Run {i+1}: no path found")
        avg = sum(all_iterations) / len(all_iterations) if all_iterations else 0
        print(f"Average: {avg:.1f}")

    print("\n" + "="*50)
    print("TEST 2: HEURISTICS IN PURE DEGREES (No conversion)")
    print("="*50)
    for h_name, h_func in heuristics_raw.items():
        all_iterations = []
        print(f"\n--- Heuristic: {h_name} ---")
        for i, (start, end) in enumerate(pairs):
            iters = astar(start, end, h_func, max_speed)
            if iters is not None:
                all_iterations.append(iters)
            else:
                print(f"Run {i+1}: no path found")
        avg = sum(all_iterations) / len(all_iterations) if all_iterations else 0
        print(f"Average: {avg:.1f}")