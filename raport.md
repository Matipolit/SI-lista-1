# Sztuczna inteligencja - lista 1

### Mateusz Polito 266581

## 0. Informacje ogólne

Jako język programowania wybrałem Rust.
Do obsługi grafów użyłem biblioteki petgraph.
Do przechowywania danych o przystankach i połączeniach między nimi stworzyłem struktury

```rust
struct BusRoute {
  company: String,
  line: String,
  departure_time: MyTime,
  arrival_time: MyTime,
}
```

i

```rust
struct BusStop {
    id: Option<NodeIndex>,
    name: String,
    coords: Coords,
}

struct Coords {
    lat: f32,
    lon: f32,
}
```

Aby przechowywać czas stworzyłem strukturę:

```rust
struct MyTime {
    hour: i32,
    minute: i32,
}
```

## 1. Przetworzenie danych

Oczytywanie danych i budowa grafu odbywa się w poniższej funkcji

```rust
fn read_records(
    file_name: String,
) -> Result<(HashMap<NodeIndex, BusStop>, Graph<BusStop, BusRoute>), Box<dyn Error>> {
    let file = File::open(file_name)?;
    let mut rdr = csv::Reader::from_reader(file);
    let mut unique_stops: HashMap<NodeIndex, BusStop> = HashMap::with_capacity(1000);
    let mut stops_cache: HashSet<BusStop> = HashSet::with_capacity(1000);
    let mut graph = Graph::with_capacity(1000, 1000000);

    for (idx, result) in rdr.deserialize::<TransportRecordSerial>().enumerate() {
        let record: TransportRecord = result.unwrap().into();

        let start_stop = record.start_stop;
        let end_stop = record.end_stop;

        let mut start_id: Option<NodeIndex> = None;
        let mut end_id: Option<NodeIndex> = None;

        // add new bus stops to set, get their ids in graph
        if !stops_cache.contains(&start_stop) {
            let _insert = start_id.insert(graph.add_node(start_stop.clone()));
            stops_cache.insert(BusStop {
                id: start_id,
                ..start_stop
            });
        } else {
            start_id = Some(stops_cache.get(&start_stop).unwrap().id.unwrap());
        }

        if !stops_cache.contains(&end_stop) {
            let _insert = end_id.insert(graph.add_node(end_stop.clone()));
            stops_cache.insert(BusStop {
                id: end_id,
                ..end_stop
            });
        } else {
            end_id = Some(stops_cache.get(&end_stop).unwrap().id.unwrap());
        }

        // add edge between stops to graph
        graph.add_edge(
            start_id.expect("Start stop id not found"),
            end_id.expect("End stop id not found"),
            record.route,
        );

        if idx % 100000 == 0 {
            println!("Deserialized {} records", idx);
        }
    }
    stops_cache.into_iter().for_each(|stop| {
        unique_stops.insert(stop.id.unwrap(), stop);
    });
    return Ok((unique_stops, graph));
}
```
## 2. Algorytm Dijkstry

Algorytm Dijkstry wyszukuje dla wierzchołka początkowego optymalne ścieżki dojścia do innych wierzchołków.

Moje implementacje algorytmów Dijkstry i A* dzielą funkcje pomocnicze

```rust 
/// for edges connecting two bus stops, get the best one
fn get_best_route_between<'a>(
    edges: EdgesConnecting<'a, &'a BusRoute, petgraph::Directed>,
    time: u16,
) -> Option<&'a &'a BusRoute> {
    let mut best_edge: Option<EdgeReference<'a, &'a BusRoute>> = None;
    edges.for_each(|edge| match best_edge {
        Some(best) => {
            if time <= edge.weight().departure_time.to_minutes()
                && edge.weight().departure_time < best.weight().departure_time
            {
                best_edge = Some(edge);
            }
        }
        None => {
            if time <= edge.weight().departure_time.to_minutes() {
                best_edge = Some(edge);
            }
        }
    });
    best_edge.map(|edge| edge.weight())
}

type Path = Vec<(BusStop, Option<BusRoute>)>;

// generate a path Vec
fn generate_path(
    path: Vec<NodeIndex>,
    graph: &Graph<BusStop, BusRoute>,
    distances: HashMap<NodeIndex, (u16, Option<BusRoute>)>,
) -> Path {
    let mut path_vec = Vec::new();
    for node in path {
        let stop = graph.node_weight(node).unwrap().clone();
        let route = distances[&node].1.clone();
        path_vec.push((stop, route));
    }
    path_vec
}

```

Moja implementacja algorytmu Dijkstry:


```rust 
fn dijkstra(
    stop_a: BusStop,
    stop_b: BusStop,
    beginning_time: MyTime,
    graph: &Graph<BusStop, BusRoute>,
    all_stops: &HashMap<NodeIndex, BusStop>,
) -> Path {
    println!("Dijkstra start");

    let stop_a_id = stop_a.id.unwrap();
    let stop_b_id = stop_b.id.unwrap();

    // separate set of all stops because removing elements from graph shifts indices
    let mut q_set: HashSet<NodeIndex> = all_stops
        .clone()
        .into_iter()
        .map(|(index, _)| index)
        .collect();
    let mut q_set_size = q_set.len();
    graph.node_indices().for_each(|idx| {
        q_set.insert(idx);
    });

    // filter out edges before departure for speedup
    let filtered_graph = graph.filter_map(
        |_, node| Some(node),
        |_, edge| {
            if edge.departure_time < beginning_time {
                None
            } else {
                Some(edge)
            }
        },
    );

    let mut distances: HashMap<NodeIndex, (u16, Option<BusRoute>)> =
        HashMap::with_capacity(q_set_size);
    q_set.clone().into_iter().for_each(|node| {
        distances.insert(node, (u16::MAX, None));
    });

    let mut predecessors: HashMap<NodeIndex, Option<NodeIndex>> =
        HashMap::with_capacity(q_set_size);
    q_set.clone().into_iter().for_each(|node| {
        predecessors.insert(node, None);
    });

    let mut current_stop_id: NodeIndex = stop_a_id;

    distances.insert(current_stop_id, (beginning_time.to_minutes(), None));

    while q_set_size > 0 {
        if q_set_size % 100 == 0 {
            println!("{} nodes in Q remaining", q_set_size);
        }

        // get node from q_set with the lowest distance
        let mut current_lowest_node = q_set.clone().into_iter().next();
        if current_lowest_node.is_some() {
            let mut current_lowest_distance = distances[&current_lowest_node.unwrap()].0;
            for &stop in &q_set {
                let checked_distance = distances[&stop].0;
                if checked_distance < current_lowest_distance {
                    current_lowest_distance = checked_distance;
                    current_lowest_node = Some(stop);
                }
            }
            current_stop_id = current_lowest_node.unwrap();
        } else {
            println!("No node found in Q! Loop should stop now!");
        }

        q_set.remove(&current_stop_id);
        q_set_size -= 1;

        let current_neighbors =
            filtered_graph.neighbors_directed(current_stop_id, petgraph::Direction::Outgoing);
        let current_distance = distances[&current_stop_id].0;

        // update the distances table with lower distances if found
        // update the predecessors table if lower distance found for neighbor
        current_neighbors.for_each(|neighbor| {
            let neighbor_edges = filtered_graph.edges_connecting(current_stop_id, neighbor);
            let best_bus_route_opt = get_best_route_between(neighbor_edges, current_distance);
            if let Some(best_bus_route) = best_bus_route_opt {
                let neighbor_weight = best_bus_route.arrival_time.to_minutes();
                if neighbor_weight < distances[&neighbor].0 {
                    distances.insert(
                        neighbor,
                        (neighbor_weight, Some(best_bus_route.deref().clone())),
                    );
                    predecessors.insert(neighbor, Some(current_stop_id));
                }
            }
        });
    }

    let mut path = Vec::new();
    let mut current_node = stop_b_id;

    // Reconstruct the path from stop_b to stop_a
    let mut i = 0;
    while let Some(&pred) = predecessors.get(&current_node) {
        i += 1;
        path.push(current_node);
        if pred.unwrap() == stop_a_id {
            path.push(pred.unwrap());
            break;
        }
        current_node = pred.unwrap();
        if i > 1000 {
            println!("Path longer than 1000 stops, breaking");
            break;
        }
    }

    path.reverse();

    // return path in a nice format
    return generate_path(path, graph, distances);
}
```

Działanie algorytmu dijkstra dla trasy Piastowska -> FAT
``` 
Stop a found: BusStop { id: Some(NodeIndex(123)), name: "Piastowska", coords: Coords { lat: 51.116207, lon: 17.060907 } }
Stop b found: BusStop { id: Some(NodeIndex(31)), name: "FAT", coords: Coords { lat: 51.094128, lon: 16.978354 } }
Euclidean distance between a and b: 0
Dijkstra start
Size of Q set: 939
900 nodes in Q remaining
800 nodes in Q remaining
700 nodes in Q remaining
600 nodes in Q remaining
500 nodes in Q remaining
400 nodes in Q remaining
300 nodes in Q remaining
200 nodes in Q remaining
100 nodes in Q remaining
It took 5767ms

Piastowska -> [MPK Tramwaje] [19] PL. GRUNWALDZKI (12:00-12:04) -> [12] most Grunwaldzki (12:05-12:06) -> Urząd Wojewódzki (Impart) (12:06-12:08) -> GALERIA DOMINIKAŃSKA (12:08-12:11) -> [2] Wzgórze Partyzantów (12:11-12:12) -> [MPK Autobusy] [149] Renoma (12:15-12:17) -> pl. Orląt Lwowskich (12:17-12:20) -> [127] Tęczowa (12:20-12:21) -> Grabiszyńska (12:21-12:23) -> [MPK Tramwaje] [14] Pereca (12:23-12:25) -> Stalowa (12:25-12:27) -> pl. Srebrny (12:27-12:28) -> Bzowa (Centrum Historii Zajezdnia) (12:28-12:29) -> Hutmen (12:29-12:30) -> FAT (12:30-12:32)
Full route time: 32 minutes
Route line changes: 5
```

## 3. Algorytm A*

Algorytm A* jest algorytmem Dijkstry zoptymalizowanym pod szukanie ścieżki do konkretnego celu.
Odzwierciedla to moja implementacja, która jest analogiczna do implementacji Dijkstry.
Za pomocą parametru ``` limit_line_changes ``` można wybrać kryterium optymalizacyjne jako minimalizację liczby zmian linii, domyślnie jest to tak jak w Dijkstra minimalizacja czasu.

Jako funkcję estymacji kosztu wybrałem odległość Euklidesową:
```rust 
/// euclidean distance between two sets of coordinates
fn euclidean_distance(a: &Coords, b: &Coords) -> f32 {
    let dx = a.lat - b.lat;
    let dy = a.lon - b.lon;
    (dx * dx + dy * dy).sqrt()
}
```

Implementacja A*:

```rust 

fn astar(
    stop_a: BusStop,
    stop_b: BusStop,
    beginning_time: MyTime,
    graph: &Graph<BusStop, BusRoute>,
    all_stops: &HashMap<NodeIndex, BusStop>,
    limit_line_changes: bool,
) -> Path {
    println!("Astar start");
    if limit_line_changes {
        println!("Limiting line changes");
    }

    let stop_a_id = stop_a.id.unwrap();
    let stop_b_id = stop_b.id.unwrap();

    // separate set of all stops because removing elements from graph shifts indices
    let mut q_set: HashSet<NodeIndex> = all_stops
        .clone()
        .into_iter()
        .map(|(index, _)| index)
        .collect();
    let mut q_set_size = q_set.len();
    graph.node_indices().for_each(|idx| {
        q_set.insert(idx);
    });

    // filter out edges before departure for speedup
    let filtered_graph = graph.filter_map(
        |_, node| Some(node),
        |_, edge| {
            if edge.departure_time < beginning_time {
                None
            } else {
                Some(edge)
            }
        },
    );
    let mut distances: HashMap<NodeIndex, (u16, Option<BusRoute>)> =
        HashMap::with_capacity(q_set_size);
    q_set.clone().into_iter().for_each(|node| {
        distances.insert(node, (u16::MAX, None));
    });

    let mut predecessors: HashMap<NodeIndex, Option<NodeIndex>> =
        HashMap::with_capacity(q_set_size);
    q_set.clone().into_iter().for_each(|node| {
        predecessors.insert(node, None);
    });

    let mut current_stop_id: NodeIndex = stop_a_id;

    println!("Size of Q set: {}", q_set_size);

    distances.insert(current_stop_id, (beginning_time.to_minutes(), None));

    while q_set_size > 0 {
        if q_set_size % 100 == 0 {
            println!("{} nodes in Q remaining", q_set_size);
        }

        if current_stop_id == stop_b_id {
            println!("Found stop b in Q set, breaking loop");
            break;
        }

        // get node from q_set with the lowest distance
        let mut current_lowest_node = q_set.clone().into_iter().next();
        if current_lowest_node.is_some() {
            let mut current_lowest_distance = distances[&current_lowest_node.unwrap()].0;
            for &stop in &q_set {
                let checked_distance = distances[&stop].0;
                if checked_distance < current_lowest_distance {
                    current_lowest_distance = checked_distance;
                    current_lowest_node = Some(stop);
                }
            }
            current_stop_id = current_lowest_node.unwrap();
        } else {
            println!("No node found in Q! Loop should stop now!");
        }

        q_set.remove(&current_stop_id);
        q_set_size -= 1;

        let current_neighbors =
            filtered_graph.neighbors_directed(current_stop_id, petgraph::Direction::Outgoing);
        let current_distance = distances[&current_stop_id].0;

        // update the distances table with lower distances if found
        // update the predecessors table if lower distance found for neighbor
        current_neighbors.for_each(|neighbor| {
            let neighbor_edges = filtered_graph.edges_connecting(current_stop_id, neighbor);
            let best_bus_route_opt = get_best_route_between(neighbor_edges, current_distance);
            if let Some(best_bus_route) = best_bus_route_opt {
                // Calculate the weight of the neighbor including the distance to the final stop
                // Add 30 if the criterion is to limit line changes and the line changes
                let neighbor_weight = {
                    let neighbor_stop = all_stops[&neighbor].clone();
                    let mut weight = best_bus_route.arrival_time.to_minutes()
                        + (12. * euclidean_distance(&neighbor_stop.coords, &stop_b.coords)) as u16;
                    if limit_line_changes {
                        let current_stop_route = distances[&current_stop_id].1.clone();
                        if current_stop_route.is_some()
                            && best_bus_route.line != current_stop_route.unwrap().line
                        {
                            weight += 30;
                        }
                    }
                    weight
                };

                if neighbor_weight < distances[&neighbor].0 {
                    distances.insert(
                        neighbor,
                        (neighbor_weight, Some(best_bus_route.deref().clone())),
                    );
                    predecessors.insert(neighbor, Some(current_stop_id));
                }
            }
        });
    }

    let mut path = Vec::new();
    let mut current_node = stop_b_id;

    // Reconstruct the path from stop_b to stop_a
    let mut i = 0;
    while let Some(&pred) = predecessors.get(&current_node) {
        i += 1;
        path.push(current_node);
        if pred.unwrap() == stop_a_id {
            path.push(pred.unwrap());
            break;
        }
        current_node = pred.unwrap();
        if i > 1000 {
            println!("Path longer than 1000 stops, breaking");
            break;
        }
    }

    path.reverse();

    //return the path in a nice format
    return generate_path(path, graph, distances);
}```

Działanie algorytmu A* dla trasy Piastowska -> FAT o godzinie 12:00 i minimalizacji czasu:

```
Stop a found: BusStop { id: Some(NodeIndex(123)), name: "Piastowska", coords: Coords { lat: 51.116207, lon: 17.060907 } }
Stop b found: BusStop { id: Some(NodeIndex(31)), name: "FAT", coords: Coords { lat: 51.094128, lon: 16.978354 } }
Euclidean distance between a and b: 0.08545551
Astar start
Size of Q set: 939
900 nodes in Q remaining
800 nodes in Q remaining
Found stop b in Q set, breaking loop
It took 4358ms

Piastowska -> [MPK Tramwaje] [19] PL. GRUNWALDZKI (12:00-12:04) -> [12] most Grunwaldzki (12:05-12:06) -> [MPK Autobusy] [149] Poczta Główna (12:09-12:11) -> skwer Krasińskiego (12:11-12:14) -> Wzgórze Partyzantów (12:14-12:15) -> Renoma (12:15-12:17) -> pl. Orląt Lwowskich (12:17-12:20) -> [127] Tęczowa (12:20-12:21) -> Grabiszyńska (12:21-12:23) -> [MPK Tramwaje] [14] Pereca (12:23-12:25) -> Stalowa (12:25-12:27) -> pl. Srebrny (12:27-12:28) -> Bzowa (Centrum Historii Zajezdnia) (12:28-12:29) -> Hutmen (12:29-12:30) -> FAT (12:30-12:32)
Full route time: 32 minutes
Route line changes: 4
```

Działanie algorytmu A* dla trasy Piastowska -> FAT o godzinie 12:00 i minimalizacji przesiadek:

```
Stop a found: BusStop { id: Some(NodeIndex(123)), name: "Piastowska", coords: Coords { lat: 51.116207, lon: 17.060907 } }
Stop b found: BusStop { id: Some(NodeIndex(31)), name: "FAT", coords: Coords { lat: 51.094128, lon: 16.978354 } }
Euclidean distance between a and b: 0.08545551
Astar start
Limiting line changes
Size of Q set: 939
900 nodes in Q remaining
800 nodes in Q remaining
700 nodes in Q remaining
Found stop b in Q set, breaking loop
It took 4458ms

Piastowska -> [MPK Tramwaje] [19] Górnickiego (12:04-12:06) -> Ogród Botaniczny (12:06-12:08) -> pl. Bema (12:08-12:10) -> Dubois (12:10-12:12) -> Pomorska (12:12-12:14) -> Kępa Mieszczańska (12:14-12:17) -> [MPK Autobusy] [144] PL. JANA PAWŁA II (12:21-12:23) -> pl. Orląt Lwowskich (12:53-12:55) -> pl. Legionów (12:55-12:57) -> Piłsudskiego (12:57-12:59) -> Zielińskiego (12:59-13:01) -> Zaporoska (13:01-13:03) -> [126] Grabiszyńska (13:04-13:05) -> Pereca (13:35-13:38) -> Stalowa (13:38-13:40) -> pl. Srebrny (13:40-13:41) -> Bzowa (Centrum Historii Zajezdnia) (13:41-13:42) -> Hutmen (13:42-13:43) -> FAT (13:43-13:45)
Full route time: 101 minutes
Route line changes: 2
```


