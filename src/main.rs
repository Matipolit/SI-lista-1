use rayon::{prelude::*, vec};
use serde::Deserializer;
use std::collections::{HashMap, VecDeque};
use std::ops::Deref;

use std::sync::{Arc, Mutex};
use std::{
    cmp::{Eq, PartialEq, PartialOrd},
    collections::HashSet,
    convert::From,
    env,
    error::Error,
    fmt,
    fs::File,
    hash::{Hash, Hasher},
    time::Instant,
};

use petgraph::graph::{EdgeReference, EdgesConnecting};
use petgraph::visit::IntoNodeReferences;
use petgraph::{graph::NodeIndex, Graph};
use rand::prelude::SliceRandom;

#[derive(Debug, Clone)]
struct BusRoute {
    company: String,
    line: String,
    departure_time: MyTime,
    arrival_time: MyTime,
}

impl petgraph::EdgeType for BusRoute {
    fn is_directed() -> bool {
        true
    }
}

#[derive(Debug, Clone)]
struct TransportRecord {
    route: BusRoute,
    start_stop: BusStop,
    end_stop: BusStop,
}

impl From<TransportRecordSerial> for TransportRecord {
    fn from(value: TransportRecordSerial) -> Self {
        TransportRecord {
            route: BusRoute {
                company: value.company,
                line: value.line,
                departure_time: value.departure_time,
                arrival_time: value.arrival_time,
            },
            start_stop: BusStop {
                id: None,
                name: value.start_stop,
                coords: Coords {
                    lat: value.start_stop_lat.parse().unwrap(),
                    lon: value.start_stop_lon.parse().unwrap(),
                },
            },
            end_stop: BusStop {
                id: None,
                name: value.end_stop,
                coords: Coords {
                    lat: value.end_stop_lat.parse().unwrap(),
                    lon: value.end_stop_lon.parse().unwrap(),
                },
            },
        }
    }
}

#[derive(serde::Deserialize)]
struct TransportRecordSerial {
    id: i32,
    company: String,
    line: String,
    departure_time: MyTime,
    arrival_time: MyTime,
    start_stop: String,
    end_stop: String,
    start_stop_lat: String,
    start_stop_lon: String,
    end_stop_lat: String,
    end_stop_lon: String,
}

#[derive(Debug, Clone, Copy)]
struct MyTime {
    hour: u16,
    minute: u16,
    raw_minutes: u16,
}

impl fmt::Display for MyTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:02}:{:02}", self.hour, self.minute)
    }
}

impl MyTime {
    fn to_minutes(self) -> u16 {
        self.raw_minutes
    }

    fn time_difference_in_mins(&self, other_time: &MyTime) -> u16 {
        self.to_minutes() - other_time.to_minutes()
    }
}

impl From<&String> for MyTime {
    fn from(value: &String) -> Self {
        let str_split: Vec<&str> = value.split(':').collect();
        let hour = str_split[0].parse().unwrap();
        let minute = str_split[1].parse().unwrap();
        MyTime {
            hour,
            minute,
            raw_minutes: hour * 60 + minute,
        }
    }
}

impl PartialEq for MyTime {
    fn eq(&self, other: &Self) -> bool {
        self.to_minutes() == other.to_minutes()
    }
}

impl Eq for MyTime {}

impl PartialOrd for MyTime {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.to_minutes().partial_cmp(&other.to_minutes())
    }
}

#[test]
fn test_time_difference() {
    let arrival_time: MyTime = "20:53:00".to_owned().into();
    let departure_time: MyTime = "20:52:00".to_owned().into();
    let difference = arrival_time.time_difference_in_mins(&departure_time);
    assert_eq!(difference, 1);
}

impl<'de> serde::Deserialize<'de> for MyTime {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Ok(s.into())
    }
}

impl From<String> for MyTime {
    fn from(str_time: String) -> Self {
        let str_split: Vec<&str> = str_time.split(':').collect();
        let hour = str_split[0].parse().unwrap();
        let minute = str_split[1].parse().unwrap();
        MyTime {
            hour,
            minute,
            raw_minutes: hour * 60 + minute,
        }
    }
}

#[derive(Debug, Copy, Clone)]
struct Coords {
    lat: f32,
    lon: f32,
}

#[derive(Debug, Clone)]
struct BusStop {
    id: Option<NodeIndex>,
    name: String,
    coords: Coords,
}

impl Hash for BusStop {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

impl PartialEq for BusStop {
    fn eq(&self, other: &Self) -> bool {
        return self.name == other.name;
    }
}

impl Eq for BusStop {}

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

fn get_stop_by_name(graph: &Graph<BusStop, BusRoute>, name: &str) -> Option<BusStop> {
    let mut lower_name = name.to_lowercase();
    let strip_res = lower_name.strip_prefix(' ');
    if let Some(stripped) = strip_res {
        lower_name = stripped.to_string();
    };
    let bus_stop = Arc::new(Mutex::new(None));
    graph
        .node_references()
        .par_bridge()
        .for_each(|(idx, stop)| {
            if stop.name.to_lowercase() == lower_name {
                let mut guard = bus_stop.lock().unwrap();
                let mut cloned_stop = stop.clone();
                cloned_stop.id = Some(idx);
                *guard = Some(cloned_stop);
            }
        });
    Arc::try_unwrap(bus_stop)
        .ok()
        .and_then(|mutex| mutex.into_inner().ok())
        .unwrap()
}

/// for edges connecting two bus stops, get the best one
fn get_best_route_between<'a>(
    edges: EdgesConnecting<'a, &'a BusRoute, petgraph::Directed>,
    time: u16,
) -> Option<&'a &'a BusRoute> {
    let mut best_edge: Option<EdgeReference<'a, &'a BusRoute>> = None;
    edges.for_each(|edge| {
        let edge_weight = edge.weight();
        match best_edge {
            Some(best) => {
                if time <= edge_weight.departure_time.to_minutes()
                    && edge_weight.departure_time < best.weight().departure_time
                {
                    best_edge = Some(edge);
                }
            }
            None => {
                if time <= edge_weight.departure_time.to_minutes() {
                    best_edge = Some(edge);
                }
            }
        }
    });
    best_edge.map(|edge| edge.weight())
}

/// euclidean distance between two sets of coordinates
fn euclidean_distance(a: &Coords, b: &Coords) -> f32 {
    let dx = a.lat - b.lat;
    let dy = a.lon - b.lon;
    (dx * dx + dy * dy).sqrt()
}

type Path = Vec<(BusStop, Option<BusRoute>)>;

// generate a path in a nice format
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
}

fn tabu_search(
    start_stop: BusStop,
    stations_list: Vec<BusStop>,
    time_at_start: MyTime,
    graph: &Graph<BusStop, BusRoute>,
    all_stops: &HashMap<NodeIndex, BusStop>,
    limit_line_changes: bool,
    max_iterations: usize,
    max_tabu_size: Option<u32>,
) -> Path {
    #[derive(Clone)]
    struct Solution {
        path: Vec<BusStop>,
        cost: u16,
        full_path: Path,
    }

    impl Solution {
        fn new(path: Vec<BusStop>, cost: u16, full_path: Option<Path>) -> Self {
            Solution {
                path,
                cost,
                full_path: full_path.unwrap_or_default(),
            }
        }

        fn clone(&self) -> Solution {
            Solution {
                path: self.path.clone(),
                cost: self.cost,
                full_path: self.full_path.clone(),
            }
        }
    }

    // calculate the number of changes in a path.
    fn get_number_of_changes(path: &Path) -> u16 {
        let mut number_of_changes = 0;
        let mut curr_line: Option<String> = None;
        path.iter().for_each(|(_, route_opt)| {
            if let Some(route) = route_opt {
                if let Some(line) = curr_line.clone() {
                    if line != route.line {
                        number_of_changes += 1;
                    }
                    curr_line = Some(route.line.clone());
                } else {
                    curr_line = Some(route.line.clone());
                }
            }
        });
        number_of_changes
    }

    // generate neighbors for a given solution.
    fn generate_neighbour(solution: &Solution, start_stop: BusStop) -> Vec<Solution> {
        let mut neighbours = Vec::new();
        for i in 0..solution.path.len() - 1 {
            for j in i + 1..solution.path.len() - 2 {
                let mut new_path = solution.path.clone();
                new_path.remove(0);
                new_path.pop();
                new_path.swap(i, j);
                new_path.insert(0, start_stop.clone());
                new_path.push(start_stop.clone());
                neighbours.push(Solution::new(new_path, u16::MAX, None));
            }
        }
        neighbours
    }

    // calculate the cost for a solution.
    fn calculate_cost_for_solution(
        solution: &mut Solution,
        graph: &Graph<BusStop, BusRoute>,
        all_stops: &HashMap<NodeIndex, BusStop>,
        time_at_start: MyTime,
        limit_line_changes: bool,
    ) {
        solution.full_path.clear();

        let mut current_time = time_at_start;

        for i in 0..solution.path.len() - 1 {
            let stop_a = solution.path[i].clone();
            let stop_b = solution.path[i + 1].clone();

            let astar_path = astar(
                stop_a,
                stop_b,
                current_time,
                graph,
                all_stops,
                limit_line_changes,
            );
            current_time = astar_path.last().cloned().unwrap().1.unwrap().arrival_time;
            let sub_path_to_station = if i != 0 {
                &astar_path[1..]
            } else {
                &astar_path
            };

            solution.full_path.extend_from_slice(sub_path_to_station);
        }
        // Set the total cost of the solution
        solution.cost = current_time.to_minutes();
        if limit_line_changes {
            solution.cost += 30 * get_number_of_changes(&solution.full_path);
        }
    }

    let ran_gen = &mut rand::thread_rng();
    let mut random_path = stations_list.clone();
    random_path.shuffle(ran_gen);
    random_path.insert(0, start_stop.clone());
    random_path.push(start_stop.clone());
    let mut best_solution = Solution::new(random_path, u16::MAX, None);
    calculate_cost_for_solution(
        &mut best_solution,
        graph,
        all_stops,
        time_at_start,
        limit_line_changes,
    );
    let mut tabu_list: Mutex<VecDeque<Vec<BusStop>>> = Mutex::new(VecDeque::new());
    tabu_list.lock().unwrap().push_back(best_solution.path.clone());

    // Main loop of the algorithm.
    for _ in 0..max_iterations {
        let neighbours = generate_neighbour(&best_solution, start_stop.clone());
        let mut best_neighbour_cost = Mutex::new(u16::MAX);
        let mut best_neighbour = Mutex::new(None);

        neighbours.into_par_iter().for_each(|neighbour|{
            let neighbour_path = neighbour.path.clone();
            if !tabu_list.lock().unwrap().contains(&neighbour_path) {
                let mut neighbour = neighbour;
                calculate_cost_for_solution(
                    &mut neighbour,
                    graph,
                    all_stops,
                    time_at_start,
                    limit_line_changes,
                );
                tabu_list.lock().unwrap().push_back(neighbour_path);

                if best_neighbour_cost.lock().unwrap().gt(&neighbour.cost) {
                    best_neighbour.lock().unwrap().insert(neighbour.clone());
                    best_neighbour_cost.lock().unwrap().clone_from(&neighbour.cost);
                }
            }
        });
        let locked_best_neighbour = best_neighbour.lock().unwrap();
        if  locked_best_neighbour.is_some() {
            if <std::option::Option<Solution> as Clone>::clone(&locked_best_neighbour).unwrap().cost < best_solution.cost {
                best_solution = <std::option::Option<Solution> as Clone>::clone(&locked_best_neighbour).unwrap();
            }
        }
        if let Some(max_tabu_size) = max_tabu_size {
            if tabu_list.lock().unwrap().len() > (max_tabu_size as usize) + stations_list.len() {
                tabu_list.lock().unwrap().pop_front();
            }
        }
    }
    println!("Cost: {}", best_solution.cost);
    best_solution.full_path
}

fn print_path(path: Path) {
    println!();
    let mut curr_line: Option<String> = None;
    let mut curr_company: Option<String> = None;
    let mut first_stop_time: Option<MyTime> = None;
    let mut last_stop_time: Option<MyTime> = None;
    let mut route_line_changes = 0;
    for (i, path_elem) in path.iter().enumerate() {
        let (stop, route_opt) = path_elem;
        if i > 0 {
            print!(" -> ");
        }
        if let Some(route) = route_opt.clone() {
            match curr_company {
                Some(company) => {
                    if company != route.company {
                        print!("[{}] ", &route.company);
                    }
                    curr_company = Some(route.company);
                }
                None => {
                    print!("[{}] ", route.company);
                    curr_company = Some(route.company);
                }
            }
            match curr_line {
                Some(line) => {
                    if line != route.line {
                        print!("[{}] ", &route.line);
                        route_line_changes += 1;
                    }
                    curr_line = Some(route.line);
                }
                None => {
                    print!("[{}] ", route.line);
                    curr_line = Some(route.line);
                }
            }
            print!("{} ", stop.name);
            print!("({}-", route.departure_time);
            print!("{})", route.arrival_time);
            if first_stop_time.is_none() {
                first_stop_time = Some(route.departure_time);
            }
            last_stop_time = Some(route.arrival_time);
        } else {
            print!("{}", stop.name);
        }
    }
    println!(
        "\nFull route time: {} minutes",
        last_stop_time
            .unwrap()
            .time_difference_in_mins(&first_stop_time.unwrap())
    );
    println!("Route line changes: {}\n", route_line_changes);
}

fn main() {
    let args = env::args().collect::<Vec<String>>();
    if args.contains(&"help".to_owned()) || args.len() == 1 {
        println!("Parameters: <function [dijkstra, astar, tabu]> + arguments for function:
            dijkstra: <stop_a> <stop_b> <time>
            astar: <stop_a> <stop_b> <time> <criterion [t, p]>
            tabu: <stop_a> <time> <criterion [t, p]> <stop_list [bus stop names separated by commas]> <length_limit [optional]>");
        return;
    }

    let function = &args[1];

    match function.as_str(){
        "dijkstra" => {
            if args.len() != 5 {
                panic!("Wrong number of arguments for Dijkstra!\nCorrect arguments: <stop_a> <stop_b> <time>")
            }
            let stop_a_str = &args[2];
            let stop_b_str = &args[3];
            let time = MyTime::from(&args[4]);

            let mut now = Instant::now();
            let (unique_stops, graph) = read_records("connection_graph.csv".to_owned()).unwrap();
            println!("Building graph took {}\n", now.elapsed().as_millis());

            let stop_a = get_stop_by_name(&graph, &stop_a_str).expect(format!("Stop {} not found", stop_a_str).as_str());
            let stop_b = get_stop_by_name(&graph, &stop_b_str).expect(format!("Stop {} not found", stop_b_str).as_str());

            now = Instant::now();
            let path = dijkstra(stop_a, stop_b, time, &graph, &unique_stops);
            println!("Dijkstra finished, took {}ms", now.elapsed().as_millis());
            print_path(path);

        }
        "astar" => {
            if args.len() != 6 {
                panic!("Wrong number of arguments for Astar!\nCorrect arguments: <stop_a> <stop_b> <time> <criterion [t, p]>")
            }
            let stop_a_str = &args[2];
            let stop_b_str = &args[3];
            let time = MyTime::from(&args[4]);
            let limit_line_changes = args[5] == "p";

            let mut now = Instant::now();
            let (unique_stops, graph) = read_records("connection_graph.csv".to_owned()).unwrap();
            println!("Building graph took {}\n", now.elapsed().as_millis());

            let stop_a = get_stop_by_name(&graph, &stop_a_str).expect(format!("Stop {} not found", stop_a_str).as_str());
            let stop_b = get_stop_by_name(&graph, &stop_b_str).expect(format!("Stop {} not found", stop_b_str).as_str());

            now = Instant::now();
            let path = astar(stop_a, stop_b, time, &graph, &unique_stops, limit_line_changes);
            println!("Astar finished, took {}ms", now.elapsed().as_millis());
            print_path(path);
        }
        "tabu" => {
            if args.len() < 6 {
                panic!("Wrong number of arguments for Tabu!\nCorrect arguments: <stop_a> <time> <criterion [t, p]> <stop_list [bus stop names separated by commas]> <length_limit [optional]>")
            }
            let stop_a_str = &args[2];
            let time = MyTime::from(&args[3]);
            let limit_line_changes = args[4] == "p";
            let stop_list_str = &args[5];
            let limit = if args.len() > 6 {
                Some(args[6].parse::<u32>().unwrap())
            } else {
                None
            };
            let mut now = Instant::now();
            let (unique_stops, graph) = read_records("connection_graph.csv".to_owned()).unwrap();
            println!("Building graph took {}\n", now.elapsed().as_millis());

            let stop_a = get_stop_by_name(&graph, &stop_a_str).expect(format!("Stop {} not found", stop_a_str).as_str());

            let stop_list = {
                let split_str = stop_list_str.split(',');
                let mut stop_list: Vec<BusStop> = Vec::new();
                for stop in split_str.clone() {
                    let bus_stop = get_stop_by_name(&graph, stop)
                        .expect(format!("Stop {} not found", stop).as_str());
                    stop_list.push(bus_stop);
                }
                stop_list
            };

            now = Instant::now();
            let path = tabu_search(stop_a, stop_list, time, &graph, &unique_stops, limit_line_changes, 50, limit);
            println!("Tabu finished, took {}ms", now.elapsed().as_millis());
            print_path(path);

        }
        other => panic!("Function {other} not found!")
    }
}

#[test]
fn compare_dijkstra_and_astar() {
    let tested_stops = vec![
        (
            "Iwiny - rondo",
            "Hala Stulecia",
            MyTime::from("14:39".to_string()),
        ),
        (
            "Piastowska",
            "Zaolziańska",
            MyTime::from("12:00".to_string()),
        ),
    ];
    let (unique_stops, graph) = read_records("connection_graph.csv".to_owned()).unwrap();
    tested_stops.into_par_iter().for_each(|test_case| {
        let stop_a = get_stop_by_name(&graph, test_case.0).unwrap();
        let stop_b = get_stop_by_name(&graph, test_case.1).unwrap();
        let path_dijkstra = dijkstra(
            stop_a.clone(),
            stop_b.clone(),
            test_case.2,
            &graph,
            &unique_stops,
        );
        let path_astar = astar(stop_a, stop_b, test_case.2, &graph, &unique_stops, false);
        assert_eq!(path_dijkstra.len(), path_astar.len());
        for (idx, path_elem) in path_dijkstra.into_iter().enumerate() {
            assert_eq!(path_elem.0, path_astar[idx].0);
        }
    });
}
