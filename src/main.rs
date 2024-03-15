use rayon::prelude::*;
use serde::{Deserialize, Deserializer};
use std::{
    cmp::{Eq, PartialEq},
    collections::HashSet,
    convert::From,
    error::Error,
    fs::File,
    hash::{Hash, Hasher},
};

use graphlib::{Graph, VertexId};

#[derive(Debug, Clone)]
struct TransportRecord {
    id: i32,
    company: String,
    line: String,
    departure_time: MyTime,
    arrival_time: MyTime,
    start_stop: BusStop,
    end_stop: BusStop,
    weighed_time: Option<f32>,
}

impl From<TransportRecordSerial> for TransportRecord {
    fn from(value: TransportRecordSerial) -> Self {
        let mut start_lat_split = value.start_stop_lat.split('.');
        let mut start_lon_split = value.start_stop_lon.split('.');
        let mut end_lat_split = value.end_stop_lat.split('.');
        let mut end_lon_split = value.end_stop_lon.split('.');
        return TransportRecord {
            id: value.id,
            company: value.company,
            line: value.line,
            departure_time: value.departure_time,
            arrival_time: value.arrival_time,
            start_stop: BusStop {
                id: None,
                name: value.start_stop,
                coords: Coords {
                    lat_whole: start_lat_split
                        .next()
                        .expect(format!("Error with start lat: {}", value.start_stop_lat).as_str())
                        .parse()
                        .unwrap(),
                    lat_frac: start_lat_split.next().unwrap().parse().unwrap(),
                    lon_whole: start_lon_split.next().unwrap().parse().unwrap(),
                    lon_frac: start_lon_split.next().unwrap().parse().unwrap(),
                },
            },
            end_stop: BusStop {
                id: None,
                name: value.end_stop,
                coords: Coords {
                    lat_whole: end_lat_split.next().unwrap().parse().unwrap(),
                    lat_frac: end_lat_split.next().unwrap().parse().unwrap(),
                    lon_whole: end_lon_split.next().unwrap().parse().unwrap(),
                    lon_frac: end_lon_split
                        .next()
                        .expect(format!("Error with end lon: {}", value.end_stop_lon).as_str())
                        .parse()
                        .expect(format!("Error with end lon: {}", value.end_stop_lon).as_str()),
                },
            },
            weighed_time: None,
        };
    }
}

impl TransportRecord {
    fn get_time_difference(&self) -> i32 {
        return self
            .arrival_time
            .time_difference_in_mins(&self.departure_time);
    }
}

#[derive(Debug, serde::Deserialize)]
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
    hour: i32,
    minute: i32,
}

impl MyTime {
    fn to_minutes(&self) -> i32 {
        return self.hour * 60 + self.minute;
    }

    fn time_difference_in_mins(&self, other_time: &MyTime) -> i32 {
        return self.to_minutes() - other_time.to_minutes();
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
        return MyTime {
            hour: str_split[0].parse().unwrap(),
            minute: str_split[1].parse().unwrap(),
        };
    }
}

#[derive(Debug, Copy, Clone)]
struct Coords {
    lat_whole: u32,
    lat_frac: u64,
    lon_whole: u32,
    lon_frac: u64,
}

#[derive(Debug, Clone)]
struct BusStop {
    id: Option<VertexId>,
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

fn read_records(file_name: String) -> Result<Vec<TransportRecord>, Box<dyn Error>> {
    let file = File::open(file_name)?;
    let mut rdr = csv::Reader::from_reader(file);
    let mut records: Vec<TransportRecord> = Vec::with_capacity(1000000);
    let mut max_time_difference = 0;
    for (idx, result) in rdr.deserialize::<TransportRecordSerial>().enumerate() {
        let fancy_record: TransportRecord = result.unwrap().into();
        let new_difference = fancy_record.get_time_difference();
        if new_difference > max_time_difference {
            max_time_difference = new_difference;
            /*println!(
                "New difference found: {}\nBus:{} - {}\nStops: {:?} - {:?}",
                new_difference,
                fancy_record.line,
                fancy_record.company,
                fancy_record.start_stop.name,
                fancy_record.end_stop.name
            );*/
        }

        records.push(fancy_record);

        if idx % 100000 == 0 {
            println!("Deserialized {} records", idx);
        }
    }

    //normalize weights

    println!("Max time difference found: {}", max_time_difference);

    for idx in 0..records.len() {
        let weighed_difference =
            records[idx].get_time_difference() as f32 / max_time_difference as f32;
        records[idx].weighed_time = Some(weighed_difference);
    }

    return Ok(records);
}

fn build_graph(records: Vec<TransportRecord>) -> (HashSet<BusStop>, Graph<BusStop>) {
    let mut unique_stops: HashSet<BusStop> = HashSet::with_capacity(10000);
    let mut new_graph = Graph::with_capacity(10000);
    for (idx, record) in records.into_iter().enumerate() {
        let start_stop = record.start_stop;
        let end_stop = record.end_stop;

        let mut start_id: Option<VertexId> = None;
        let mut end_id: Option<VertexId> = None;

        // add new bus stops to set, get their ids in graph
        if !unique_stops.contains(&start_stop) {
            let _insert = start_id.insert(new_graph.add_vertex(start_stop.clone()));
            unique_stops.insert(BusStop {
                id: start_id,
                ..start_stop
            });
        } else {
            start_id = unique_stops
                .get(&start_stop)
                .expect("Could not find start stop in unique stops set")
                .id;
        }

        if !unique_stops.contains(&end_stop) {
            let _insert = end_id.insert(new_graph.add_vertex(end_stop.clone()));
            unique_stops.insert(BusStop {
                id: end_id,
                ..end_stop
            });
        } else {
            end_id = unique_stops
                .get(&end_stop)
                .expect("Could not find end stop in unique stops set")
                .id;
        }

        // add edge between stops to graph
        new_graph
            .add_edge_with_weight(
                &start_id.expect("Start stop id not found"),
                &end_id.expect("End stop id not found"),
                record.weighed_time.unwrap(),
            )
            .expect("faled adding edge");

        if idx % 100000 == 0 {
            println!("Added {} records to graph", idx);
        }
    }
    return (unique_stops, new_graph);
}

fn zad_1_a(stop_a: BusStop, stop_b: BusStop, beginning_time: MyTime, graph: Graph<BusStop>) {
    let stop_a_id = stop_a.id.unwrap();
    let stop_b_id = stop_b.id.unwrap();
    let correct_result = graph.dijkstra(&stop_a_id, &stop_b_id);
    println!("Correct result");
    for stop in correct_result {}
}

fn main() {
    let records = read_records("connection_graph.csv".to_owned()).unwrap();
    let (unique_stops, graph) = build_graph(records);
    println!(
        "Graph stats:\nVertex count: {}\nEdge count: {}",
        graph.vertex_count(),
        graph.edge_count()
    );
    println!("Unique bus stops count: {}", unique_stops.len());
}
