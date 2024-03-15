import igraph
import csv
import numpy
from PIL import Image
from matplotlib import pyplot, image

def print_list(list: list):
    for idx, elem in enumerate(list):
        print(f"{idx}: {elem}")


class BusStop:
    def __init__(self, name: str, lat: str, lon: str):
        self.name = name
        self.id = 0

        self.lat: float = float(lat)
        self.lon: float = float(lon)

    def set_id(self, new_id: int):
        self.id = new_id

    def __str__(self) -> str:
        return f"{self.name}: {self.lat}, {self.lon}"        

    def __eq__(self, other) -> bool:
        return self.lat == other.lat and self.lon == other.lon

    def __hash__(self) -> int:
        return hash(self.lat + self.lon)

class CustomTime:
    def __init__(self, time_str: str):
        split_str = time_str.split(":")
        self.hour = int(split_str[0])
        self.minute = int(split_str[1])

    def to_minutes(self) -> int :
        return self.hour * 60 + self.minute

    def __sub__(self, other) -> int:
        return self.to_minutes() - other.to_minutes()

file = open("connection_graph.csv", "r")
reader = csv.DictReader(file, delimiter = ",")

unique_stops = set([])

graph = igraph.Graph()

stop_id = 0
for idx, row in enumerate(reader):

    start_stop = BusStop(row["start_stop"], row["start_stop_lat"], row["start_stop_lon"])
    end_stop = BusStop(row["end_stop"], row["end_stop_lat"], row["end_stop_lon"])

    if not(start_stop in unique_stops):
        start_stop.set_id(stop_id)
        graph.add_vertices(1)
        unique_stops.add(start_stop)
        stop_id += 1
    if not(end_stop in unique_stops):
        end_stop.set_id(stop_id)
        graph.add_vertices(1)
        unique_stops.add(end_stop)
        stop_id += 1

    start_time = CustomTime(row["departure_time"])
    end_time = CustomTime(row["arrival_time"])

    graph.add_edge(start_stop.id, end_stop.id, time = end_time - start_time)
    if(idx % 10000 == 0):
        print(f"Row {idx}")



