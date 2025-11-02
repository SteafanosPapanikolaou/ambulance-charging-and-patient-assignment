import pandas as pd
import osmnx as ox
import networkx as nx
import joblib
import numpy as np
import random

def get_route_edge_attributes(G, path, attr):
    values = []
    for u, v in zip(path[:-1], path[1:]):
        data = G.get_edge_data(u, v)
        # For MultiDiGraph, pick the first available edge
        if isinstance(data, dict):
            edge_data = list(data.values())[0]
        else:
            edge_data = data
        values.append(edge_data.get(attr, 0))
    return values

df = pd.read_excel("filepath")
df.columns = ['Call_Date', 'emergency', 'Valid', 'Pick_Up_Adress', 'Municipality',
              'Sector', 'Destination', 'Sex', 'Re-arrange', 'Arrival_time',
              'Left_time', 'End_time_event']
df['emergency'] = df['emergency'].replace({'Επείγον': True, 'Μη Επείγον': False})
df['Valid'] = df['Valid'].replace({'Έγκυρο': True, 'Άκυρο': False})
df = df[df['Valid'] != False]

df['Call_Date'] = pd.to_datetime(df['Call_Date'])
df['year'] = df['Call_Date'].dt.year
df['month'] = df['Call_Date'].dt.month
df['day'] = df['Call_Date'].dt.day
df['hour'] = df['Call_Date'].dt.hour
df['minute'] = df['Call_Date'].dt.minute

to_replace = ['.*ΠΑΠΑΓΕΩΡΓΙΟΥ.*', '.*424 Σ.Ν.Θ.*', '.*ΑΧΕΠΑ.*', '.*ΙΠΠΟΚΡΑΤΕΙΟ.*',
              '.*ΑΓΙΟΣ ΠΑΥΛΟΣ.*', '.*ΠΑΠΑΝΙΚΟΛΑΟΥ.*', '.*ΑΓΙΟΣ ΔΗΜΗΤΡΙΟΣ.*',
              '.*ΓΕΝΝΗΜΑΤΑΣ.*', '.*ΘΕΑΓΕΝΕΙΟ.*', '.*ΔΗΜ. ΨΥΧ. ΘΕΣ.*', '.*ΜΑΔΥΤΟΥ.*',
              '.*ΔΕΡΜΑΤΟΛΟΓΙΚΟ.*', '.*ΜΗΧΑΝΙΩΝΑΣ.*', '.*ΚΑΛΛΙΚΡΑΤΕΙΑΣ.*', '.*ΛΑΓΚΑΔΑ.*',
              '.*ΔΙΑΒΑΤΩΝ.*', '.*ΚΟΥΦΑΛΙΩΝ.*', '.*ΧΑΛΑΣΤΡΑΣ.*', '.*ΘΕΡΜΗΣ.*', '.*ΣΟΧΟΥ.*',]

replaced_with = ['Νοσοκομειο Παπαγεωργίου', '424 Στρατιωτικό Νοσοκομείο, Πολίχνη', 'Νοσοκομείο ΑΧΕΠΑ', 'Νοσοκομείο Ιπποκράτειο, Θεσσαλονίκη',
                 'Νοσοκομείο Άγιος Παύλος', 'Νοσοκομείο Παπανικολάου, Παπανικολάου', 'Νοσοκομείο Άγιος Δημήτριος',
                 'Νοσοκομείο Γεννηματάς', 'Νοσοκομείο Θεαγένειο', 'Ψυχιατρικό Νοσοκομείο, Θεσσαλονίκης', 'Κέντρο Υγείας, Βόλβης',
                 'Λοιμοδών', 'Κέντρο Υγείας Μηχανιώνας', 'Κέντρο Υγείας Καλλικράτειας', 'Κέντρο Υγείας Λαγκαδά',
                 'Κέντρο Υγείας Διαβατών', 'Κέντρο Υγείας Κουφαλιών', 'Κέντρο Υγείας Χαλάστρας', 'Κέντρο Υγείας Θέρμης', 'Κέντρο Υγείας Σοχού']

df['Destination'] = df['Destination'].replace(to_replace, replaced_with, regex=True)

df_2024 = df[df['year'] == 2024]
df_02_2024 = df_2024[df_2024['month'] == 2]
df_02_2024 = df_02_2024.reset_index()
df_02_2024 = df_02_2024.drop(columns=['index', 'Call_Date', 'Pick_Up_Adress',
                                      'Valid', 'Sector', 'Sex', 'Re-arrange'])

df_02_2024.dropna(subset=['Municipality'], inplace=True)

df_municipalities_dict = pd.DataFrame({
    'neighboor': ["ΘΕΡΜΑΙΚΟΥ",
                  "ΘΕΣΣΑΛΟΝΙΚΗΣ",
                  "ΧΑΛΚΗΔΟΝΟΣ",
                  "ΣΥΚΕΩΝ",
                  "ΣΤΑΥΡΟΥΠΟΛΗΣ",
                  "ΠΥΛΑΙΑΣ",
                  "ΕΥΟΣΜΟΥ",
                  "ΧΑΛΑΣΤΡΑΣ",
                  "ΚΑΛΑΜΑΡΙΑΣ",
                  "ΩΡΑΙΟΚΑΣΤΡΟΥ",
                  "ΚΟΥΦΑΛΙΩΝ",
                  "ΠΟΛΙΧΝΗΣ",
                  "ΑΓΙΟΥ ΠΑΥΛΟΥ",
                  "ΠΑΝΟΡΑΜΑΤΟΣ",
                  "ΝΕΑΠΟΛΗΣ",
                  "ΘΕΡΜΗΣ",
                  "ΛΑΓΚΑΔΑ",
                  "ΕΧΕΔΩΡΟΥ",
                  "ΕΛΕΥΘΕΡΙΟΥ - ΚΟΡΔΕΛΙΟΥ",
                  "ΜΗΧΑΝΙΩΝΑΣ",
                  "ΜΕΝΕΜΕΝΗΣ",
                  "ΠΕΥΚΩΝ",
                  "ΑΞΙΟΥ",
                  "ΑΜΠΕΛΟΚΗΠΩΝ",
                  "ΜΙΚΡΑΣ",
                  "ΕΥΚΑΡΠΙΑΣ",
                  "ΑΓΙΟΥ ΓΕΩΡΓΙΟΥ",
                  "ΑΠΟΛΩΝΙΑΣ",
                  "ΤΡΙΑΝΔΡΙΑΣ",
                  "ΛΑΧΑΝΑ",
                  "ΧΟΡΤΙΑΤΗ",
                  "ΜΑΔΥΤΟΥ",
                  "ΣΟΧΟΥ",
                  "ΒΑΣΙΛΙΚΩΝ",
                  "ΚΟΡΩΝΕΙΑΣ",
                  "ΕΠΑΝΟΜΗΣ",
                  "ΑΓΙΟΥ ΑΘΑΝΑΣΙΟΥ",
                  "ΑΣΣΗΡΟΥ",
                  "ΚΑΛΛΙΚΡΑΤΕΙΑΣ",
                  "ΑΡΕΘΟΥΣΑΣ",
                  "ΡΕΝΤΙΝΑΣ",
                  "ΑΠΟΛΛΩΝΙΑΣ"],
    'municipality': ["Δήμος Θερμαϊκού, Ελλάδα",
                     "Δήμος Θεσσαλονίκης",
                     "Δήμος Χαλκηδόνας",
                     "Δήμος Νεάπολης Συκεών",
                     "Δήμος Σταυρούπολης",
                     "Δήμος Πυλαίας",
                     "Δήμος Ευόσμου",
                     "Δήμος Δέλτα",
                     "Δήμος Καλαμαριάς",
                     "Δήμος Ωραιοκάστρου",
                     "Δήμος Κουφαλίων",
                     "Δήμος Παύλου Μελά",
                     "Δήμος Νεάπολης Συκεών",
                     "Δήμος Πυλαίας",
                     "Δήμος Νεάπολης Συκεών",
                     "Δήμος Θέρμης",
                     "Δήμος Λαγκαδά",
                     "Δήμος Δέλτα",
                     "Δήμος Ευόσμου",
                     "Δήμος Θερμαϊκού, Ελλάδα",
                     "Δήμος Αμπελοκήπων Μενεμένης",
                     "Δήμος Νεάπολης Συκεών",
                     "Δήμος Δέλτα",
                     "Δήμος Αμπελοκήπων Μενεμένης",
                     "Δήμος Καλαμαριάς",
                     "Δήμος Παύλου Μελά",
                     "Δήμος Βόλβης",
                     "Δήμος Βόλβης",
                     "Thessaloniki Municipality, Greece",
                     "Δήμος Λαγκαδά",
                     "Δήμος Πυλαίας",
                     "Δήμος Βόλβης",
                     "Δήμος Λαγκαδά",
                     "Δήμος Θέρμης",
                     "Δήμος Λαγκαδά",
                     "Δήμος Θερμαϊκού, Ελλάδα",
                     "Δήμος Χαλκηδόνας",
                     "Δήμος Λαγκαδά",
                     "Δήμος Προποντίδας",
                     "Δήμος Βόλβης",
                     "Δήμος Βόλβης",
                     "Δήμος Βόλβης",]
})

# Initialize an empty list to store the data
datanodes = []
unique_municipalities = df_municipalities_dict['municipality'].unique()

for municipality in unique_municipalities:
    try:
        print(f"Processing: {municipality}")
        # Retrieve the street network for the municipality
        G = ox.graph_from_place(municipality, network_type="drive")
        # Extract the list of nodes
        nodeslist = list(G.nodes)
        # Append the data to the list
        datanodes.append({"name": municipality, "nodeslist": nodeslist})
    except Exception as e:
        print(f"Error processing {municipality}: {e}")

# Create a DataFrame from the collected data
df_municipalities = pd.DataFrame(datanodes)

# Initialize an empty list to store the data
datanodes = []
unique_hospitals = df_02_2024['Destination'].unique()

# Download and combine graphs in one go
print("Downloading and building road network...")
G = ox.graph_from_place(["Thessaloniki Regional Unit, Greece", "Chalkidiki Regional Unit, Greece"],
                        network_type="drive")

# Add speed and travel time
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)

print("Graph ready.")


for hospital in unique_hospitals:
    try:
        print(f"Processing: {hospital}")
        # Geocode the hospital name to get coordinates
        location_point = ox.geocode(hospital)
        # Get the closest node in the network
        node = ox.distance.nearest_nodes(G, location_point[1], location_point[0])
        # Append the data to the list
        datanodes.append({"name": hospital, "nodeslist": node})
    except Exception as e:
        print(f"Error processing {hospital}: {e}")

# Create a DataFrame from the collected data
df_hospitals = pd.DataFrame(datanodes)

df_02_2024_data  = df_02_2024
df_02_2024_data['Event_node'] = ''
df_02_2024_data['Destination_node'] = ''
df_02_2024_data['Travel_Time'] = ''
df_02_2024_data['Length'] = ''
df_02_2024_data.reset_index(drop=True, inplace=True)

for i, row in df_02_2024_data.iterrows():
  df_02_2024_data.loc[i, "Destination_node"] = df_hospitals[df_hospitals['name']==row['Destination']]['nodeslist'].values[0]
  hospital = df_02_2024_data.loc[i, "Destination_node"]

  temp_municipality = df_municipalities_dict[df_municipalities_dict['neighboor'] == row['Municipality']]['municipality'].values[0]
  municicaplity_nodes = df_municipalities[df_municipalities['name'] == temp_municipality]['nodeslist'].values[0]
  ep = random.choice(municicaplity_nodes)

  shortest_path = None
  k = 0
  while shortest_path is None:
    ep = random.choice(municicaplity_nodes)
    shortest_path = ox.shortest_path(G, hospital, ep, weight="travel_time")
    if k > 0:
      print(f"Try time: {k}")
    k = k +1

  df_02_2024_data.loc[i, 'Event_node'] = ep

  print(i)
  print(shortest_path)

  lengths = get_route_edge_attributes(G, shortest_path, "length")
  df_02_2024_data.loc[i, 'Length'] = sum(lengths)

  travel_times = get_route_edge_attributes(G, shortest_path, "travel_time")
  df_02_2024_data.loc[i, 'Travel_Time'] = sum(travel_times)

  # df_02_2024_data.loc[i, 'Travel_Time'] = int(sum(ox.utils_graph.get_route_edge_attributes(G, shortest_path, "travel_time"))/60)

  # df_02_2024_data.loc[i, 'Length'] = sum(ox.utils_graph.get_route_edge_attributes(G, shortest_path, "length"))

df_02_2024_data.to_csv('filepath', index=False)
