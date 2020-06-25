import json

def save_json(train_data,filename):
#   Convert and ndarrays to list
    train_data_list = {}
    for page in train_data.keys():
        graph, nodes, words, y = train_data[page][0], train_data[page][1], train_data[page][2], train_data[page][3]
        graph_list = {}
        for w in graph.keys():
            graph_list[w] = list(graph[w])
        train_data_list[page] = [graph_list, nodes.tolist(), words.tolist(), y.tolist()]


    with open(filename, 'w') as outfile:
        json.dump(train_data_list, outfile)