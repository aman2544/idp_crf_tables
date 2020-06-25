import numpy as np

def index_of(e,l):
    return np.where(l==e)

def build_graph(data,include_page_key = False):
    num_edges ={}
    for page in data.keys():
        num_edges[page] = 0
        graph,nodes,words,y =  data[page][0],data[page][1],data[page][2],data[page][3]
        for _w, neighbours in graph.items():
            num_edges[page] = num_edges[page] + len(neighbours)
    X1 =[]
    Y1=[]
    page_num = 0
    added_nodes =set()
    for page in data.keys():
        #print("Building graph Page",page_num)
        i=0
        n_edges = num_edges[page]
        edges = np.zeros((n_edges,2))

        graph,nodes,words,y =  data[page][0],data[page][1],data[page][2],data[page][3]
        if  isinstance(words,list):
            words = np.asarray(words)
        if  isinstance(nodes,list):
            nodes = np.asarray(nodes)
        if  isinstance(y,list):
            y = np.asarray(y)
        for _w, neighbours in graph.items():
            _from = index_of(_w,words)
            for n in neighbours:
                if n in added_nodes:
                    continue
                _to = index_of(n,words)
                edges[i,:] = np.asarray([_from,_to]).reshape(2)
                i = i+1
            added_nodes.add(_w)
        edges= edges.astype(int)
        if include_page_key:
            if edges.shape[0] > 0 and i > 0:
                X1.append((page,nodes,edges[0:i,:]))
                Y1.append(y)

            else:
                print("The page has no edges in the graph")
        else:
            if edges.shape[0] > 0 and i > 0:
                X1.append((nodes, edges[0:i, :]))
                Y1.append(y)
            else:
                print("The page has no edges in the graph")

        page_num = page_num + 1
    return X1,Y1