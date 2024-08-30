import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from community import community_louvain
import folium
import os
from copy import deepcopy

def carregar_dados(geograf_path, internacoes_path):
    """
    Carrega os dados de georreferenciamento e internações.
    
    Args:
    - geograf_path (str): Caminho para o arquivo de georreferenciamento (geograf.csv).
    - internacoes_path (str): Caminho para o arquivo de internações (pre.csv, durante.csv, pos.csv, etc.).
    
    Returns:
    - pd.DataFrame: Dados de georreferenciamento.
    - pd.DataFrame: Dados de internações.
    """
    geograf = pd.read_csv(geograf_path)
    internacoes = pd.read_csv(internacoes_path)
    return geograf, internacoes

def criar_grafo(geograf, internacoes):
    """
    Cria um grafo bipartido direcionado a partir dos dados de internações e georreferenciamento.
    
    Args:
    - geograf (pd.DataFrame): Dados de georreferenciamento.
    - internacoes (pd.DataFrame): Dados de internações.
    
    Returns:
    - nx.DiGraph: Grafo bipartido direcionado.
    """
    G = nx.DiGraph()

    # Adicionar nós de municipios e hospitais com atributos de latitude e longitude
    for _, row in internacoes.iterrows():
        municipio = row['MUNICIPIO']
        hospital = row['HOSPITAL']

        # Verificar e adicionar o nó do município
        if not G.has_node(municipio):
            municipio_data = geograf[geograf['Nome'] == municipio]
            if not municipio_data.empty:
                G.add_node(municipio, bipartite=0, latitude=municipio_data['Latitude'].values[0], longitude=municipio_data['Longitude'].values[0])
            else:
                print(f"Erro: Coordenadas não encontradas para o município {municipio}")

        # Verificar e adicionar o nó do hospital
        if not G.has_node(hospital):
            hospital_data = geograf[geograf['Nome'] == hospital]
            if not hospital_data.empty:
                G.add_node(hospital, bipartite=1, latitude=hospital_data['Latitude'].values[0], longitude=hospital_data['Longitude'].values[0])
            else:
                print(f"Erro: Coordenadas não encontradas para o hospital {hospital}")

        # Adicionar aresta com a frequência de internações
        if G.has_node(municipio) and G.has_node(hospital):
            if G.has_edge(municipio, hospital):
                G[municipio][hospital]['weight'] += 1
            else:
                G.add_edge(municipio, hospital, weight=1)

    return G

def estatisticas_grafo(G):
    """
    Calcula e imprime estatísticas do grafo.
    
    Args:
    - G (nx.DiGraph): Grafo bipartido direcionado.
    """
    num_nos = G.number_of_nodes()
    num_arestas = G.number_of_edges()
    graus = [grau for no, grau in G.degree()]
    grau_medio = np.mean(graus)
    
    densidade = nx.density(G)
    componentes_fracas = nx.number_weakly_connected_components(G)
    componentes_fortes = nx.number_strongly_connected_components(G)
   
    print("Quantidade de Nós: ", num_nos)
    print("Quantidade de Arestas: ", num_arestas)
    print("Grau médio: ", grau_medio)
    print("Densidade: ", densidade)
    print("Componentes Conectadas (Fracas): ", componentes_fracas)
    print("Componentes Conectadas (Fortes): ", componentes_fortes)
    print("Componentes Conectadas: ", nx.number_connected_components(nx.to_undirected(G)))
    
def estatisticas_grafo_proj(G):
    """
    Calcula e imprime estatísticas do grafo.
    
    Args:
    - G (nx.DiGraph): Grafo bipartido direcionado.
    """
    num_nos = G.number_of_nodes()
    num_arestas = G.number_of_edges()
    graus = [grau for no, grau in G.degree()]
    grau_medio = np.mean(graus)
    densidade = nx.density(G)
    coefClustering = nx.average_clustering(G,weight='weight')

    print("Quantidade de Nós: ", num_nos)
    print("Quantidade de Arestas: ", num_arestas)
    print("Grau médio: ", grau_medio)
    print("Densidade: ", densidade)
    print("Coeficiente de Clusterização: ", coefClustering)
    print("Componentes Conectadas: ", nx.number_connected_components(G))

def detectar_clusters(G):
    """
    Detecta e imprime clusters (comunidades) no grafo usando o algoritmo de Louvain,
    ordenando os nós de cada cluster em ordem decrescente de acordo com o peso.
    
    Args:
    - G (nx.DiGraph): Grafo bipartido direcionado.
    
    Returns:
    - dict: Dicionário com os nós como chaves e as comunidades como valores.
    """
    undirected_G = G.to_undirected()
    partition = community_louvain.best_partition(undirected_G)
    
    num_clusters = len(set(partition.values()))
    print("Número de Clusters (Comunidades): ", num_clusters)
    
    for cluster in set(partition.values()):
        nodesCluster = [node for node in partition if partition[node] == cluster]
        nodesCluster_sorted = sorted(nodesCluster, key=lambda node: G.nodes[node].get('weight', 0), reverse=True)
        print(f"Cluster {cluster} ({len(nodesCluster_sorted)}):")
        for node in nodesCluster_sorted:
            print(f"  Nó: {node}, Peso: {G.nodes[node].get('weight', 0)}")
    
    return partition

def plotar_grafo(G, partition=None):
    """
    Plots a directed bipartite graph on a georeferenced map, coloring nodes based on their communities.

    Args:
        G (nx.DiGraph): Directed bipartite graph.
        partition (dict, optional): Dictionary assigning communities to nodes.
    """

    # Create a GeoDataFrame for nodes
    nodes = []
    for node, data in G.nodes(data=True):
        nodes.append([node, data['latitude'], data['longitude'], data['bipartite'], partition[node] if partition else None])

    nodes_df = pd.DataFrame(nodes, columns=['name', 'latitude', 'longitude', 'bipartite', 'cluster'])
    gdf_nodes = gpd.GeoDataFrame(nodes_df, geometry=gpd.points_from_xy(nodes_df.longitude, nodes_df.latitude))

    # Create the map
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))

    # Plot municipalities and hospitals
    municipios = gdf_nodes[gdf_nodes['bipartite'] == 0]
    hospitais = gdf_nodes[gdf_nodes['bipartite'] == 1]

    if not municipios.empty:
        municipios.plot(ax=ax, colormap='tab20', column='cluster', markersize=50)
    if not hospitais.empty:
        hospitais.plot(ax=ax, colormap='tab20', column='cluster', markersize=50)

    # Plot edges
    for u, v, data in G.edges(data=True):
        u_lat = G.nodes[u]['latitude']
        u_lon = G.nodes[u]['longitude']
        v_lat = G.nodes[v]['latitude']
        v_lon = G.nodes[v]['longitude']

        ax.plot([u_lon, v_lon], [u_lat, v_lat], color='black', alpha=0.5)

    # # Add labels to nodes
    # for x, y, label in zip(gdf_nodes.geometry.x, gdf_nodes.geometry.y, gdf_nodes['name']):
    #     ax.text(x, y, label, fontsize=5, ha='right')

    # Add title, legend, and labels
    plt.title('Grafo Bipartido Direcionado Georreferenciado')
    plt.legend(title='Comunidades')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig('./output/Imagens/img1.png', format='png', transparent=True)
    plt.show()

def projetar_grafo_hospitais(G):
    """
    Cria uma projeção dos nós hospitais, onde a relação entre eles é a quantidade de nós municípios em comum no grafo inicial.
    
    Args:
    - G (nx.DiGraph): Grafo bipartido original.
    
    Returns:
    - nx.Graph: Grafo projetado dos hospitais.
    """
    hospitais = [n for n, d in G.nodes(data=True) if d['bipartite'] == 1]
    projecao = nx.Graph()
    hospitais_sem_coordenadas = []

    # Adicionar todos os hospitais à projeção
    for hospital in hospitais:
        projecao.add_node(hospital, bipartite=1)
        if hospital in G.nodes:
            try:
                projecao.nodes[hospital]['latitude'] = G.nodes[hospital]['latitude']
                projecao.nodes[hospital]['longitude'] = G.nodes[hospital]['longitude']
                # Adicionar o peso do nó do hospital
                projecao.nodes[hospital]['weight'] = sum(G[u][hospital]['weight'] for u in G.predecessors(hospital))
            except KeyError:
                hospitais_sem_coordenadas.append(hospital)
        else:
            hospitais_sem_coordenadas.append(hospital)

    # Adicionar arestas com base em municípios em comum
    for hospital1 in hospitais:
        for hospital2 in hospitais:
            if hospital1 != hospital2:
                municipios_comuns = set(G.predecessors(hospital1)).intersection(set(G.predecessors(hospital2)))
                if municipios_comuns:
                    projecao.add_edge(hospital1, hospital2, weight=len(municipios_comuns))

    if hospitais_sem_coordenadas:
        print(f"Erro: Coordenadas não encontradas para os seguintes hospitais na projeção: {set(hospitais_sem_coordenadas)}")

    return projecao

def plotar_projecao_hospitais(G):
    """
    Plota a projeção dos nós hospitais.
    
    Args:
    - G (nx.Graph): Grafo projetado dos hospitais.
    """
    pos = {node: (data['longitude'], data['latitude']) for node, data in G.nodes(data=True)}
    weights = [G.nodes[node]['weight'] for node in G.nodes()]
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]

    plt.figure(figsize=(12, 12))
    nx.draw(G, pos, with_labels=True, node_color='red', node_size=[w * 10 for w in weights], font_size=10, edge_color='gray', width=edge_weights)
    plt.title('Projeção dos Hospitais')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

def exibir_informacoes_nos(G):
    """
    Exibe informações de todos os nós do grafo.
    
    Args:
    - G (nx.DiGraph): Grafo bipartido direcionado.
    """
    print("Informações dos Nós:")
    for node, data in G.nodes(data=True):
        print(f"Nó: {node}")
        for key, value in data.items():
            print(f"  {key}: {value}")
        print()

def exibir_informacoes_arestas(G):
    """
    Exibe informações de todas as arestas do grafo.
    
    Args:
    - G (nx.DiGraph): Grafo bipartido direcionado.
    """
    print("Informações das Arestas:")
    for u, v, data in G.edges(data=True):
        print(f"Aresta: {u} -> {v}")
        for key, value in data.items():
            print(f"  {key}: {value}")
        print()

def plotar_grafo_folium(G, output_path='output/grafo_interativo.html'):
    """
    Plota o grafo bipartido direcionado em um mapa interativo usando folium.
    
    Args:
    - G (nx.DiGraph): Grafo bipartido direcionado.
    - output_path (str): Caminho para salvar o arquivo HTML do mapa interativo.
    
    Returns:
    - folium.Map: Mapa interativo gerado.
    """
    # Determinar os limites do mapa
    latitudes = [data['latitude'] for node, data in G.nodes(data=True)]
    longitudes = [data['longitude'] for node, data in G.nodes(data=True)]
    min_lat, max_lat = min(latitudes), max(latitudes)
    min_lon, max_lon = min(longitudes), max(longitudes)
    
    # Criar o mapa
    m = folium.Map(location=[(min_lat + max_lat) / 2, (min_lon + max_lon) / 2], zoom_start=7)

    # Adicionar as arestas
    for u, v, data in G.edges(data=True):
        u_lat = G.nodes[u]['latitude']
        u_lon = G.nodes[u]['longitude']
        v_lat = G.nodes[v]['latitude']
        v_lon = G.nodes[v]['longitude']
        
        folium.PolyLine(
            locations=[[u_lat, u_lon], [v_lat, v_lon]],
            color='gray',
            weight=2,
            opacity=0.5,
            tooltip=f"Peso: {data['weight']}"
        ).add_to(m)

    # Adicionar os nós
    for node, data in G.nodes(data=True):
        folium.CircleMarker(
            location=[data['latitude'], data['longitude']],
            radius=5,
            popup=node,
            color='blue' if data['bipartite'] == 0 else 'red',
            fill=True,
            fill_color='blue' if data['bipartite'] == 0 else 'red'
        ).add_to(m)

    # Coordenadas dos limites do estado do Ceará
    bounds = [[-7.5, -41.5], [-2.5, -37.0]]

    # Ajustar os limites do mapa
    m.fit_bounds(bounds)

    # Criar o diretório de saída, se não existir
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Salvar o mapa como HTML
    m.save(output_path)

    # Exibir o mapa
    return m

def baixar_arquivo(output_path):
    """
    Função para baixar o arquivo HTML do mapa interativo.
    
    Args:
    - output_path (str): Caminho para o arquivo HTML do mapa interativo.
    
    Returns:
    - None
    """
    import shutil
    from IPython.display import FileLink

    # Copiar o arquivo HTML para um diretório acessível para download
    shutil.copy(output_path, './output' + output_path)
    # Fornecer link para download
    return FileLink('./output' + output_path)

import os
import folium
import networkx as nx

def plotar_clusters_individualmente(G, partition, output_dir='output/clusters'):
    """
    Plota os clusters individualmente em mapas interativos usando folium.
    
    Args:
    - G (nx.DiGraph): Grafo bipartido direcionado.
    - partition (dict): Dicionário com os nós como chaves e as comunidades como valores.
    - output_dir (str): Diretório para salvar os arquivos HTML dos mapas interativos.
    
    Returns:
    - None
    """
    # Criar o diretório de saída, se não existir
    os.makedirs(output_dir, exist_ok=True)

    # Plotar cada cluster individualmente
    for cluster in set(partition.values()):
        nodes_in_cluster = [node for node in partition if partition[node] == cluster]
        cluster_map = folium.Map(location=[-5.2, -39.5], zoom_start=7)

        for node in nodes_in_cluster:
            lat = G.nodes[node]['latitude']
            lon = G.nodes[node]['longitude']
            color = 'blue' if G.nodes[node]['bipartite'] == 0 else 'red'
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                popup=node,
                color=color,
                fill=True,
                fill_color=color
            ).add_to(cluster_map)

        # Adicionar as arestas dentro do cluster
        for u, v, data in G.edges(data=True):
            if u in nodes_in_cluster and v in nodes_in_cluster:
                u_lat = G.nodes[u]['latitude']
                u_lon = G.nodes[u]['longitude']
                v_lat = G.nodes[v]['latitude']
                v_lon = G.nodes[v]['longitude']
                
                folium.PolyLine(
                    locations=[[u_lat, u_lon], [v_lat, v_lon]],
                    color='gray',
                    weight=2,
                    opacity=0.5,
                    tooltip=f"Peso: {data['weight']}"
                ).add_to(cluster_map)

        # Coordenadas dos limites do estado do Ceará
        bounds = [[-7.5, -41.5], [-2.5, -37.0]]
        cluster_map.fit_bounds(bounds)

        # Salvar o mapa do cluster como HTML
        cluster_map.save(os.path.join(output_dir, f'cluster_{cluster}.html'))

def encontrar_clusters_correspondentes(clusters_periodo1, clusters_periodo2):
    """
    Encontra os clusters correspondentes entre dois períodos.

    Args:
    - clusters_periodo1 (dict): Clusters do primeiro período.
    - clusters_periodo2 (dict): Clusters do segundo período.
    
    Returns:
    - dict: Dicionário com a correspondência de clusters entre os períodos.
    """
    correspondencia = {}
    
    for cluster1 in set(clusters_periodo1.values()):
        nos_cluster1 = {node for node in clusters_periodo1 if clusters_periodo1[node] == cluster1}
        melhor_correspondencia = None
        maior_intersecao = 0
        
        for cluster2 in set(clusters_periodo2.values()):
            nos_cluster2 = {node for node in clusters_periodo2 if clusters_periodo2[node] == cluster2}
            intersecao = len(nos_cluster1.intersection(nos_cluster2))
            
            if intersecao > maior_intersecao:
                melhor_correspondencia = cluster2
                maior_intersecao = intersecao
        
        correspondencia[cluster1] = melhor_correspondencia
    
    return correspondencia

def calcular_diferencas_clusters(G_periodo1, clusters_periodo1, G_periodo2, clusters_periodo2, correspondencia):
    """
    Calcula as diferenças entre clusters correspondentes de dois períodos.

    Args:
    - G_periodo1 (nx.DiGraph): Grafo do primeiro período.
    - clusters_periodo1 (dict): Clusters do primeiro período.
    - G_periodo2 (nx.DiGraph): Grafo do segundo período.
    - clusters_periodo2 (dict): Clusters do segundo período.
    - correspondencia (dict): Correspondência de clusters entre os períodos.

    Returns:
    - list: Lista de dicionários com as diferenças entre clusters correspondentes.
    """
    diferencas = []

    for cluster1, cluster2 in correspondencia.items():
        nos_cluster1 = {node for node in clusters_periodo1 if clusters_periodo1[node] == cluster1}
        nos_cluster2 = {node for node in clusters_periodo2 if clusters_periodo2[node] == cluster2}

        peso_total_cluster1 = sum(G_periodo1.nodes[node].get('weight', 0) for node in nos_cluster1)
        peso_total_cluster2 = sum(G_periodo2.nodes[node].get('weight', 0) for node in nos_cluster2)

        diferencas.append({
            'cluster_periodo1': cluster1,
            'cluster_periodo2': cluster2,
            'num_nos_cluster1': len(nos_cluster1),
            'num_nos_cluster2': len(nos_cluster2),
            'peso_total_cluster1': peso_total_cluster1,
            'peso_total_cluster2': peso_total_cluster2,
            'mudanca_num_nos': len(nos_cluster2) - len(nos_cluster1),
            'mudanca_peso_total': peso_total_cluster2 - peso_total_cluster1
        })

    return diferencas

def plotar_municipios_por_hospital(G, output_dir='output/hospitals'):
    """
    Gera mapas interativos mostrando os municípios conectados a cada hospital.
    
    Args:
    - G (nx.DiGraph): Grafo bipartido direcionado, onde os nós representam hospitais e municípios.
    - output_dir (str): Diretório para salvar os arquivos HTML dos mapas interativos.
    
    Returns:
    - None
    """
    # Criar o diretório de saída, se não existir
    os.makedirs(output_dir, exist_ok=True)

    # Iterar sobre cada nó que representa um hospital no grafo
    for hospital in [node for node in G.nodes if G.nodes[node]['bipartite'] == 1]:  # Assumindo bipartite=1 para hospitais
        hospital_map = folium.Map(location=[G.nodes[hospital]['latitude'], G.nodes[hospital]['longitude']], zoom_start=10)

        # Adicionar o marcador do hospital (estilo original)
        folium.Marker(
            location=[G.nodes[hospital]['latitude'], G.nodes[hospital]['longitude']],
            popup=hospital,
            icon=folium.Icon(color='red')
        ).add_to(hospital_map)

        # Iterar sobre todos os municípios e verificar quais estão conectados a este hospital
        for municipio in [node for node in G.nodes if G.nodes[node]['bipartite'] == 0]:  # Assumindo bipartite=0 para municípios
            if hospital in G.neighbors(municipio):
                lat = G.nodes[municipio]['latitude']
                lon = G.nodes[municipio]['longitude']

                folium.CircleMarker(
                    location=[lat, lon],
                    radius=5,
                    popup=municipio,
                    color='blue',
                    fill=True,
                    fill_color='blue'
                ).add_to(hospital_map)

                # Adicionar a linha que conecta o hospital ao município com tooltip mostrando o peso
                weight = G.edges[municipio, hospital]['weight']
                folium.PolyLine(
                    locations=[[G.nodes[hospital]['latitude'], G.nodes[hospital]['longitude']], [lat, lon]],
                    color='gray',
                    weight=2,  # Espessura da linha
                    opacity=0.5,
                    tooltip=f"Frequência: {weight}"  # Tooltip exibindo a frequência do deslocamento
                ).add_to(hospital_map)

        # Ajustar o mapa para os limites do Ceará
        bounds = [[-7.5, -41.5], [-2.5, -37.0]]
        hospital_map.fit_bounds(bounds)

        # Salvar o mapa com o nome do hospital
        hospital_map.save(os.path.join(output_dir, f'{hospital}.html'))

def plotar_grafo_por_macrorregiao(G, macrorregiao, municipios_macrorregiao, output_dir='output/macrorregioes'):
    """
    Plota um grafo contendo apenas os nós dos municípios de uma determinada macrorregião de saúde 
    e os hospitais que se ligam a esses municípios.
    
    Args:
    - G (nx.DiGraph): Grafo bipartido direcionado.
    - macrorregiao (str): Nome da macrorregião de saúde.
    - municipios_macrorregiao (list): Lista dos nomes dos municípios que pertencem à macrorregião.
    - output_dir (str): Diretório para salvar o arquivo HTML do mapa interativo.
    
    Returns:
    - folium.Map: Mapa interativo gerado ou None se não houver nós na macrorregião.
    """
    # Criar um subgrafo contendo apenas os nós dos municípios da macrorregião e os hospitais conectados a eles
    subgrafo = nx.DiGraph()

    for municipio in municipios_macrorregiao:
        if municipio in G.nodes:
            subgrafo.add_node(municipio, **G.nodes[municipio])
            for hospital in G.neighbors(municipio):
                if hospital not in subgrafo.nodes:
                    subgrafo.add_node(hospital, **G.nodes[hospital])
                subgrafo.add_edge(municipio, hospital, **G.edges[municipio, hospital])

    # Verificar se o subgrafo contém nós
    if not subgrafo.nodes:
        print(f"Nenhum nó encontrado para a macrorregião '{macrorregiao}'. Mapa não será gerado.")
        return None

    # Determinar os limites do mapa com base nos nós do subgrafo
    latitudes = [data['latitude'] for node, data in subgrafo.nodes(data=True)]
    longitudes = [data['longitude'] for node, data in subgrafo.nodes(data=True)]
    min_lat, max_lat = min(latitudes), max(latitudes)
    min_lon, max_lon = min(longitudes), max(longitudes)

    # Criar o mapa centrado na macrorregião
    m = folium.Map(location=[(min_lat + max_lat) / 2, (min_lon + max_lon) / 2], zoom_start=7)

    # Adicionar as arestas ao mapa
    for u, v, data in subgrafo.edges(data=True):
        u_lat = subgrafo.nodes[u]['latitude']
        u_lon = subgrafo.nodes[u]['longitude']
        v_lat = subgrafo.nodes[v]['latitude']
        v_lon = subgrafo.nodes[v]['longitude']
        
        folium.PolyLine(
            locations=[[u_lat, u_lon], [v_lat, v_lon]],
            color='gray',
            weight=2,
            opacity=0.5,
            tooltip=f"Frequência: {data['weight']}"
        ).add_to(m)

    # Adicionar os nós ao mapa
    for node, data in subgrafo.nodes(data=True):
        color = 'red' if data['bipartite'] == 1 else 'blue'  # Vermelho para hospitais, azul para municípios
        folium.CircleMarker(
            location=[data['latitude'], data['longitude']],
            radius=5,
            popup=node,
            color=color,
            fill=True,
            fill_color=color
        ).add_to(m)

    # Ajustar o mapa para os limites da macrorregião
    m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

    # Criar o diretório de saída, se não existir
    os.makedirs(output_dir, exist_ok=True)

    # Definir o nome do arquivo como o nome da macrorregião
    output_path = os.path.join(output_dir, f'{macrorregiao}.html')

    # Salvar o mapa como HTML
    m.save(output_path)

    # Exibir o mapa
    return m

