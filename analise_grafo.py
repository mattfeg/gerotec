import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def carregar_grafo(file_path):
    # Carregar o grafo a partir do arquivo .gexf
    return nx.read_gexf(file_path)

def calcular_componentes_conectadas(G):
    """
    Calcula as componentes conectadas no grafo bipartido original.
    
    Args:
    - G (nx.DiGraph): Grafo bipartido original.
    
    Returns:
    - dict: Dicionário com o número de componentes fortemente e fracamente conectadas.
    """
    componentes_fortemente_conectadas = list(nx.strongly_connected_components(G))
    componentes_fracamente_conectadas = list(nx.weakly_connected_components(G))
    
    return {
        "numero_componentes_fortemente_conectadas": len(componentes_fortemente_conectadas),
        "tamanho_medio_componente_fortemente_conectada": sum(len(c) for c in componentes_fortemente_conectadas) / len(componentes_fortemente_conectadas),
        "numero_componentes_fracamente_conectadas": len(componentes_fracamente_conectadas),
        "tamanho_medio_componente_fracamente_conectada": sum(len(c) for c in componentes_fracamente_conectadas) / len(componentes_fracamente_conectadas)
    }

def exibir_estatisticas_componentes(estatisticas_componentes):
    print("Estatísticas das Componentes Conectadas:")
    for chave, valor in estatisticas_componentes.items():
        print(f"{chave}: {valor}")

def listar_componentes_fracamente_conectadas(G):
    """
    Lista os nós em cada componente fracamente conectada.
    
    Args:
    - G (nx.DiGraph): Grafo bipartido original.
    
    Returns:
    - list: Lista de componentes fracamente conectadas, cada uma contendo os nós dessa componente.
    """
    componentes_fracamente_conectadas = list(nx.weakly_connected_components(G))
    return componentes_fracamente_conectadas

def plotar_wccs(G, wccs):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    
    # Desenhar o grafo
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos)
    
    # Desenhar os nós, colorindo cada WCC de uma cor diferente
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
    for i, wcc in enumerate(wccs):
        nx.draw_networkx_nodes(G, pos, nodelist=wcc, node_color=colors[i % len(colors)], label=f'WCC {i+1}')
    
    plt.legend()
    plt.show()

import folium

def plotar_wccs_folium(G, wccs, output_path='output/wccs_interativo.html'):
    # Coordenadas do centro do estado do Ceará
    centro_latitude = -5.2
    centro_longitude = -39.5
    
    # Coordenadas dos limites do estado do Ceará
    bounds = [[-7.5, -41.5], [-2.5, -37.0]]

    # Criar o mapa com o centro e zoom inicial ajustado
    m = folium.Map(location=[centro_latitude, centro_longitude], zoom_start=7)

    # Cores para diferentes componentes
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'pink', 'yellow', 'brown', 'gray']

    # Adicionar os nós das WCCs
    for i, wcc in enumerate(wccs):
        color = colors[i % len(colors)]
        for node in wcc:
            lat = G.nodes[node]['latitude']
            lon = G.nodes[node]['longitude']
            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                popup=node,
                color=color,
                fill=True,
                fill_color=color
            ).add_to(m)

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
            opacity=0.5
        ).add_to(m)

    # Ajustar os limites do mapa para cobrir o estado do Ceará
    m.fit_bounds(bounds)

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
    shutil.copy(output_path,output_path)
    # Fornecer link para download
    return FileLink(output_path)


if __name__ == "__main__":
    # Caminho para o arquivo .gexf
    file_path = 'preExport.gexf'

    # Carregar o grafo
    grafo_pre = carregar_grafo(file_path)

    # Calcular componentes conectadas
    estatisticas_componentes = calcular_componentes_conectadas(grafo_pre)
    exibir_estatisticas_componentes(estatisticas_componentes)

    # Listar componentes fracamente conectadas
    wccs = listar_componentes_fracamente_conectadas(grafo_pre)
    for i, wcc in enumerate(wccs):
        print(f"Componente Fracamente Conectada {i+1}: {wcc}")

    # Plotar as WCCs em um mapa interativo
    output_path = 'output/wccs_interativo.html'
    mapa_interativo = plotar_wccs_folium(grafo_pre, wccs, output_path)

    # Fornecer link para download do arquivo HTML
    download_link = baixar_arquivo(output_path)
    print(download_link)
