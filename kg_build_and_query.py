import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stdin.reconfigure(encoding="utf-8")

import streamlit as st
import streamlit.components.v1 as components

import re

import random

CODE_BUILD_KG = """

# Prepare for GraphStore

os.environ['NEBULA_USER'] = "root"
os.environ['NEBULA_PASSWORD'] = "nebula" # default password
os.environ['NEBULA_ADDRESS'] = "127.0.0.1:9669" # assumed we have NebulaGraph installed locally

space_name = "guardians"
edge_types, rel_prop_names = ["relationship"], ["relationship"] # default, could be omit if create from an empty kg
tags = ["entity"] # default, could be omit if create from an empty kg

graph_store = NebulaGraphStore(space_name=space_name, edge_types=edge_types, rel_prop_names=rel_prop_names, tags=tags)
storage_context = StorageContext.from_defaults(graph_store=graph_store)

# Download and Preprocess Data

from llama_index import download_loader

WikipediaReader = download_loader("WikipediaReader")

loader = WikipediaReader()

documents = loader.load_data(pages=['Guardians of the Galaxy Vol. 3'], auto_suggest=False)

# Build Knowledge Graph

kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    max_triplets_per_chunk=10,
    service_context=service_context,
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
    include_embeddings=True,
)

"""

CODE_NL2CYPHER_LANGCHAIN = """
## Langchain
# Doc: https://python.langchain.com/docs/modules/chains/additional/graph_nebula_qa

from langchain.chat_models import ChatOpenAI
from langchain.chains import NebulaGraphQAChain
from langchain.graphs import NebulaGraph

graph = NebulaGraph(
    space=space_name,
    username="root",
    password="nebula",
    address="127.0.0.1",
    port=9669,
    session_pool_size=30,
)

chain = NebulaGraphQAChain.from_llm(
    llm, graph=graph, verbose=True
)

chain.run(
    "Tell me about Peter Quill?",
)
"""

CODE_NL2CYPHER_LLAMAINDEX = """

## Llama Index
# Doc: https://gpt-index.readthedocs.io/en/latest/examples/query_engine/knowledge_graph_query_engine.html

from llama_index.query_engine import KnowledgeGraphQueryEngine

from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import NebulaGraphStore

nl2kg_query_engine = KnowledgeGraphQueryEngine(
    storage_context=storage_context,
    service_context=service_context,
    llm=llm,
    verbose=True,
)

response = nl2kg_query_engine.query(
    "Tell me about Peter Quill?",
)
"""


import os
import json
import openai
from llama_index.llms import AzureOpenAI
from langchain.embeddings import OpenAIEmbeddings
from llama_index import LangchainEmbedding
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    KnowledgeGraphIndex,
    LLMPredictor,
    ServiceContext,
)

from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import NebulaGraphStore

import logging
import sys

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

openai.api_type = "azure"
openai.api_base = st.secrets["OPENAI_API_BASE"]
# openai.api_version = "2022-12-01" azure gpt-3
openai.api_version = "2023-05-15"  # azure gpt-3.5 turbo
openai.api_key = st.secrets["OPENAI_API_KEY"]

llm = AzureOpenAI(
    engine=st.secrets["DEPLOYMENT_NAME"],
    temperature=0,
    model="gpt-35-turbo",
)
llm_predictor = LLMPredictor(llm=llm)

# You need to deploy your own embedding model as well as your own chat completion model
embedding_llm = LangchainEmbedding(
    OpenAIEmbeddings(
        model="text-embedding-ada-002",
        deployment=st.secrets["EMBEDDING_DEPLOYMENT_NAME"],
        openai_api_key=openai.api_key,
        openai_api_base=openai.api_base,
        openai_api_type=openai.api_type,
        openai_api_version=openai.api_version,
    ),
    embed_batch_size=1,
)

service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor,
    embed_model=embedding_llm,
)
os.environ["NEBULA_USER"] = st.secrets["graphd_user"]
os.environ["NEBULA_PASSWORD"] = st.secrets["graphd_password"]
os.environ[
    "NEBULA_ADDRESS"
] = f"{st.secrets['graphd_host']}:{st.secrets['graphd_port']}"

space_name = "guardians"
edge_types, rel_prop_names = ["relationship"], [
    "relationship"
]  # default, could be omit if create from an empty kg
tags = ["entity"]  # default, could be omit if create from an empty kg

graph_store = NebulaGraphStore(
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
)
storage_context = StorageContext.from_defaults(graph_store=graph_store)

from llama_index.query_engine import KnowledgeGraphQueryEngine

from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import NebulaGraphStore

nl2kg_query_engine = KnowledgeGraphQueryEngine(
    storage_context=storage_context,
    service_context=service_context,
    llm=llm,
    verbose=True,
)


def cypher_to_all_paths(query):
    # Find the MATCH and RETURN parts
    match_parts = re.findall(r"(MATCH .+?(?=MATCH|$))", query, re.I | re.S)
    return_part = re.search(r"RETURN .+", query).group()

    modified_matches = []
    path_ids = []

    # Go through each MATCH part
    for i, part in enumerate(match_parts):
        path_id = f"path_{i}"
        path_ids.append(path_id)

        # Replace the MATCH keyword with "MATCH path_i = "
        modified_part = part.replace("MATCH ", f"MATCH {path_id} = ")
        modified_matches.append(modified_part)

    # Join the modified MATCH parts
    matches_string = " ".join(modified_matches)

    # Construct the new RETURN part
    return_string = f"RETURN {', '.join(path_ids)};"

    # Remove the old RETURN part from matches_string
    matches_string = matches_string.replace(return_part, "")

    # Combine everything
    modified_query = f"{matches_string}\n{return_string}"

    return modified_query


# write string to file
def result_to_df(result):
    from typing import Dict

    import pandas as pd

    columns = result.keys()
    d: Dict[str, list] = {}
    for col_num in range(result.col_size()):
        col_name = columns[col_num]
        col_list = result.column_values(col_name)
        d[col_name] = [x.cast() for x in col_list]
    return pd.DataFrame(d)


def render_pd_item(g, item):
    from nebula3.data.DataObject import Node, PathWrapper, Relationship

    if isinstance(item, Node):
        node_id = item.get_id().cast()
        tags = item.tags()  # list of strings
        props = dict()
        for tag in tags:
            props.update(item.properties(tag))
        g.add_node(node_id, label=node_id, title=str(props))
    elif isinstance(item, Relationship):
        src_id = item.start_vertex_id().cast()
        dst_id = item.end_vertex_id().cast()
        edge_name = item.edge_name()
        props = item.properties()
        # ensure start and end vertex exist in graph
        if not src_id in g.node_ids:
            g.add_node(src_id)
        if not dst_id in g.node_ids:
            g.add_node(dst_id)
        g.add_edge(src_id, dst_id, label=edge_name, title=str(props))
    elif isinstance(item, PathWrapper):
        for node in item.nodes():
            render_pd_item(g, node)
        for edge in item.relationships():
            render_pd_item(g, edge)
    elif isinstance(item, list):
        for it in item:
            render_pd_item(g, it)


def create_pyvis_graph(result_df):
    from pyvis.network import Network

    g = Network(
        notebook=True,
        directed=True,
        cdn_resources="in_line",
        height="500px",
        width="100%",
    )
    for _, row in result_df.iterrows():
        for item in row:
            render_pd_item(g, item)
    g.repulsion(
        node_distance=100,
        central_gravity=0.2,
        spring_length=200,
        spring_strength=0.05,
        damping=0.09,
    )
    return g


def query_nebulagraph(
    query,
    space_name=space_name,
    address=st.secrets["graphd_host"],
    port=9669,
    user=st.secrets["graphd_user"],
    password=st.secrets["graphd_password"],
):
    from nebula3.Config import SessionPoolConfig
    from nebula3.gclient.net.SessionPool import SessionPool

    config = SessionPoolConfig()
    session_pool = SessionPool(user, password, space_name, [(address, port)])
    session_pool.init(config)
    return session_pool.execute(query)


st.title("Demo: Knowledge Graph Build and Query with LLM")

(
    tab_code_kg,
    tab_notebook,
    tab_graph_view,
    tab_cypher,
    tab_nl2cypher,
    tab_code_nl2cypher,
) = st.tabs(
    [
        "Code: Build KG",
        "Full Notebook",
        "Graph View",
        "Query",
        "Natural Language to Cypher",
        "Code: NL2Cypher",
    ]
)

with tab_code_kg:
    st.write(
        "With a few lines of code, we can build a knowledge graph with LLM, LlamaIndex and NebulaGraph."
    )
    st.write(
        "See full notebook for more details and try Graph Visualizations, Query, and Natural Language to Cypher by clicking on the tabs on the right."
    )
    st.code(body=CODE_BUILD_KG, language="python")

with tab_notebook:
    st.write("> Full Notebook")
    st.markdown(
        """

This is the full notebook to demonstrate how to:

- Extract from data sources and build a knowledge graph with LLM and Llama Index, NebulaGraph in 3 lines of code
- Query the Knowledge Graph with nGQL and visualize the graph
- Query the knowledge graph with natural language in 1 line of code(both Langchain and Llama Index)
                """
    )
    # link to download notebook
    st.markdown(
        """
[Download](https://www.siwei.io/demo-dumps/kg-llm/KG_Building.ipynb) the notebook.
"""
    )

    components.iframe(
        src="https://www.siwei.io/demo-dumps/kg-llm/KG_Building.html",
        height=2000,
        width=800,
        scrolling=True,
    )

with tab_graph_view:
    st.write(
        "> Sub-Graph View of the Knowledge Graph about [Guardians of the Galaxy Vol. 3](https://en.wikipedia.org/wiki/Guardians_of_the_Galaxy_Vol._3)"
    )
    components.iframe(
        src="https://www.siwei.io/demo-dumps/kg-llm/nebulagraph_draw_sample.html",
        height=500,
        scrolling=True,
    )

with tab_cypher:
    st.write("> Query Knowledge Graph in nGQL")
    query_string = st.text_input(
        label="Enter nGQL query string", value="MATCH ()-[e]->() RETURN e LIMIT 25"
    )
    if st.button("> execute"):
        # run query
        result = query_nebulagraph(query_string)

        # convert to pandas dataframe
        result_df = result_to_df(result)

        # display pd dataframe
        st.dataframe(result_df)

        # create pyvis graph
        g = create_pyvis_graph(result_df)

        # render with random file name
        import random

        graph_html = g.generate_html(f"graph_{random.randint(0, 1000)}.html")

        components.html(graph_html, height=500, scrolling=True)

with tab_nl2cypher:
    st.write("> Natural Language to Cypher")
    nl_query_string = st.text_input(
        label="Enter natural language query string", value="Tell me about Peter Quill?"
    )
    if st.button("Ask KG"):
        response = nl2kg_query_engine.query(nl_query_string)
        graph_query = list(response.metadata.values())[0]["graph_store_query"]
        graph_query = graph_query.replace("WHERE", "\n  WHERE").replace(
            "RETURN", "\nRETURN"
        )
        answer = str(response)
        st.write(f"*Answer*: {answer}")
        st.markdown(
            f"""
## Generated NebulaGraph Cypher Query
```cypher
{graph_query}
```
"""
        )
        st.write("## Rendered Graph")
        render_query = cypher_to_all_paths(graph_query)
        result = query_nebulagraph(render_query)
        result_df = result_to_df(result)

        # create pyvis graph
        g = create_pyvis_graph(result_df)

        # render with random file name
        graph_html = g.generate_html(f"graph_{random.randint(0, 1000)}.html")

        components.html(graph_html, height=500, scrolling=True)


with tab_code_nl2cypher:
    st.write(
        "> Natural Language to NebulaGraph Cypher Code with Langchain and Llama Index"
    )
    tab_langchain, tab_llamaindex = st.tabs(["Langchain", "Llama Index"])
    with tab_langchain:
        st.code(body=CODE_NL2CYPHER_LANGCHAIN, language="python")
    with tab_llamaindex:
        st.code(body=CODE_NL2CYPHER_LLAMAINDEX, language="python")

    st.markdown(
        """

## References
                
- [Langchain: NebulaGraphQAChain](https://python.langchain.com/docs/modules/chains/additional/graph_nebula_qa)
- [Llama Index: KnowledgeGraphQueryEngine](https://gpt-index.readthedocs.io/en/latest/examples/query_engine/knowledge_graph_query_engine.html)
"""
    )
