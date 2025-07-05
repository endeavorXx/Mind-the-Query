#!/usr/bin/env python3
"""
Natural Language to Cypher Query Generator using Gemini AI
Converted from Jupyter notebook to standalone Python script
"""

import json
import requests
import time as python_time
import re
import argparse
from dataclasses import dataclass
from typing import Dict, List
from random import sample
from neo4j import GraphDatabase
import google.generativeai as genai
from dotenv import load_dotenv
# Import local utility modules
from utils.utilities import *
from utils.neo4j_conn import *
from utils.neo4j_schema import *
from utils.graph_utils import *
from data_manager import DataManager
import os

print("Loading environment variables...")
load_dotenv()

@dataclass
class Config:
    neo4j_uri: str = os.getenv("NEO4J_URI")
    neo4j_user: str = "neo4j"
    neo4j_password: str = os.getenv("NEO4J_PASSWORD")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY")
    database: str = "neo4j"


class QuestionGenerator:
    def __init__(self, config: Config, api_key: str, temperature: float):
        self.model_name = "gemini-2.0-flash"
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.model_name)
        self.system_instruction = "You are an experienced Neo4j Cypher developer and a helpful assistant!"
        self.config = genai.GenerationConfig(temperature=temperature)
        print("Gemini model initialized with temperature:", temperature)
        
    def create_prompt(self, schema, node_values, rels_values, category, query_type: str, in_context, num_questions: int) -> str:
        """Create a formatted prompt for question generation"""
        
        # Prompt v2.5 with few-shot examples + guided instructions
        return f"""
Your task is to generate {num_questions} questions that are directly related to a specific graph schema in Neo4j. Each question should target distinct aspects of the schema, such as relationships between nodes, properties of nodes, or characteristics of node types. 
Imagine you are a user at a company that needs to present all the types of questions that the graph can answer.
You have to be very diligent at your job.
The goal of these questions is to create a dataset for training AI models to convert natural language queries into Cypher queries effectively.
Task - 
- Generate {num_questions} Questions and corresponding cypher statement from the following graph schema : {schema}.\n 
- Generate logical natural language questions based on the schema, avoid ambiguous question which can be interpreted in multiple ways or does not have a straightforward answer. For example, avoid asking, "What is related to this?" without specifying the node type or relationship. 
- The questions should be diverse and vary in increasing complexity.
- These questions should target a specific query type category {category} which has following description {query_type}. Here are some examples for the the mentioned query type {in_context}.
- It is vital that the database contains information that can answer the question. Don't use any not provided node and relationship values. \n provided node values : {node_values} \n provided relationship values : {rels_values}. 
- while making cypher and natural language questions, Be careful while inducing node values and relationship values. They should match letter by letter. since, they will be directly used in the query.
- while making cypher and natural language questions, Be careful while using Node and relationship names. They should match letter by letter.
- while making cypher and natural language questions, Be careful while using Node and relationship properties. They should match letter by letter.
- Always keep in mind the data type of attributes for the nodes and relationship. For example for simple aggregation or complex aggregation type queries, It makes no sense to ask for average of Descriptions which has "STRING" datatype.
- for every question you have to provide a reason why that question belongs to that specific query type. Therefore you have to give {num_questions} questions and their reasoning why it belong to that particular category of query type. 
- Note that while constructing cyphers, you can not use aggregator inside an aggregator, for example AVG(COUNT()), it's not valid.
- While making queries involving dates, Mae sure to follow these formats and principles provided in the example -
(i) When comparing dates, convert the date property using a generic date string format (e.g., date(dob) = date("YYYY-MM-DD")).
(ii) To extract date components, use the dot notation (e.g., player.dob.year, player.dob.month, player.dob.day) instead of using functions like year(player.dob).
(iii) Some properties can be NULL - check for not null in queries.
Also, do not ask questions that there is no way to answer based on the schema or provided example values. 
Find good natural language questions that will test the capabilities of graph answering.
put all the questions in a list, each question with reasoning and it's cypher with reasoning must be enclosed in a dictionary !!
[
{{
"NL Question": # your_question,
"Reason for NL": # Provide reason why it is correct logically, 
"Cypher": # Cypher statement corresponding to that question, 
"Reason for cypher": # Provide reason why your cypher is correct logically for the natural language question
}}, 
{{
"NL Question": # your_question,
"Reason for NL": # Provide reason why it is correct logically, 
"Cypher": # Cypher statement corresponding to that question, 
"Reason for cypher": # Provide reason why your cypher is correct logically for the natural language question
}}, 
so on...
]
Just write the json file !!
"""
        
    def clean_gemini_response(self, response_text: str) -> str:
        """Clean the Gemini API response text to extract valid JSON."""
        
        # Find content between ```json and ``` markers
        json_pattern = r'```json\s*(.*?)\s*```'
        matches = re.findall(json_pattern, response_text, re.DOTALL)
        
        if matches:
            # Take the first match (assuming single JSON block)
            json_str = matches[0].strip()
            return json_str
        return ""
        
    def post_request(self, message) -> Dict:
        """Send a POST request to the Gemini API"""
        contents = [message]
        response = self.model.generate_content(
            contents=contents,
            generation_config=self.config,
        )
        outcome = response.text
        print("=============================== RESPONSE ===============================")
        print(outcome)
        clean_text = self.clean_gemini_response(outcome)
        print("Cleaned Response: ", clean_text)
        try:
            outcome = json.loads(clean_text)
        except json.JSONDecodeError:
            print("Failed to decode JSON response")
            return {}
        return outcome
    
    def generate_questions(self, schema: Dict, node_values, rels_values, query_type: str, category, in_context_samples, num_questions: int) -> List[Dict]:
        """Generate questions using direct API calls to Gemini"""
        print("Creating Prompt")
        prompt = self.create_prompt(schema, node_values, rels_values, category, query_type, in_context_samples[category], num_questions=num_questions)
        print("\n=================================================  PROMPT  ======================================================\n")
        print(prompt)
        outcome = self.post_request(prompt)
        return outcome


def sample_items(items: list, num_samples: int) -> list:
    """
    Sample random items from a list
    
    Args:
        items (list): List to sample from
        num_samples (int): Number of samples to take
        
    Returns:
        list: List of sampled items
    """
    return sample(items, min(num_samples, len(items)))


def generate_nodes_rels_instances(nodes, relationships, gutils, node_instances_size, rels_instances_size):
    """Generate node and relationship instances from the graph"""
    # Extract the node instances from the graph
    node_instances = gutils.extract_node_instances(nodes, node_instances_size)
    # Extract the relationship instances from the graph
    rels_instances = gutils.extract_multiple_relationships_instances(relationships, rels_instances_size)
    
    # Serialize extracted neo4j.time data - for saving to json files
    nodes_instances_serialized = serialize_nodes_data(node_instances)
    rels_instances_serialized = serialize_relationships_data(rels_instances)
    sampled_node_set, sampled_rels_set = "", ""
    
    for i in nodes_instances_serialized:
        data = str(sample_items(i, 1))
        data = data.replace("[", "")
        data = data.replace("]", "")
        sampled_node_set += data
        sampled_node_set += "\n"
        
    for i in rels_instances_serialized:
        data = str(sample_items(i, 1))
        data = data.replace("[", "")
        data = data.replace("]", "")
        sampled_rels_set += data
        sampled_rels_set += "\n"

    return sampled_node_set, sampled_rels_set


def load_few_shots(schema_type: str) -> Dict:
    """Load few-shot examples based on schema type"""
    with open("few_shots.json", "r") as f:
        few_shots_data = json.load(f)
    
    if schema_type == "healthcare":
        return few_shots_data["healthcare"]
    elif schema_type == "wwc":
        return few_shots_data["wwc"]
    elif schema_type in ["twitter_trolls", "legis_graph", "osm", "er", "air_routes", "star_wars"]:
        return few_shots_data["twitter_trolls"]
    elif schema_type in ["covid", "pole", "bloom50", "gdsc"]:
        return few_shots_data["covid"]
    else:
        # Default fallback
        return few_shots_data.get(schema_type, {})


def Print(values):
    """Helper function to print values"""
    for value in values:
        print(value)


def main():
    """Main function to run the question generation process"""
    
    # Define query categories
    QUERY_CATEGORIES = {
        "Simple_Retrieval": 'simple retrieval questions focus on basic data extraction, retrieving nodes or relationships based on straightforward criteria such as labels, properties, or direct relationships. Examples include fetching all nodes labeled as "x" or retrieving relationships of a specific type like "y". Simple retrieval is essential for initial data inspections and basic reporting tasks.',

        "Complex_Retrieval": 'complex retrieval are advanced questions use the rich pattern-matching capabilities of Cypher to handle multiple node types and relationship patterns. They involve sophisticated filtering conditions and logical operations to extract nuanced insights from interconnected data points',

        "Simple_Aggregation": 'simple aggregation involves calculating basic statistical metrics over properties of nodes or relationships, such as counting the number of nodes, averaging property values, or determining maximum and minimum values. These questions summarize data characteristics and support quick analytical conclusions.',

        "Complex_Aggregation": 'complex aggregation, the most sophisticated category, these questions involve multiple aggregation functions and often group results over complex subgraphs. They calculate metrics like average number of reports per manager or total sales volume through a network, supporting strategic decision making and advanced reporting.',

        "Evaluation_query": "Evaluation query type focuses on retrieving specific pieces of data from complex databases with precision. Use clear and detailed instructions to extract relevant information, such as movie titles, product names, or employee IDs, depending on the context. Always ask for a single property or item, titled intuitively based on the data retrieved. ",
    }
    # Configuration for sampling
    # Get user configuration
    parser = argparse.ArgumentParser(description="Question Generation from Graph Schema")
    parser.add_argument("--m", type=int, default=80, help="Number of times to multiply the question generation")
    parser.add_argument("--n", type=int, default=5, help="Number of questions to generate for each category")
    parser.add_argument("--r", type=int, default=7, help="Number of relationship instances and nodes to sample")
    parser.add_argument("--kb", type=str, default=7, help="Enter the name of the knowledge base to use for question generation")
    args = parser.parse_args()
    
    # Get user input
    schema_type = args.kb
    
    # Load few-shot examples
    in_context_samples = load_few_shots(schema_type)
    
    # Initialize configuration
    
    config = Config()

    URI = Config.neo4j_uri
    PWD = Config.neo4j_password
    
    # Initialize the Neo4j connector module
    graph = Neo4jGraph(url=URI, username='neo4j', password=PWD, database='neo4j')
    
    # Module to extract data from the graph
    gutils = Neo4jSchema(url=URI, username='neo4j', password=PWD, database='neo4j')
    jschema = gutils.get_structured_schema
    schema = gutils.get_structured_schema
    
    # Extract schema components
    relationships = jschema['relationships']
    nodes = get_nodes_list(jschema)
    node_props_types = jschema['node_props']
    node_dtypes = retrieve_datatypes(jschema, "node")
    rel_dtypes = retrieve_datatypes(jschema, "rel")
    
    print("Nodes:", nodes)
    print("Relationships:", relationships)

    node_instances_size = args.r
    rels_instances_size = args.r
    
    multiplier = args.m
    num_question = args.n
    
    main_folder_name = input("Enter the main folder name:")
    data_folder_name = "data_" + input("Enter version of the data folder:")
    prompt_version = input("Enter the prompt version: ")
    temp = float(input("Enter the temperature: "))
    
    manager = DataManager(main_folder_name, data_folder_name, prompt_version, database=schema_type, temperature=temp)
    
    # Generate questions for each category
    for category, description in QUERY_CATEGORIES.items():
        category_questions = []
        
        for i in range(multiplier):
            print("=================================================================================")
            print(f"Generating questions for: {category}")
            print(f"============== Taking in-context samples for {category} ===============")
            
            sampled_node_set, sampled_rels_set = generate_nodes_rels_instances(
                nodes, relationships, gutils, node_instances_size, rels_instances_size
            )
            
            print("============== Generating questions ===============")
            generator = QuestionGenerator(config, api_key=config.gemini_api_key, temperature=temp)
            
            questions = generator.generate_questions(
                schema, sampled_node_set, sampled_rels_set, description, 
                category, in_context_samples, num_questions=num_question
            )
            print("============== Questions Generated ===============")
            print("========== cleaning questions ==========")
            print("===== cleaned data ======")
            print(questions)
            sample_data = manager.get_sampled_data(sampled_node_set, sampled_rels_set)
            if questions:
                if isinstance(questions, dict):
                    continue
                questions.append([sample_data])
                category_questions.extend(questions)
            
            python_time.sleep(5)

        print(" ============ Saving data ================")
        manager.save_data(category_questions, category)
        print(" ============ Data saved ================")


if __name__ == "__main__":
    main()