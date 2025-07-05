#!/usr/bin/env python3
"""
Database Query Validation Script
Converted from Jupyter notebook to standalone Python script
"""

# =============================================================================
# IMPORTS AND SETUP
# =============================================================================
import json
import requests
import csv
import os
import traceback
import io
import sys
import time as py_time
import re
from dataclasses import dataclass
from typing import Dict, List
from neo4j import GraphDatabase
from tqdm import tqdm
from utils.utilities import *
from utils.neo4j_conn import *
from utils.neo4j_schema import *
from utils.graph_utils import *
from dotenv import load_dotenv

print("Loading environment variables...")
load_dotenv()

# Configuration for Neo4j connection
URI = os.getenv("NEO4J_URI")
PWD = os.getenv("NEO4J_PASSWORD")


# Initialize the Neo4j connector module
graph = Neo4jGraph(url=URI, username='neo4j', password=PWD, database='neo4j')

# Module to extract data from the graph
gutils = Neo4jSchema(url=URI, username='neo4j', password=PWD, database='neo4j')
db_schema = gutils.get_structured_schema

# =============================================================================
# DATASETS AND CATEGORIES
# =============================================================================

datasets = ['wwc', 'bloom50', 'covid','er', 'osm', 'healthcare', 'pole', 'gdsc', 'twitter_trolls', 'legis_graph', 'star_wars']
categories = ['Simple_Retrieval', 'Simple_Aggregation', 'Complex_Retrieval', 'Complex_Aggregation', 'Evaluation_query']

# Base path for datasets
DATASETS_BASE_PATH = " < Enter the destination folder > "

# =============================================================================
# SCHEMA VALIDATOR
# =============================================================================


SPECIAL_DATATYPES = {
    "POINT": {"x", "y", "srid"},
    "DURATION": {"years", "months", "days", "hours", "minutes", "seconds", "milliseconds", "microseconds", "nanoseconds"}
}

def preprocess_schema(database_schema):
    schema = {"nodes": {}, "relationships": {}}

    # Process node properties
    for node, props in database_schema["node_props"].items():
        schema["nodes"][node] = {}
        for prop in props:
            datatype = prop["datatype"]
            property_name = prop["property"]
            if datatype in SPECIAL_DATATYPES:
                schema["nodes"][node][property_name] = SPECIAL_DATATYPES[datatype]
            else:
                schema["nodes"][node][property_name] = datatype

    # Process relationship properties
    rel_props = database_schema.get("rel_props", {})
    for rel in database_schema["relationships"]:
        rel_type = rel["type"]
        start = rel["start"]
        end = rel["end"]
        if rel_type not in schema["relationships"]:
            schema["relationships"][rel_type] = {"valid_pairs": [], "properties": {}}
        schema["relationships"][rel_type]["valid_pairs"].append((start, end))

    for rel_type, props in rel_props.items():
        if rel_type not in schema["relationships"]:
            schema["relationships"][rel_type] = {"valid_pairs": [], "properties": {}}
        if "properties" not in schema["relationships"][rel_type]:
            schema["relationships"][rel_type]["properties"] = {}

        for prop in props:
            datatype = prop["datatype"]
            property_name = prop["property"]

            if datatype in SPECIAL_DATATYPES:
                schema["relationships"][rel_type]["properties"][property_name] = SPECIAL_DATATYPES[datatype]
            else:
                schema["relationships"][rel_type]["properties"][property_name] = datatype

    return schema

def parse_inline_props(prop_string):
    """
    Parses inline Cypher properties safely, allowing:
    - strings (quoted),
    - numbers,
    - function calls,
    - variable names (e.g., region),
    - booleans (true/false).
    Returns:
        valid_props: list of (key, value) tuples
        invalid_segments: list of strings that failed to match
    """
    pattern = re.compile(
        r'''(\w+)\s*:\s*(
            "(?:\\.|[^"\\])*"              |   # double-quoted string with escapes
            '(?:\\.|[^'\\])*'              |   # single-quoted string with escapes
            \btrue\b|\bfalse\b             |   # boolean literals
            [-+]?\d*\.\d+|\d+              |   # float or int
            \w+\s*\([^()]*\)               |   # function calls like date('...')
            \w+                                # variable names like region, continentName
        )''',
        re.VERBOSE | re.IGNORECASE
    )

    matches = list(pattern.finditer(prop_string))

    valid_props = []
    valid_spans = []

    for m in matches:
        key = m.group(1)
        val = m.group(2).strip()
        valid_props.append((key, val))
        valid_spans.append(m.span())

    # Find any unmatched segments
    unmatched = []
    last_end = 0
    for start, end in valid_spans:
        if last_end < start:
            unmatched.append(prop_string[last_end:start].strip(", "))
        last_end = end
    if last_end < len(prop_string):
        unmatched.append(prop_string[last_end:].strip(", "))

    # Filter out empty strings
    invalid_segments = [seg for seg in unmatched if seg]

    return valid_props, invalid_segments


def validate_cypher(query, schema):
    errors = []

    # Find node patterns like (p:Person {healthstatus: "Sick"})
    node_matches = re.findall(r'\((\w+):(\w+)(?:\s*\{([^}]+)\})?\)', query)
    alias_to_label = {}
    for alias, label, inline_props in node_matches:
        alias_to_label[alias] = label
        if label not in schema["nodes"]:
            errors.append(f"❌ Node label `{label}` is not in the schema.")
        else:
            # Validate inline properties (robust parsing)
            if inline_props:
                prop_pairs, invalids = parse_inline_props(inline_props)
                
                # Report malformed property segments
                for invalid in invalids:
                    errors.append(f"❌ Malformed inline property: `{invalid}` in node `{label}`.")

                # Validate parsed key-value pairs
                for key, val in prop_pairs:
                    if key not in schema["nodes"][label]:
                        errors.append(f"❌ Inline property `{key}` not in node `{label}`.")


    # Find relationships like -[:VISITS]- or -[r:VISITS]-> 
    rel_matches = re.findall(r'-\[(\w*):(\w+)\]-[>]*', query)
    rel_alias_to_type = {}
    for alias, rel_type in rel_matches:
        rel_alias_to_type[alias] = rel_type
        if rel_type not in schema["relationships"]:
            errors.append(f"❌ Relationship type `{rel_type}` is not in the schema.")

    # Attribute access like p.name or v.duration.seconds
    attributes = re.findall(r'(\w+)\.((?:\w+\.)*\w+)', query)
    for alias, full_attr in attributes:
        attr_parts = full_attr.split(".")
        first_attr = attr_parts[0]

        # Check if alias is a node
        if alias in alias_to_label:
            node_label = alias_to_label[alias]
            node_props = schema["nodes"].get(node_label, {})
            if first_attr not in node_props:
                errors.append(f"❌ Attribute `{first_attr}` does not exist in node `{node_label}`.")
            elif isinstance(node_props[first_attr], set):  # Nested attribute
                expected_subfields = node_props[first_attr]
                for sub_attr in attr_parts[1:]:
                    if sub_attr not in expected_subfields:
                        errors.append(f"❌ `{first_attr}.{sub_attr}` is not valid in node `{node_label}`.")
        # Check if alias is a relationship
        elif alias in rel_alias_to_type:
            rel_type = rel_alias_to_type[alias]
            rel_props = schema["relationships"][rel_type]["properties"]
            if first_attr not in rel_props:
                errors.append(f"❌ Attribute `{first_attr}` not in relationship `{rel_type}`.")
            elif isinstance(rel_props[first_attr], set):  # Nested attribute
                expected_subfields = rel_props[first_attr]
                for sub_attr in attr_parts[1:]:
                    if sub_attr not in expected_subfields:
                        errors.append(f"❌ `{first_attr}.{sub_attr}` not valid in relationship `{rel_type}`.")
        else:
            # Could be RETURN alias or unknown
            continue

    return "✅ Query is valid according to schema!" if not errors else "\n".join(errors)

# =============================================================================
# VALUE VALIDATOR
# =============================================================================

def extract_entity_value_pairs_from_cypher(query):
    """Extracts (property, value) pairs from Cypher WHERE or inline filter blocks."""
    # Matches patterns like p.healthstatus = "Sick" or {name: "Timothy Norton"}
    prop_value_pairs = []

    # Extract inline filters: (p:Person {name:"Timothy Norton"})
    inline_matches = re.findall(r'\{([^\}]+)\}', query)
    for match in inline_matches:
        entries = match.split(',')
        for entry in entries:
            parts = entry.split(':', 1)
            if len(parts) == 2:
                prop = parts[0].strip()
                value = parts[1].strip().strip('"\'')
                prop_value_pairs.append((prop, value))

    # Extract WHERE filters: p.healthstatus = "Sick"
    where_matches = re.findall(r'(\w+)\.(\w+)\s*=\s*["\']([^"\']+)["\']', query)
    for _, prop, value in where_matches:
        prop_value_pairs.append((prop, value))

    return prop_value_pairs


def validate_values_against_question(query, question):
    """Validates if each value in Cypher query is reflected in the natural language question."""
    prop_value_pairs = extract_entity_value_pairs_from_cypher(query)

    # print(f"Extracted property-value pairs: {prop_value_pairs}")

    errors = []
    question_lower = question.lower()

    for prop, value in prop_value_pairs:
        # Normalize and try to find the value (lowercased) in question
        value_lower = value.lower()
        if value_lower in ["true", "false"]:
            continue
        if value_lower not in question_lower:
            errors.append(f"❌ Value '{value}' for property '{prop}' not found in the NL question.")

    if not errors:
        return "✅ All values in Cypher are consistent with the NL question."
    else:
        return "\n".join(errors)


# =============================================================================
# VALIDATION FOR ALL DATASETS
# =============================================================================

def validate_queries_for_dataset(dataset, database_schema):
    """Validates all queries for a dataset using sequential pipeline and saves the results."""
    schema = preprocess_schema(database_schema)
    print(schema)
    # Create pipeline folders for storing queries at each stage
    pipeline_dir = f"Automated_Validated_Datasets/{dataset}"
    os.makedirs(pipeline_dir, exist_ok=True)
    
    # Create folders for each pipeline stage - updated order: schema -> runtime -> value
    schema_passed_dir = f"{pipeline_dir}/01_Schema_Passed"
    runtime_passed_dir = f"{pipeline_dir}/02_Runtime_Passed"
    value_passed_dir = f"{pipeline_dir}/All_passed"
    failed_schema_dir = f"{pipeline_dir}/Failed_Schema"
    failed_runtime_dir = f"{pipeline_dir}/Failed_Runtime"
    failed_value_dir = f"{pipeline_dir}/Failed_Value"
    
    for dir_path in [schema_passed_dir, runtime_passed_dir, value_passed_dir, 
                     failed_schema_dir, failed_runtime_dir, failed_value_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Initialize dataset-level pipeline statistics - updated order
    dataset_pipeline_stats = {
        "dataset": dataset,
        "categories": {},
        "overall": {
            "total_queries": 0,
            "schema_stage": {"input": 0, "passed": 0, "failed": 0},
            "runtime_stage": {"input": 0, "passed": 0, "failed": 0},
            "value_stage": {"input": 0, "passed": 0, "failed": 0},
            "final_passed": 0
        }
    }
    
    # Process each category
    for category in categories:
        file_path = f"{DATASETS_BASE_PATH}/{dataset}/{category}.json"
        
        # Skip if file doesn't exist
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
            
        try:
            # Load the queries
            with open(file_path, 'r', encoding='utf-8') as f:
                queries = json.load(f)
                
            # Initialize category pipeline statistics - updated order
            category_stats = {
                "category": category,
                "total_queries": len(queries),
                "schema_stage": {"input": len(queries), "passed": 0, "failed": 0},
                "runtime_stage": {"input": 0, "passed": 0, "failed": 0},
                "value_stage": {"input": 0, "passed": 0, "failed": 0},
                "final_passed": 0
            }
            
            # Initialize pipeline stage collections
            pipeline_stages = {
                "schema_passed": [],
                "runtime_passed": [],
                "value_passed": [],
                "failed_schema": [],
                "failed_runtime": [],
                "failed_value": []
            }
            
            # STAGE 1: Schema Validation - Process ALL queries
            schema_passed_queries = []
            print(f"\n=== STAGE 1: Schema Validation for {dataset}/{category} ===")
            
            for query_obj in tqdm(queries, desc=f"Schema validation {category}", leave=True):
                if 'unique_id' not in query_obj or 'Cypher' not in query_obj:
                    continue
                
                cypher_query = query_obj['Cypher']
                query_copy = query_obj.copy()
                
                try:
                    # Schema validation
                    schema_validation = validate_cypher(cypher_query, schema)
                    schema_passed = 1 if schema_validation == "✅ Query is valid according to schema!" else 0
                    
                    if schema_passed:
                        # Passed schema validation
                        query_copy.update({
                            "pipeline_stage": "PASSED_SCHEMA",
                            "schema_validation": schema_validation
                        })
                        pipeline_stages["schema_passed"].append(query_copy)
                        schema_passed_queries.append(query_copy)
                        category_stats["schema_stage"]["passed"] += 1
                    else:
                        # Failed schema validation
                        query_copy.update({
                            "pipeline_stage": "FAILED_SCHEMA",
                            "failure_reason": schema_validation,
                            "schema_validation": schema_validation
                        })
                        pipeline_stages["failed_schema"].append(query_copy)
                        category_stats["schema_stage"]["failed"] += 1
                        
                except (KeyError, Exception) as e:
                    # Handle schema validation errors (like missing relationship properties)
                    error_message = f"Schema validation error: {str(e)}"
                    query_copy.update({
                        "pipeline_stage": "FAILED_SCHEMA",
                        "failure_reason": error_message,
                        "schema_validation": error_message
                    })
                    pipeline_stages["failed_schema"].append(query_copy)
                    category_stats["schema_stage"]["failed"] += 1
                    continue
            
            # STAGE 2: Runtime Validation - Only process schema-passed queries
            runtime_passed_queries = []
            category_stats["runtime_stage"]["input"] = len(schema_passed_queries)
            
            print(f"=== STAGE 2: Runtime Validation for {dataset}/{category} ===")
            print(f"Processing {len(schema_passed_queries)} queries that passed schema validation")
            
            for query_obj in tqdm(schema_passed_queries, desc=f"Runtime validation {category}", leave=True):
                cypher_query = query_obj['Cypher']
                query_copy = query_obj.copy()
                
                # Runtime validation
                output_capture = io.StringIO()
                original_stdout = sys.stdout
                sys.stdout = output_capture
                
                try:
                    result = gutils.run_cypher(cypher_query)
                    # Restore stdout and get captured output
                    sys.stdout = original_stdout
                    captured_output = output_capture.getvalue()
                    
                    # Check if there's a runtime error in the output
                    if captured_output.startswith("Runtime Error Caught\n"):
                        # Extract error message (remove the "Runtime Error Caught\n" prefix)
                        runtime_error = captured_output.replace("Runtime Error Caught\n", "", 1)
                        neo4j_runtime_validation = runtime_error
                        neo4j_runtime_passed = 0
                    else:
                        neo4j_runtime_validation = "No runtime errors."
                        neo4j_runtime_passed = 1
                                    
                    # Result Validation
                    if result == []:
                        neo4j_result_validation = "No results returned."
                        neo4j_result_passed = 0
                    else:
                        neo4j_result_validation = "Results returned successfully."
                        neo4j_result_passed = 1

                    # Content Validation
                    if result != []:
                        # Check if any value in any tuple is not null and not zero
                        has_valid_content = False
                        for row in result:
                            for value in row.values():
                                if value is not None and value != '' and value != []:
                                    has_valid_content = True
                                    break
                            if has_valid_content:
                                break
                        
                        if has_valid_content:
                            neo4j_content_validation = "Results contain non-null, non-zero values."
                            neo4j_content_passed = 1
                        else:
                            neo4j_content_validation = "Results only contain null or zero values."
                            neo4j_content_passed = 0
                    else:
                        neo4j_content_validation = "No results returned."
                        neo4j_content_passed = 1
                    
                    # Determine overall runtime stage result based on runtime and content validation
                    if neo4j_runtime_passed == 1 and neo4j_content_passed == 1 and neo4j_result_passed == 1:
                        # Passed both runtime and content validation
                        query_copy.update({
                            "pipeline_stage": "PASSED_RUNTIME",
                            "runtime_validation": "Passed",
                            "neo4j_runtime_validation": neo4j_runtime_validation,
                            "neo4j_runtime_passed": neo4j_runtime_passed,
                            "neo4j_result_validation": neo4j_result_validation,
                            "neo4j_result_passed": neo4j_result_passed,
                            "content_validation": neo4j_content_validation,
                            "content_passed": neo4j_content_passed
                        })
                        pipeline_stages["runtime_passed"].append(query_copy)
                        runtime_passed_queries.append(query_copy)
                        category_stats["runtime_stage"]["passed"] += 1
                    else:
                        # Failed either runtime or content validation
                        failure_reasons = []
                        if neo4j_runtime_passed == 0:
                            failure_reasons.append(f"Runtime: {neo4j_runtime_validation}")
                        if neo4j_content_passed == 0:
                            failure_reasons.append(f"Content: {neo4j_content_validation}")
                        
                        query_copy.update({
                            "pipeline_stage": "FAILED_RUNTIME",
                            "failure_reason": " | ".join(failure_reasons),
                            "runtime_validation": "Failed",
                            "neo4j_runtime_validation": neo4j_runtime_validation,
                            "neo4j_runtime_passed": neo4j_runtime_passed,
                            "neo4j_result_validation": neo4j_result_validation,
                            "neo4j_result_passed": neo4j_result_passed,
                            "content_validation": neo4j_content_validation,
                            "content_passed": neo4j_content_passed
                        })
                        pipeline_stages["failed_runtime"].append(query_copy)
                        category_stats["runtime_stage"]["failed"] += 1
                        
                    # sys.stdout = original_stdout  # Restore stdout in both success and error cases
                        
                except (ValueError, Exception) as e:
                    # Restore stdout in case of exception
                    sys.stdout = original_stdout
                    
                    # Handle Cypher syntax errors and other runtime errors
                    error_message = str(e)
                    if "Generated Cypher Statement is not valid" in error_message:
                        # Extract the actual Neo4j error message
                        runtime_error = f"Cypher Syntax Error: {error_message}"
                    else:
                        runtime_error = f"Runtime Error: {error_message}"
                    
                    query_copy.update({
                        "pipeline_stage": "FAILED_RUNTIME",
                        "failure_reason": runtime_error,
                        "runtime_validation": "Failed",
                        "content_validation": "Not applicable - runtime failed",
                        "content_passed": 0
                    })
                    pipeline_stages["failed_runtime"].append(query_copy)
                    category_stats["runtime_stage"]["failed"] += 1
                    
                    # Continue processing the next query instead of crashing
            
            # STAGE 3: Value Validation - Only process runtime-passed queries
            category_stats["value_stage"]["input"] = len(runtime_passed_queries)
            
            print(f"=== STAGE 3: Value Validation for {dataset}/{category} ===")
            print(f"Processing {len(runtime_passed_queries)} queries that passed runtime validation")
            
            for query_obj in tqdm(runtime_passed_queries, desc=f"Value validation {category}", leave=True):
                cypher_query = query_obj['Cypher']
                nl_question = query_obj.get('NL Question', '')
                query_copy = query_obj.copy()
                
                # Value validation
                value_validation = validate_values_against_question(cypher_query, nl_question)
                value_passed = 1 if value_validation.startswith("✅") else 0
                
                if value_passed:
                    # Passed all validations
                    query_copy.update({
                        "pipeline_stage": "PASSED_ALL",
                        "value_validation": value_validation
                    })
                    pipeline_stages["value_passed"].append(query_copy)
                    category_stats["value_stage"]["passed"] += 1
                    category_stats["final_passed"] += 1
                else:
                    # Failed value validation
                    query_copy.update({
                        "pipeline_stage": "FAILED_VALUE",
                        "failure_reason": value_validation,
                        "value_validation": value_validation
                    })
                    pipeline_stages["failed_value"].append(query_copy)
                    category_stats["value_stage"]["failed"] += 1
            
            # Save pipeline stage results to separate files
            for stage, stage_queries in pipeline_stages.items():
                if stage_queries:
                    stage_filename = f"{category}_{stage}.json"
                    
                    if stage == "schema_passed":
                        stage_file_path = f"{schema_passed_dir}/{stage_filename}"
                    elif stage == "runtime_passed":
                        stage_file_path = f"{runtime_passed_dir}/{stage_filename}"
                    elif stage == "value_passed":
                        stage_file_path = f"{value_passed_dir}/{stage_filename}"
                    elif stage == "failed_schema":
                        stage_file_path = f"{failed_schema_dir}/{stage_filename}"
                    elif stage == "failed_runtime":
                        stage_file_path = f"{failed_runtime_dir}/{stage_filename}"
                    elif stage == "failed_value":
                        stage_file_path = f"{failed_value_dir}/{stage_filename}"
                    
                    with open(stage_file_path, 'w', encoding='utf-8') as f:
                        json.dump(stage_queries, f, indent=2, ensure_ascii=False)
            
            # Print category pipeline statistics in the requested format - updated order
            total = category_stats["total_queries"]
            schema_input = category_stats["schema_stage"]["input"]
            schema_passed = category_stats["schema_stage"]["passed"]
            runtime_input = category_stats["runtime_stage"]["input"]
            runtime_passed = category_stats["runtime_stage"]["passed"]
            value_input = category_stats["value_stage"]["input"]
            value_passed = category_stats["value_stage"]["passed"]
            
            print(f"\n=== SEQUENTIAL PIPELINE STATS for {dataset}/{category} ===")
            print(f"Total Queries: {total}")
            print(f"Schema Validator: ({schema_passed}, {schema_input}) - {schema_passed/schema_input*100:.1f}% passed | Out-of-Total: {schema_passed}/{total} ({schema_passed/total*100:.1f}%)")
            print(f"Runtime Validator: ({runtime_passed}, {runtime_input}) - {runtime_passed/runtime_input*100:.1f}% passed | Out-of-Total: {runtime_passed}/{total} ({runtime_passed/total*100:.1f}%)" if runtime_input > 0 else f"Runtime Validator: (0, 0) - No queries to process | Out-of-Total: 0/{total} (0.0%)")
            print(f"Value Validator: ({value_passed}, {value_input}) - {value_passed/value_input*100:.1f}% passed | Out-of-Total: {value_passed}/{total} ({value_passed/total*100:.1f}%)" if value_input > 0 else f"Value Validator: (0, 0) - No queries to process | Out-of-Total: 0/{total} (0.0%)")
            print(f"All Passed Queries: {value_passed}/{total} ({value_passed/total*100:.1f}%)")
            
            # Also print the clean tuple format you requested - updated order
            print(f"\nCLEAN TUPLE FORMAT:")
            print(f"Schema: ({schema_passed}, {total})")
            print(f"Runtime: ({runtime_passed}, {schema_passed})")  
            print(f"Value: ({value_passed}, {runtime_passed})")
            print(f"Passed All: {value_passed}")
            
            # Store category stats
            dataset_pipeline_stats["categories"][category] = category_stats
            
            # Update overall dataset stats - updated order
            dataset_pipeline_stats["overall"]["total_queries"] += total
            dataset_pipeline_stats["overall"]["schema_stage"]["input"] += schema_input
            dataset_pipeline_stats["overall"]["schema_stage"]["passed"] += schema_passed
            dataset_pipeline_stats["overall"]["schema_stage"]["failed"] += category_stats["schema_stage"]["failed"]
            dataset_pipeline_stats["overall"]["runtime_stage"]["input"] += runtime_input
            dataset_pipeline_stats["overall"]["runtime_stage"]["passed"] += runtime_passed
            dataset_pipeline_stats["overall"]["runtime_stage"]["failed"] += category_stats["runtime_stage"]["failed"]
            dataset_pipeline_stats["overall"]["value_stage"]["input"] += value_input
            dataset_pipeline_stats["overall"]["value_stage"]["passed"] += value_passed
            dataset_pipeline_stats["overall"]["value_stage"]["failed"] += category_stats["value_stage"]["failed"]
            dataset_pipeline_stats["overall"]["final_passed"] += value_passed
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            traceback.print_exc()
    
    # Save dataset pipeline summary
    with open(f"{pipeline_dir}/dataset_pipeline_stats.json", 'w', encoding='utf-8') as f:
        json.dump(dataset_pipeline_stats, f, indent=2, ensure_ascii=False)
    
    # Print overall dataset statistics - updated order
    overall = dataset_pipeline_stats["overall"]
    print(f"\n=== OVERALL DATASET STATS for {dataset.upper()} ===")
    print(f"Total Queries: {overall['total_queries']}")
    print(f"Schema Validator: ({overall['schema_stage']['passed']}, {overall['schema_stage']['input']}) - {overall['schema_stage']['passed']/overall['schema_stage']['input']*100:.1f}% passed")
    print(f"Runtime Validator: ({overall['runtime_stage']['passed']}, {overall['runtime_stage']['input']}) - {overall['runtime_stage']['passed']/overall['runtime_stage']['input']*100:.1f}% passed" if overall['runtime_stage']['input'] > 0 else "Runtime Validator: (0, 0) - No queries to process")
    print(f"Value Validator: ({overall['value_stage']['passed']}, {overall['value_stage']['input']}) - {overall['value_stage']['passed']/overall['value_stage']['input']*100:.1f}% passed" if overall['value_stage']['input'] > 0 else "Value Validator: (0, 0) - No queries to process")
    print(f"All Passed Queries: {overall['final_passed']}")
    
    # Save overall dataset stats to text file - updated order
    with open(f"{pipeline_dir}/overall_dataset_stats.txt", 'w', encoding='utf-8') as f:
        f.write(f"=== OVERALL DATASET STATS for {dataset.upper()} ===\n")
        f.write(f"Total Queries: {overall['total_queries']}\n")
        f.write(f"Schema Validator: ({overall['schema_stage']['passed']}, {overall['schema_stage']['input']}) - {overall['schema_stage']['passed']/overall['schema_stage']['input']*100:.1f}% passed\n")
        if overall['runtime_stage']['input'] > 0:
            f.write(f"Runtime Validator: ({overall['runtime_stage']['passed']}, {overall['runtime_stage']['input']}) - {overall['runtime_stage']['passed']/overall['runtime_stage']['input']*100:.1f}% passed\n")
        else:
            f.write("Runtime Validator: (0, 0) - No queries to process\n")
        if overall['value_stage']['input'] > 0:
            f.write(f"Value Validator: ({overall['value_stage']['passed']}, {overall['value_stage']['input']}) - {overall['value_stage']['passed']/overall['value_stage']['input']*100:.1f}% passed\n")
        else:
            f.write("Value Validator: (0, 0) - No queries to process\n")
        f.write(f"All Passed Queries: {overall['final_passed']}\n")
    
    print(f"\nOverall dataset stats saved to: {pipeline_dir}/overall_dataset_stats.txt")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Main function to execute the validation process."""    
    # Process all datasets
    for dataset in datasets:
        # wait for 2 minutes before processing each dataset
        print(f"\nWaiting for 2 minutes before processing dataset: {dataset}")
        py_time.sleep(60)

        # Configuration for Neo4j connection
        URI = os.getenv("NEO4J_URI")
        PWD = os.getenv("NEO4J_PASSWORD")

        # Initialize the Neo4j connector module
        graph = Neo4jGraph(url=URI, username='neo4j', password=PWD, database='neo4j')
        
        # Module to extract data from the graph
        gutils = Neo4jSchema(url=URI, username='neo4j', password=PWD, database='neo4j')
        db_schema = gutils.get_structured_schema
        
        print(f"\nProcessing dataset: {dataset}")
        validate_queries_for_dataset(dataset, db_schema)

    # Collect overall statistics for sequential pipeline
    total_queries = 0
    runtime_valid_queries = 0
    schema_valid_queries = 0
    value_valid_queries = 0
    neo4j_result_valid_queries = 0

    # Initialize dataset-wise statistics for sequential pipeline
    dataset_stats = {}

    for dataset in datasets:
        dataset_stats[dataset] = {
            "total_queries": 0,
            "runtime_valid_queries": 0,
            "schema_valid_queries": 0,
            "value_valid_queries": 0,
            "neo4j_result_passed": 0,
            "pipeline_stages": {
                "FAILED_RUNTIME": 0,
                "PASSED_RUNTIME": 0,
                "FAILED_SCHEMA": 0,
                "PASSED_SCHEMA": 0,
                "PASSED_ALL": 0,
                "FAILED_VALUE": 0
            },
            "categories": {}
        }
        
        for category in categories:
            dataset_stats[dataset]["categories"][category] = {
                "total_queries": 0,
                "runtime_valid_queries": 0,
                "schema_valid_queries": 0,
                "value_valid_queries": 0,
                "neo4j_result_passed": 0,
                "pipeline_stages": {
                    "FAILED_RUNTIME": 0,
                    "PASSED_RUNTIME": 0,
                    "FAILED_SCHEMA": 0,
                    "PASSED_SCHEMA": 0,
                    "PASSED_ALL": 0,
                    "FAILED_VALUE": 0
                }
            }
            
            result_path = f"{DATASETS_BASE_PATH}/{dataset}/Validation_Results/{category}_validation.json"
            if os.path.exists(result_path):
                with open(result_path, 'r', encoding='utf-8') as f:
                    queries = json.load(f)
                    total_queries += len(queries)
                    
                    # Update dataset category stats
                    dataset_stats[dataset]["categories"][category]["total_queries"] = len(queries)
                    
                    for q in queries:
                        # Count pipeline stages
                        pipeline_status = q.get('pipeline_status', 'FAILED_RUNTIME')
                        dataset_stats[dataset]["categories"][category]["pipeline_stages"][pipeline_status] += 1
                        dataset_stats[dataset]["pipeline_stages"][pipeline_status] += 1
                        
                        # Sequential pipeline counts
                        if q.get('neo4j_runtime_passed', 0) == 1:
                            runtime_valid_queries += 1
                            dataset_stats[dataset]["categories"][category]["runtime_valid_queries"] += 1
                            dataset_stats[dataset]["runtime_valid_queries"] += 1
                        
                        if q.get('schema_validation_passed', 0) == 1:
                            schema_valid_queries += 1
                            dataset_stats[dataset]["categories"][category]["schema_valid_queries"] += 1
                            dataset_stats[dataset]["schema_valid_queries"] += 1
                            
                        if q.get('value_validation_passed', 0) == 1:
                            value_valid_queries += 1
                            dataset_stats[dataset]["categories"][category]["value_valid_queries"] += 1
                            dataset_stats[dataset]["value_valid_queries"] += 1
                            
                        if q.get('neo4j_result_passed', 0) == 1:
                            neo4j_result_valid_queries += 1
                            dataset_stats[dataset]["categories"][category]["neo4j_result_passed"] += 1
                            dataset_stats[dataset]["neo4j_result_passed"] += 1

                    # Update dataset total stats
                    dataset_stats[dataset]["total_queries"] += dataset_stats[dataset]["categories"][category]["total_queries"]

    # Print Sequential Pipeline Statistics
    print(f"\n" + "="*60)
    print("SEQUENTIAL PIPELINE VALIDATION RESULTS")
    print("="*60)
    print(f"Total queries processed: {total_queries}")
    print(f"\nPIPELINE FLOW STATISTICS:")
    print(f"1. Runtime Validation Passed: {runtime_valid_queries} ({runtime_valid_queries/total_queries*100:.2f}%)")
    print(f"2. Schema Validation Passed: {schema_valid_queries} ({schema_valid_queries/total_queries*100:.2f}%)")
    print(f"3. Value Validation Passed: {value_valid_queries} ({value_valid_queries/total_queries*100:.2f}%)")
    print(f"4. Neo4j Result Valid: {neo4j_result_valid_queries} ({neo4j_result_valid_queries/total_queries*100:.2f}%)")
    
    # Calculate drop-off rates
    runtime_to_schema_dropoff = runtime_valid_queries - schema_valid_queries
    schema_to_value_dropoff = schema_valid_queries - value_valid_queries
    
    print(f"\nPIPELINE DROP-OFF ANALYSIS:")
    print(f"Queries lost from Runtime to Schema: {runtime_to_schema_dropoff} ({runtime_to_schema_dropoff/total_queries*100:.2f}%)")
    print(f"Queries lost from Schema to Value: {schema_to_value_dropoff} ({schema_to_value_dropoff/total_queries*100:.2f}%)")
    print(f"Total queries completing full pipeline: {value_valid_queries} ({value_valid_queries/total_queries*100:.2f}%)")

    # Print dataset-wise sequential pipeline statistics
    print("\n" + "="*60)
    print("DATASET-WISE SEQUENTIAL PIPELINE STATISTICS:")
    print("="*60)
    for dataset in datasets:
        ds_stats = dataset_stats[dataset]
        if ds_stats["total_queries"] > 0:
            print(f"\n{dataset.upper()}:")
            print(f"  Total queries: {ds_stats['total_queries']}")
            print(f"  Runtime Validation: {ds_stats['runtime_valid_queries']} ({ds_stats['runtime_valid_queries']/ds_stats['total_queries']*100:.2f}%)")
            print(f"  Schema Validation: {ds_stats['schema_valid_queries']} ({ds_stats['schema_valid_queries']/ds_stats['total_queries']*100:.2f}%)")
            print(f"  Value Validation: {ds_stats['value_valid_queries']} ({ds_stats['value_valid_queries']/ds_stats['total_queries']*100:.2f}%)")
            print(f"  Neo4j Result Valid: {ds_stats['neo4j_result_passed']} ({ds_stats['neo4j_result_passed']/ds_stats['total_queries']*100:.2f}%)")
            
            # Pipeline stage breakdown
            print(f"  Pipeline Stage Breakdown:")
            for stage, count in ds_stats["pipeline_stages"].items():
                if count > 0:
                    print(f"    {stage}: {count} ({count/ds_stats['total_queries']*100:.2f}%)")

            print(f"\n  Category breakdown:")
            for category in categories:
                cat_stats = ds_stats["categories"].get(category, {"total_queries": 0})
                if cat_stats["total_queries"] > 0:
                    print(f"    {category}:")
                    print(f"      Total queries: {cat_stats['total_queries']}")
                    print(f"      Runtime Validation: {cat_stats['runtime_valid_queries']} ({cat_stats['runtime_valid_queries']/cat_stats['total_queries']*100:.2f}%)")
                    print(f"      Schema Validation: {cat_stats['schema_valid_queries']} ({cat_stats['schema_valid_queries']/cat_stats['total_queries']*100:.2f}%)")
                    print(f"      Value Validation: {cat_stats['value_valid_queries']} ({cat_stats['value_valid_queries']/cat_stats['total_queries']*100:.2f})")
                    print(f"      Neo4j Result Valid: {cat_stats['neo4j_result_passed']} ({cat_stats['neo4j_result_passed']/cat_stats['total_queries']*100:.2f}%)")

    # Write the sequential pipeline validation summary to a file for each dataset
    for dataset in datasets:
        ds_stats = dataset_stats[dataset]
        if ds_stats["total_queries"] > 0:
            # Create the validation directory if it doesn't exist
            validation_dir = f"{DATASETS_BASE_PATH}/{dataset}/Validation_Results"
            os.makedirs(validation_dir, exist_ok=True)
            
            # Write the sequential pipeline summary to a file
            with open(f"{validation_dir}/sequential_pipeline_summary_report.txt", 'w', encoding='utf-8') as f:
                f.write(f"SEQUENTIAL PIPELINE VALIDATION SUMMARY FOR {dataset.upper()}\n")
                f.write(f"=====================================================\n\n")
                f.write(f"Total queries: {ds_stats['total_queries']}\n\n")
                
                f.write(f"SEQUENTIAL PIPELINE FLOW:\n")
                f.write(f"1. Runtime Validation: {ds_stats['runtime_valid_queries']} ({ds_stats['runtime_valid_queries']/ds_stats['total_queries']*100:.2f}%)\n")
                f.write(f"2. Schema Validation: {ds_stats['schema_valid_queries']} ({ds_stats['schema_valid_queries']/ds_stats['total_queries']*100:.2f}%)\n")
                f.write(f"3. Value Validation: {ds_stats['value_valid_queries']} ({ds_stats['value_valid_queries']/ds_stats['total_queries']*100:.2f})\n")
                f.write(f"4. Neo4j Result Valid: {ds_stats['neo4j_result_passed']} ({ds_stats['neo4j_result_passed']/ds_stats['total_queries']*100:.2f}%)\n\n")
                
                # Drop-off analysis
                ds_runtime_to_schema_drop = ds_stats['runtime_valid_queries'] - ds_stats['schema_valid_queries']
                ds_schema_to_value_drop = ds_stats['schema_valid_queries'] - ds_stats['value_valid_queries']
                
                f.write(f"PIPELINE DROP-OFF ANALYSIS:\n")
                f.write(f"Queries lost from Runtime to Schema: {ds_runtime_to_schema_drop} ({ds_runtime_to_schema_drop/ds_stats['total_queries']*100:.2f}%)\n")
                f.write(f"Queries lost from Schema to Value: {ds_schema_to_value_drop} ({ds_schema_to_value_drop/ds_stats['total_queries']*100:.2f}%)\n")
                f.write(f"Total queries completing full pipeline: {ds_stats['value_valid_queries']} ({ds_stats['value_valid_queries']/ds_stats['total_queries']*100:.2f}%)\n\n")
                
                f.write(f"PIPELINE STAGE BREAKDOWN:\n")
                for stage, count in ds_stats["pipeline_stages"].items():
                    if count > 0:
                        f.write(f"{stage}: {count} ({count/ds_stats['total_queries']*100:.2f}%)\n")
                f.write(f"\n")

                f.write("CATEGORY BREAKDOWN:\n")
                f.write("------------------\n")
                for category in categories:
                    cat_stats = ds_stats["categories"].get(category, {"total_queries": 0})
                    if cat_stats["total_queries"] > 0:
                        f.write(f"\n{category}:\n")
                        f.write(f"  Total queries: {cat_stats['total_queries']}\n")
                        f.write(f"  Runtime Validation: {cat_stats['runtime_valid_queries']} ({cat_stats['runtime_valid_queries']/cat_stats['total_queries']*100:.2f}%)\n")
                        f.write(f"  Schema Validation: {cat_stats['schema_valid_queries']} ({cat_stats['schema_valid_queries']/cat_stats['total_queries']*100:.2f}%)\n")
                        f.write(f"  Value Validation: {cat_stats['value_valid_queries']} ({cat_stats['value_valid_queries']/cat_stats['total_queries']*100:.2f})\n")
                        f.write(f"  Neo4j Result Valid: {cat_stats['neo4j_result_passed']} ({cat_stats['neo4j_result_passed']/cat_stats['total_queries']*100:.2f}%)\n")
                        
                        f.write(f"  Pipeline Stages:\n")
                        for stage, count in cat_stats["pipeline_stages"].items():
                            if count > 0:
                                f.write(f"    {stage}: {count} ({count/cat_stats['total_queries']*100:.2f}%)\n")

    # Create an overall sequential pipeline summary report
    with open("sequential_pipeline_overall_summary.txt", 'w', encoding='utf-8') as f:
        f.write("OVERALL SEQUENTIAL PIPELINE VALIDATION SUMMARY\n")
        f.write("============================================\n\n")
        f.write(f"Total Queries Processed: {total_queries}\n\n")
        
        f.write(f"SEQUENTIAL PIPELINE FLOW:\n")
        f.write(f"1. Runtime Validation Passed: {runtime_valid_queries} ({runtime_valid_queries/total_queries*100:.2f}%)\n")
        f.write(f"2. Schema Validation Passed: {schema_valid_queries} ({schema_valid_queries/total_queries*100:.2f}%)\n")
        f.write(f"3. Value Validation Passed: {value_valid_queries} ({value_valid_queries/total_queries*100:.2f}%)\n")
        f.write(f"4. Neo4j Result Valid: {neo4j_result_valid_queries} ({neo4j_result_valid_queries/total_queries*100:.2f}%)\n\n")
        
        f.write(f"PIPELINE DROP-OFF ANALYSIS:\n")
        f.write(f"Queries lost from Runtime to Schema: {runtime_to_schema_dropoff} ({runtime_to_schema_dropoff/total_queries*100:.2f}%)\n")
        f.write(f"Queries lost from Schema to Value: {schema_to_value_dropoff} ({schema_to_value_dropoff/total_queries*100:.2f}%)\n")
        f.write(f"Total queries completing full pipeline: {value_valid_queries} ({value_valid_queries/total_queries*100:.2f}%)\n\n")
        
        # Dataset summaries
        f.write("DATASET SUMMARIES\n")
        f.write("================\n\n")
        for dataset in datasets:
            ds_stats = dataset_stats[dataset]
            if ds_stats["total_queries"] > 0:
                f.write(f"{dataset.upper()}:\n")
                f.write(f"Total Queries: {ds_stats['total_queries']}\n")
                f.write(f"Runtime Valid: {ds_stats['runtime_valid_queries']} ({ds_stats['runtime_valid_queries']/ds_stats['total_queries']*100:.2f}%)\n")
                f.write(f"Schema Valid: {ds_stats['schema_valid_queries']} ({ds_stats['schema_valid_queries']/ds_stats['total_queries']*100:.2f}%)\n")
                f.write(f"Value Valid: {ds_stats['value_valid_queries']} ({ds_stats['value_valid_queries']/ds_stats['total_queries']*100:.2f}%)\n")
                f.write(f"Neo4j Result Valid: {ds_stats['neo4j_result_passed']} ({ds_stats['neo4j_result_passed']/ds_stats['total_queries']*100:.2f}%)\n")
                f.write(f"Pipeline Completion Rate: {ds_stats['value_valid_queries']/ds_stats['total_queries']*100:.2f}%\n")
                f.write("\n")
        
        # Category summaries across all datasets
        f.write("CATEGORY SUMMARIES\n")
        f.write("=================\n\n")
        
        category_stats = {}
        for category in categories:
            category_stats[category] = {
                "total_queries": 0,
                "runtime_valid_queries": 0,
                "schema_valid_queries": 0,
                "value_valid_queries": 0,
                "neo4j_result_passed": 0
            }
        
        # Aggregate category stats across all datasets
        for dataset in datasets:
            for category in categories:
                if category in dataset_stats[dataset]["categories"]:
                    cat_stats = dataset_stats[dataset]["categories"][category]
                    category_stats[category]["total_queries"] += cat_stats["total_queries"]
                    category_stats[category]["runtime_valid_queries"] += cat_stats["runtime_valid_queries"]
                    category_stats[category]["schema_valid_queries"] += cat_stats["schema_valid_queries"]
                    category_stats[category]["value_valid_queries"] += cat_stats["value_valid_queries"]
                    category_stats[category]["neo4j_result_passed"] += cat_stats["neo4j_result_passed"]
        
        # Write category summaries
        for category in categories:
            cat_stats = category_stats[category]
            if cat_stats["total_queries"] > 0:
                f.write(f"{category}:\n")
                f.write(f"Total Queries: {cat_stats['total_queries']}\n")
                f.write(f"Runtime Valid: {cat_stats['runtime_valid_queries']} ({cat_stats['runtime_valid_queries']/cat_stats['total_queries']*100:.2f}%)\n")
                f.write(f"Schema Valid: {cat_stats['schema_valid_queries']} ({cat_stats['schema_valid_queries']/cat_stats['total_queries']*100:.2f}%)\n")
                f.write(f"Value Valid: {cat_stats['value_valid_queries']} ({cat_stats['value_valid_queries']/cat_stats['total_queries']*100:.2f}%)\n")
                f.write(f"Neo4j Result Valid: {cat_stats['neo4j_result_passed']} ({cat_stats['neo4j_result_passed']/cat_stats['total_queries']*100:.2f}%)\n")
                f.write(f"Pipeline Completion Rate: {cat_stats['value_valid_queries']/cat_stats['total_queries']*100:.2f}%\n")
                f.write("\n")

    # Create a detailed CSV file for sequential pipeline analysis
    with open("sequential_pipeline_stats.csv", 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        # Write headers
        csvwriter.writerow(['Dataset', 'Category', 'Total Queries', 
                          'Runtime Valid', 'Runtime Valid %', 
                          'Schema Valid', 'Schema Valid %', 
                          'Value Valid', 'Value Valid %',
                          'Neo4j Result Valid', 'Neo4j Result Valid %',
                          'Pipeline Completion Rate %',
                          'Runtime to Schema Drop %',
                          'Schema to Value Drop %'])
        
        # Write overall stats as first row
        overall_completion_rate = value_valid_queries/total_queries*100 if total_queries > 0 else 0
        overall_runtime_to_schema_drop = runtime_to_schema_dropoff/total_queries*100 if total_queries > 0 else 0
        overall_schema_to_value_drop = schema_to_value_dropoff/total_queries*100 if total_queries > 0 else 0
        
        csvwriter.writerow(['OVERALL', 'ALL', total_queries, 
                          runtime_valid_queries, f"{runtime_valid_queries/total_queries*100:.2f}" if total_queries > 0 else "0.00",
                          schema_valid_queries, f"{schema_valid_queries/total_queries*100:.2f}" if total_queries > 0 else "0.00",
                          value_valid_queries, f"{value_valid_queries/total_queries*100:.2f}" if total_queries > 0 else "0.00",
                          neo4j_result_valid_queries, f"{neo4j_result_valid_queries/total_queries*100:.2f}" if total_queries > 0 else "0.00",
                          f"{overall_completion_rate:.2f}",
                          f"{overall_runtime_to_schema_drop:.2f}",
                          f"{overall_schema_to_value_drop:.2f}"])
        
        # Write dataset stats
        for dataset in datasets:
            ds_stats = dataset_stats[dataset]
            if ds_stats["total_queries"] > 0:
                ds_completion_rate = ds_stats['value_valid_queries']/ds_stats['total_queries']*100
                ds_runtime_to_schema_drop = (ds_stats['runtime_valid_queries'] - ds_stats['schema_valid_queries'])/ds_stats['total_queries']*100
                ds_schema_to_value_drop = (ds_stats['schema_valid_queries'] - ds_stats['value_valid_queries'])/ds_stats['total_queries']*100
                
                csvwriter.writerow([dataset.upper(), 'ALL', ds_stats['total_queries'], 
                                  ds_stats['runtime_valid_queries'], f"{ds_stats['runtime_valid_queries']/ds_stats['total_queries']*100:.2f}",
                                  ds_stats['schema_valid_queries'], f"{ds_stats['schema_valid_queries']/ds_stats['total_queries']*100:.2f}",
                                  ds_stats['value_valid_queries'], f"{ds_stats['value_valid_queries']/ds_stats['total_queries']*100:.2f}",
                                  ds_stats['neo4j_result_passed'], f"{ds_stats['neo4j_result_passed']/ds_stats['total_queries']*100:.2f}",
                                  f"{ds_completion_rate:.2f}",
                                  f"{ds_runtime_to_schema_drop:.2f}",
                                  f"{ds_schema_to_value_drop:.2f}"])
                
                # Write dataset's category stats
                for category in categories:
                    if category in ds_stats["categories"]:
                        cat_stats = ds_stats["categories"][category]
                        if cat_stats["total_queries"] > 0:
                            cat_completion_rate = cat_stats['value_valid_queries']/cat_stats['total_queries']*100
                            cat_runtime_to_schema_drop = (cat_stats['runtime_valid_queries'] - cat_stats['schema_valid_queries'])/cat_stats['total_queries']*100
                            cat_schema_to_value_drop = (cat_stats['schema_valid_queries'] - cat_stats['value_valid_queries'])/cat_stats['total_queries']*100
                            
                            csvwriter.writerow([dataset.upper(), category, cat_stats['total_queries'],
                                              cat_stats['runtime_valid_queries'], f"{cat_stats['runtime_valid_queries']/cat_stats['total_queries']*100:.2f}",
                                              cat_stats['schema_valid_queries'], f"{cat_stats['schema_valid_queries']/cat_stats['total_queries']*100:.2f}",
                                              cat_stats['value_valid_queries'], f"{cat_stats['value_valid_queries']/cat_stats['total_queries']*100:.2f}",
                                              cat_stats['neo4j_result_passed'], f"{cat_stats['neo4j_result_passed']/cat_stats['total_queries']*100:.2f}",
                                              f"{cat_completion_rate:.2f}",
                                              f"{cat_runtime_to_schema_drop:.2f}",
                                              f"{cat_schema_to_value_drop:.2f}"])

    print("\n" + "="*60)
    print("SEQUENTIAL PIPELINE VALIDATION REPORTS GENERATED:")
    print("="*60)
    print("1. sequential_pipeline_overall_summary.txt - Overall pipeline statistics")
    print("2. sequential_pipeline_stats.csv - Detailed CSV with all metrics")
    print("3. Individual dataset reports in each dataset's Validation_Results folder:")
    print("   - sequential_pipeline_summary_report.txt")
    print("\nKey Insights:")
    print(f"• Pipeline Completion Rate: {value_valid_queries/total_queries*100:.1f}% of queries pass all validation steps")
    print(f"• Runtime Validation: {runtime_valid_queries/total_queries*100:.1f}% pass (entry point)")
    print(f"• Schema Validation: {schema_valid_queries/total_queries*100:.1f}% pass (conditional on runtime)")
    print(f"• Value Validation: {value_valid_queries/total_queries*100:.1f}% pass (conditional on schema)")


if __name__ == "__main__":
    main()

