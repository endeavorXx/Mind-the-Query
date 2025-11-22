# NL-Cypher Dataset Generation and Validation Framework

A comprehensive framework for generating, validating, and consolidating natural language to Cypher query datasets using AI-powered question generation and multi-stage validation pipelines.

## 🚀 Overview

This project provides an end-to-end solution for creating high-quality datasets that map natural language questions to Cypher queries for Neo4j graph databases. The framework includes automated question generation using Google's Gemini AI, rigorous validation through schema/runtime/value checks, and sophisticated data consolidation with deduplication.

## 📁 Project Structure

```
Mind-the-Query/
├── nl_cypher_pair_generator.py    # AI-powered question generation
├── automated_validators.py        # Multi-stage validation pipeline
├── dataset_consolidator.py        # Data consolidation & deduplication
└── data_manager.py               # Data management utilities
├── .env                          # Environment variables (Neo4j, API keys)
├── db_schema.jsonl              # Database schema definitions
├── incontext_few_shots.json     # Few-shot examples for AI prompting
├── ReadMe.md                    # This documentation
│
├── 📂 Automated_Validated_Datasets/ # Results from validation pipeline
│   ├── bloom50/
│   │   ├── dataset_pipeline_stats.json
│   │   ├── overall_dataset_stats.txt
│   │   ├── *.json (5 files of NL‑Cypher pairs)
│   ├── covid/
│   ....
│
├── 📂 Manually_Validated_Datasets/  # Human-annotated quality datasets
│   ├── annotation_protocol.txt
│   ├── bloom/
│   │   ├── *.json (2 files of NL‑Cypher complex query pairs with reasoning)
│   ├── healthcare/
│   └── wwc/
│
├── 📂 Datasets/                     # Original Neo4j database dumps
│   ├── bloom-50.dump
│   ├── contact-tracing-50.dump
│   ├── entity-resolution-50.dump
│   ....
|
├── 📂 Prompt_Logs/              # AI prompt optimization logs
│   └── prompt_refinement_logs.txt
│
└── 📂 utils/                        # Utility modules and helpers
    ├── __init__.py
    ├── graph_utils.py
    ├── neo4j_conn.py
    ├── neo4j_schema.py
    |__ utilities.py
```

## 🛠️ Core Components

### 1. Question Generation (`nl_cypher_pair_generator.py`)
- **AI-Powered Generation**: Uses Google Gemini AI to generate natural language questions and corresponding Cypher queries
- **Schema-Aware**: Generates questions based on actual database schema and sample data
- **Category-Based**: Supports 5 query categories:
  - **Simple Retrieval**: Basic data extraction queries
  - **Complex Retrieval**: Advanced pattern-matching queries
  - **Simple Aggregation**: Basic statistical operations (COUNT, AVG, etc.)
  - **Complex Aggregation**: Multi-level aggregations with grouping
  - **Evaluation Query**: Precision data retrieval queries

### 2. Validation Pipeline (`automated_validators.py`)
A rigorous 3-stage automated sequential validation system:

#### Stage 1: Schema Validation
- Validates node labels and relationship types against database schema
- Checks property existence and data types
- Ensures syntactic correctness of Cypher queries

#### Stage 2: Runtime Validation
- Executes queries against actual Neo4j database
- Catches runtime errors and syntax issues
- Validates query executability and non-empty result generation

#### Stage 3: Value Validation
- Ensures consistency between natural language questions and Cypher values
- Validates that specific values mentioned in queries appear in the questions

### 3. Data Consolidation (`dataset_consolidator.py`)
- **Deduplication**: Removes duplicate questions within each dataset
- **Cross-chunk Integration**: Combines data from multiple generation runs
- **Statistics Generation**: Provides detailed consolidation metrics and generates comprehensive summary reports

### 4. Data Management (`data_manager.py`)
- Handles file organization and storage for generation pipeline

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Neo4j Database (local or cloud)
- Google Gemini API access

### Installation
1. Clone the repository:
```bash
git clone <repository_url>
cd Mind-the-Query
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
# Edit .env with your credentials:
NEO4J_URI=your_neo4j_uri
NEO4J_PASSWORD=your_password
GEMINI_API_KEY=your_gemini_api_key
```

### Usage

#### 1. Generate Questions
```bash
python nl_cypher_pair_generator.py --kb healthcare --m 10 --n 5 --r 7
```
Parameters:
- `--kb`: Knowledge base/dataset name
- `--m`: Number of generation iterations
- `--n`: Questions per category per iteration
- `--r`: Number of sample instances of nodes and relationships to use

#### 2. Consolidate and Deduplicate
```bash
python dataset_consolidator.py
```
Combines chunks and removes duplicates within each dataset.

#### 3. Validate Generated Datasets
```bash
python automated_validators.py
```
This runs the complete validation pipeline across all datasets.


## 📊 Query Categories

### Simple Retrieval
Basic data extraction focusing on straightforward criteria:
```cypher
MATCH (p:Person {name: "John"}) RETURN p
```

### Complex Retrieval
Advanced pattern-matching with multiple nodes and relationships:
```cypher
MATCH (p:Person)-[:WORKS_FOR]->(c:Company)-[:LOCATED_IN]->(city:City {name: "New York"}) 
WHERE p.age > 30 RETURN p.name, c.name
```

### Simple Aggregation
Basic statistical operations:
```cypher
MATCH (p:Person) RETURN COUNT(p) as total_people
```

### Complex Aggregation
Multi-level aggregations with grouping:
```cypher
MATCH (p:Person)-[:WORKS_FOR]->(c:Company) 
RETURN c.name, AVG(p.age) as avg_age, COUNT(p) as employee_count
ORDER BY avg_age DESC
```

### Evaluation Query
Precision data retrieval for specific information:
```cypher
MATCH (m:Movie {title: "The Matrix"}) RETURN m.releaseYear
```

## 📈 Datasets

Below are the synthetic datasets used in our experiments :-

- **[Healthcare Analytics (HCA)](https://github.com/neo4j-graph-examples/healthcare-analytics)**
- **[Contact Tracing (CT)](https://github.com/neo4j-graph-examples/contact-tracing)**
- **[Women World Cup 2019 (WWC)](https://github.com/neo4j-graph-examples/wwc2019)**
- **[U.S. Election Twitter Trolls (USTT)](https://github.com/neo4j-graph-examples/twitter-trolls)**
- **[Open Street Map (OSM)](https://github.com/neo4j-graph-examples/openstreetmap)**
- **[Legis Graph (LG)](https://github.com/neo4j-graph-examples/legis-graph)**
- **[Entity Resolution (ER)](https://github.com/neo4j-graph-examples/entity-resolution)**
- **[Graph Data Science (GDSC)](https://github.com/neo4j-graph-examples/graph-data-science)**
- **[Pole](https://github.com/neo4j-graph-examples/pole)**
- **[Bloom](https://github.com/neo4j-graph-examples/bloom)**
- **[Star Wars (SW)](https://github.com/neo4j-graph-examples/star-wars)**


## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

<!-- ## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. -->

## Acknowledgments

- Google Gemini AI for question generation
- Neo4j for graph database support
- [Text2Cypher repo for utilities](https://github.com/neo4j-labs/text2cypher)
- Contributors to the various dataset schemas

## Support

For questions or issues:
1. Check the documentation
2. Review existing issues
3. Create a new issue with detailed description

---

**Note**: This framework is designed for research and educational purposes. Ensure compliance with data usage policies and API terms of service.
