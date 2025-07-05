import json
import os
from pathlib import Path
from collections import defaultdict
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_consolidation.log'),
        logging.StreamHandler()
    ]
)

class DatasetConsolidator:
    """
    Consolidates datasets from multiple chunks, removing duplicates based on NL questions using direct string comparison.
    Uniqueness is determined independently for each dataset.
    """
    
    def __init__(self, base_path, output_path, temp):
        self.base_path = Path(base_path)
        self.output_path = Path(output_path)
        self.temp = temp
        self.query_categories = [
            "Simple_Retrieval",
            "Complex_Retrieval", 
            "Simple_Aggregation",
            "Complex_Aggregation",
            "Evaluation_query"
        ]
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Initialize stats
        self.overall_stats = {
            "consolidation_timestamp": datetime.now().isoformat(),
            "total_datasets_processed": 0,
            "total_unique_questions": 0,
            "total_duplicates_removed": 0,
            "datasets": {}
        }
    
    def scan_all_datasets(self):
        """
        Scan directory structure to identify all datasets and their chunks.
        """
        datasets_info = {}
        
        logging.info(f"Scanning base path: {self.base_path}")
        
        if not self.base_path.exists():
            logging.error(f"Base path {self.base_path} does not exist")
            return datasets_info
        
        for item in self.base_path.iterdir():
            if item.is_dir():
                t_path = item / f"t_{self.temp}"

                if t_path.exists():
                    chunks = [
                        chunk.name for chunk in t_path.iterdir() 
                        if chunk.is_dir() and chunk.name.startswith("data_")
                    ]
                    if chunks:
                        datasets_info[item.name] = sorted(chunks)
                        logging.info(f"Found dataset '{item.name}' with chunks: {chunks}")
        
        logging.info(f"Total datasets found: {len(datasets_info)}")
        return datasets_info
    
    def normalize_question(self, question):
        """
        Normalize NL question for comparison (remove extra spaces, convert to lowercase).
        """
        return ' '.join(question.lower().strip().split())
    
    def process_chunk_files(self, chunk_path, chunk_name, combined_data, seen_questions, duplicate_stats, dataset_name):
        """
        Process all query category files in a chunk using direct string comparison.
        Uniqueness is determined within the specific dataset only.
        """
        chunk_path = Path(chunk_path)
        cypher_pairs_path = chunk_path / "cypher_pairs_only"
        
        if not cypher_pairs_path.exists():
            logging.warning(f"cypher_pairs_only directory not found in {chunk_path}")
            return
        
        logging.info(f"Processing chunk: {chunk_name} for dataset: {dataset_name}")
        
        for category in self.query_categories:
            file_path = cypher_pairs_path / f"{category}.json"
            
            if not file_path.exists():
                logging.warning(f"File not found: {file_path}")
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                logging.info(f"Processing {len(data)} items from {category} in {dataset_name}")
                
                for idx, item in enumerate(data):
                    if isinstance(item, dict) and 'NL Question' in item and 'Cypher' in item:
                        nl_question = item['NL Question']
                        
                        # Use direct string comparison for uniqueness check within this dataset
                        if nl_question not in seen_questions[category]:
                            # Add metadata for tracking
                            enhanced_item = {
                                **item,
                                'source_dataset': dataset_name,
                                'source_chunk': chunk_name,
                                'original_file': f"{category}.json",
                                'original_index': idx,
                                'unique_id': f"{dataset_name}_{chunk_name}_{category}_{idx}"
                            }
                            
                            combined_data[category].append(enhanced_item)
                            seen_questions[category].add(nl_question)
                        else:
                            duplicate_stats[category] += 1
                            logging.debug(f"Duplicate found in {dataset_name}/{category}: {nl_question[:50]}...")
                    
                    elif isinstance(item, dict) and len(item) == 0:
                        # Skip empty dictionaries
                        continue
                    else:
                        logging.warning(f"Invalid item structure in {file_path} at index {idx}")
                        
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error in {file_path}: {e}")
            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")
    
    def combine_dataset_chunks(self, dataset_name, chunks):
        """
        Combine all chunks for a specific dataset using direct string comparison.
        Uniqueness is determined independently within this dataset only.
        """
        logging.info(f"Combining chunks for dataset: {dataset_name}")
        logging.info(f"Uniqueness will be determined independently within {dataset_name}")
        
        # Initialize data structures for THIS DATASET ONLY
        combined_data = {category: [] for category in self.query_categories}
        seen_questions = {category: set() for category in self.query_categories}
        duplicate_stats = {category: 0 for category in self.query_categories}
        
        # Create dataset output directory
        dataset_output_dir = self.output_path / dataset_name
        dataset_output_dir.mkdir(exist_ok=True)
        
        # Process each chunk for this dataset
        for chunk in chunks:
            chunk_path = self.base_path / dataset_name / f"t_{self.temp}" / chunk
            self.process_chunk_files(chunk_path, chunk, combined_data, seen_questions, duplicate_stats, dataset_name)
        
        # Save combined datasets and generate statistics for this dataset
        dataset_stats = self.save_combined_data(dataset_output_dir, combined_data, duplicate_stats, dataset_name, chunks)
        
        return dataset_stats
    
    def save_combined_data(self, output_dir, combined_data, duplicate_stats, dataset_name, chunks):
        """
        Save combined data and generate statistics for a specific dataset.
        """
        logging.info(f"Saving combined data for {dataset_name}")
        
        stats = {
            "dataset": dataset_name,
            "chunks_processed": chunks,
            "consolidation_timestamp": datetime.now().isoformat(),
            "uniqueness_scope": "within_dataset_only",
            "total_unique_questions": 0,
            "total_duplicates_removed": 0,
            "categories": {}
        }
        
        # Save individual category files and collect statistics
        for category, items in combined_data.items():
            if items:  # Only save if there are items
                # Save category file
                category_file = output_dir / f"{category}.json"
                with open(category_file, 'w', encoding='utf-8') as f:
                    json.dump(items, f, indent=2, ensure_ascii=False)
                
                logging.info(f"Saved {len(items)} unique items to {category_file}")
            
            # Update statistics
            unique_count = len(items)
            duplicate_count = duplicate_stats[category]
            
            stats["categories"][category] = {
                "unique_questions": unique_count,
                "duplicates_removed": duplicate_count,
                "total_processed": unique_count + duplicate_count,
                "deduplication_rate": f"{(duplicate_count / (unique_count + duplicate_count) * 100):.2f}%" if (unique_count + duplicate_count) > 0 else "0.00%"
            }
            
            stats["total_unique_questions"] += unique_count
            stats["total_duplicates_removed"] += duplicate_count
            
            logging.info(f"  {category}: {unique_count} unique (within {dataset_name}), {duplicate_count} duplicates removed")
        
        # Save master combined file (all categories together)
        master_file = output_dir / "combined_all_categories.json"
        with open(master_file, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)
        
        # Save detailed statistics
        stats_file = output_dir / "consolidation_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        # Create summary report
        self.create_dataset_summary_report(output_dir, stats)
        
        logging.info(f"Dataset {dataset_name} consolidation completed:")
        logging.info(f"  Total unique questions (within dataset): {stats['total_unique_questions']}")
        logging.info(f"  Total duplicates removed (within dataset): {stats['total_duplicates_removed']}")
        
        return stats
    
    def create_dataset_summary_report(self, output_dir, stats):
        """
        Create a human-readable summary report for the dataset.
        """
        report_file = output_dir / "summary_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"DATASET CONSOLIDATION SUMMARY\n")
            f.write(f"="*50 + "\n\n")
            f.write(f"Dataset: {stats['dataset']}\n")
            f.write(f"Processed on: {stats['consolidation_timestamp']}\n")
            f.write(f"Uniqueness scope: {stats['uniqueness_scope']}\n")
            f.write(f"Chunks processed: {', '.join(stats['chunks_processed'])}\n\n")
            
            f.write(f"OVERALL STATISTICS (WITHIN THIS DATASET ONLY)\n")
            f.write(f"-"*50 + "\n")
            f.write(f"Total unique questions: {stats['total_unique_questions']}\n")
            f.write(f"Total duplicates removed: {stats['total_duplicates_removed']}\n")
            f.write(f"Overall deduplication rate: {(stats['total_duplicates_removed'] / (stats['total_unique_questions'] + stats['total_duplicates_removed']) * 100):.2f}%\n\n")
            
            f.write(f"CATEGORY BREAKDOWN\n")
            f.write(f"-"*30 + "\n")
            for category, cat_stats in stats['categories'].items():
                f.write(f"{category}:\n")
                f.write(f"  Unique questions: {cat_stats['unique_questions']}\n")
                f.write(f"  Duplicates removed: {cat_stats['duplicates_removed']}\n")
                f.write(f"  Total processed: {cat_stats['total_processed']}\n")
                f.write(f"  Deduplication rate: {cat_stats['deduplication_rate']}\n\n")
            
            f.write(f"NOTE: Uniqueness is determined independently within this dataset.\n")
            f.write(f"Questions may appear in other datasets but are considered unique within {stats['dataset']}.\n")
    
    def create_combined_dataset(self):
        """
        Main function to combine datasets across all chunks for each database.
        Each dataset is processed independently for uniqueness.
        """
        logging.info("Starting dataset consolidation process")
        logging.info("Uniqueness will be determined independently for each dataset")
        
        # Get all datasets and their chunks
        datasets_info = self.scan_all_datasets()
        
        if not datasets_info:
            logging.error("No datasets found to process")
            return
        
        # Process each dataset independently
        for dataset_name, chunks in datasets_info.items():
            try:
                logging.info(f"Processing dataset {dataset_name} independently")
                dataset_stats = self.combine_dataset_chunks(dataset_name, chunks)
                self.overall_stats["datasets"][dataset_name] = dataset_stats
                self.overall_stats["total_datasets_processed"] += 1
                self.overall_stats["total_unique_questions"] += dataset_stats["total_unique_questions"]
                self.overall_stats["total_duplicates_removed"] += dataset_stats["total_duplicates_removed"]
                
            except Exception as e:
                logging.error(f"Failed to process dataset {dataset_name}: {e}")
        
        # Save overall statistics
        self.save_overall_statistics()
        
        logging.info("Dataset consolidation process completed")
        logging.info("Each dataset was processed independently for uniqueness")
    
    def save_overall_statistics(self):
        """
        Save overall consolidation statistics.
        """
        overall_stats_file = self.output_path / "overall_consolidation_stats.json"
        with open(overall_stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.overall_stats, f, indent=2, ensure_ascii=False)
        
        # Create overall summary report
        self.create_overall_summary_report()
        
        logging.info(f"Overall statistics saved to {overall_stats_file}")
    
    def create_overall_summary_report(self):
        """
        Create an overall summary report across all datasets.
        """
        report_file = self.output_path / "overall_summary_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"OVERALL DATASET CONSOLIDATION SUMMARY\n")
            f.write(f"="*60 + "\n\n")
            f.write(f"Consolidation completed on: {self.overall_stats['consolidation_timestamp']}\n")
            f.write(f"Total datasets processed: {self.overall_stats['total_datasets_processed']}\n")
            f.write(f"Total unique questions across all datasets: {self.overall_stats['total_unique_questions']}\n")
            f.write(f"Total duplicates removed across all datasets: {self.overall_stats['total_duplicates_removed']}\n\n")
            
            f.write(f"IMPORTANT: Uniqueness was determined independently within each dataset.\n")
            f.write(f"Questions may appear across multiple datasets but are counted as unique within each dataset.\n\n")
            
            f.write(f"DATASET BREAKDOWN\n")
            f.write(f"-"*40 + "\n")
            for dataset_name, dataset_stats in self.overall_stats["datasets"].items():
                f.write(f"{dataset_name}:\n")
                f.write(f"  Unique questions (within dataset): {dataset_stats['total_unique_questions']}\n")
                f.write(f"  Duplicates removed (within dataset): {dataset_stats['total_duplicates_removed']}\n")
                f.write(f"  Chunks processed: {len(dataset_stats['chunks_processed'])}\n\n")
            
            f.write(f"CATEGORY SUMMARY ACROSS ALL DATASETS\n")
            f.write(f"-"*40 + "\n")
            
            # Aggregate by category
            category_totals = {}
            for dataset_stats in self.overall_stats["datasets"].values():
                for category, cat_stats in dataset_stats["categories"].items():
                    if category not in category_totals:
                        category_totals[category] = {"unique": 0, "duplicates": 0}
                    category_totals[category]["unique"] += cat_stats["unique_questions"]
                    category_totals[category]["duplicates"] += cat_stats["duplicates_removed"]
            
            for category, totals in category_totals.items():
                f.write(f"{category}:\n")
                f.write(f"  Total unique (across all datasets): {totals['unique']}\n")
                f.write(f"  Total duplicates removed (across all datasets): {totals['duplicates']}\n\n")
    
    def validate_output(self):
        """
        Validate the generated output files.
        """
        logging.info("Validating output files...")
        
        validation_results = {
            "datasets_validated": 0,
            "issues_found": [],
            "total_files_checked": 0
        }
        
        for dataset_dir in self.output_path.iterdir():
            if dataset_dir.is_dir() and dataset_dir.name != "__pycache__":
                validation_results["datasets_validated"] += 1
                
                # Check for required files
                required_files = ["consolidation_stats.json", "combined_all_categories.json", "summary_report.txt"]
                for req_file in required_files:
                    file_path = dataset_dir / req_file
                    validation_results["total_files_checked"] += 1
                    if not file_path.exists():
                        validation_results["issues_found"].append(f"Missing {req_file} in {dataset_dir.name}")
                
                # Check category files
                for category in self.query_categories:
                    category_file = dataset_dir / f"{category}.json"
                    validation_results["total_files_checked"] += 1
                    if category_file.exists():
                        try:
                            with open(category_file, 'r') as f:
                                data = json.load(f)
                                if not isinstance(data, list):
                                    validation_results["issues_found"].append(f"Invalid format in {dataset_dir.name}/{category}.json")
                        except json.JSONDecodeError:
                            validation_results["issues_found"].append(f"Invalid JSON in {dataset_dir.name}/{category}.json")
        
        # Save validation results
        validation_file = self.output_path / "validation_results.json"
        with open(validation_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        logging.info(f"Validation completed. Issues found: {len(validation_results['issues_found'])}")
        return validation_results


def main():
    """
    Main execution function.
    """
    # Configuration - using environment variables with fallbacks
  
    BASE_PATH = "< Enter this with your raw dataset folder path >"                
    
    OUTPUT_PATH = "< Enter this with your raw dataset folder path >"    
    
    temp = input("Enter the temperature for the dataset (e.g., '0.5'): ")
    
    # Initialize consolidator
    consolidator = DatasetConsolidator(BASE_PATH, OUTPUT_PATH, temp)
    
    # Run consolidation
    consolidator.create_combined_dataset()
    
    # Validate output
    validation_results = consolidator.validate_output()
    
    # Print summary
    print("\n" + "="*60)
    print("CONSOLIDATION COMPLETED")
    print("="*60)
    print(f"Total datasets processed: {consolidator.overall_stats['total_datasets_processed']}")
    print(f"Total unique questions: {consolidator.overall_stats['total_unique_questions']}")
    print(f"Total duplicates removed: {consolidator.overall_stats['total_duplicates_removed']}")
    print(f"Output directory: {consolidator.output_path}")
    print(f"Validation issues: {len(validation_results['issues_found'])}")
    print("\nIMPORTANT: Uniqueness was determined independently within each dataset.")
    
    if validation_results['issues_found']:
        print("\nValidation Issues:")
        for issue in validation_results['issues_found']:
            print(f"  - {issue}")
    
    print(f"\nDetailed logs available in: dataset_consolidation.log")
    print(f"Overall statistics: {consolidator.output_path}/overall_consolidation_stats.json")


if __name__ == "__main__":
    main()