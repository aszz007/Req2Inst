"""
UML Dataset Quality Validation Script

Validates the quality and integrity of UML dataset, including:
1. Instruction format completeness (3-part structure)
2. Things to Avoid section completeness
3. Correspondence between Description and Instruction
4. Error markers and empty values
5. JSON validity of Description field

Author: Dataset Validation System
Date: 2026-02-11
"""

import os
import sys
import json
import pandas as pd
import chardet
import re
from datetime import datetime
from typing import Dict, List, Tuple, Any


class UMLDatasetValidator:
    """UML Dataset Quality Validator"""

    def __init__(self, dataset_path: str, enable_period_check: bool = False):
        """
        Initialize validator

        Args:
            dataset_path: Path to the CSV dataset file
            enable_period_check: Whether to check for periods at end of sentences
        """
        self.dataset_path = dataset_path
        self.enable_period_check = enable_period_check
        self.validation_results = []
        self.error_count = 0
        self.warning_count = 0

    def detect_encoding(self, filepath: str) -> str:
        """Detect file encoding"""
        try:
            with open(filepath, 'rb') as f:
                raw_data = f.read(100000)
                result = chardet.detect(raw_data)
                return result['encoding']
        except Exception as e:
            print(f"Error detecting encoding: {e}")
            return 'utf-8'

    def load_dataset(self) -> pd.DataFrame:
        """Load dataset with automatic encoding detection"""
        print(f"\nLoading dataset: {os.path.basename(self.dataset_path)}")

        encoding = self.detect_encoding(self.dataset_path)
        print(f"Detected encoding: {encoding}")

        try:
            df = pd.read_csv(self.dataset_path, encoding=encoding)
            print(f"Loaded {len(df)} rows successfully\n")
            return df
        except Exception as e:
            print(f"Error loading with {encoding}, trying alternative encodings...")
            for enc in ['utf-8', 'gbk', 'gb18030', 'latin1']:
                try:
                    df = pd.read_csv(self.dataset_path, encoding=enc)
                    print(f"Successfully loaded with {enc} encoding")
                    print(f"Loaded {len(df)} rows\n")
                    return df
                except:
                    continue
            raise Exception(f"Failed to load dataset: {e}")

    def validate_json_description(self, description: str, row_num: int) -> Tuple[bool, List[str]]:
        """
        Validate JSON structure of Description field

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        if not description or pd.isna(description):
            errors.append("Description is empty")
            return False, errors

        try:
            desc_json = json.loads(description)

            # Check required fields
            required_fields = ['actors', 'use_cases', 'relationships', 'overall_description']
            for field in required_fields:
                if field not in desc_json:
                    errors.append(f"Missing required field: {field}")

            # Validate actors structure
            if 'actors' in desc_json:
                if not isinstance(desc_json['actors'], list):
                    errors.append("'actors' should be a list")
                else:
                    for idx, actor in enumerate(desc_json['actors']):
                        if not isinstance(actor, dict) or 'name' not in actor:
                            errors.append(f"Invalid actor structure at index {idx}")

            # Validate use_cases structure
            if 'use_cases' in desc_json:
                if not isinstance(desc_json['use_cases'], list):
                    errors.append("'use_cases' should be a list")
                else:
                    for idx, uc in enumerate(desc_json['use_cases']):
                        if not isinstance(uc, dict):
                            errors.append(f"Invalid use_case structure at index {idx}")
                        elif 'name' not in uc:
                            errors.append(f"Use case at index {idx} missing 'name'")

            # Validate relationships structure
            if 'relationships' in desc_json:
                if not isinstance(desc_json['relationships'], list):
                    errors.append("'relationships' should be a list")
                else:
                    for idx, rel in enumerate(desc_json['relationships']):
                        if not isinstance(rel, dict):
                            errors.append(f"Invalid relationship structure at index {idx}")
                        else:
                            required_rel_fields = ['type', 'from', 'to']
                            for field in required_rel_fields:
                                if field not in rel:
                                    errors.append(f"Relationship at index {idx} missing '{field}'")

        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON format: {str(e)}")
            return False, errors
        except Exception as e:
            errors.append(f"Unexpected error: {str(e)}")
            return False, errors

        return len(errors) == 0, errors

    def validate_three_part_format(self, instruction: str, row_num: int) -> Tuple[bool, List[str]]:
        """
        Validate 3-part instruction format:
        - Definition: ...
        - Emphasis & Caution: ...
        - Things to Avoid: ...

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        if not instruction or pd.isna(instruction) or instruction.strip() == '':
            errors.append("Instruction is empty")
            return False, errors

        lines = [line.strip() for line in instruction.strip().split('\n') if line.strip()]

        if len(lines) < 3:
            errors.append(f"Insufficient lines (expected 3, got {len(lines)})")

        has_definition = False
        has_emphasis = False
        has_avoid = False

        for line in lines:
            if line.startswith('Definition:'):
                has_definition = True
                content = line[len('Definition:'):].strip()
                if not content.lower().startswith('in this task'):
                    errors.append("Definition does not start with 'In this task'")
                if self.enable_period_check and not content.endswith('.'):
                    errors.append("Definition missing ending period")

            elif line.startswith('Emphasis & Caution:') or line.startswith('Emphasis and Caution:'):
                has_emphasis = True
                content = line.split(':', 1)[1].strip() if ':' in line else ""
                if self.enable_period_check and content and content != '-' and not content.endswith('.'):
                    errors.append("Emphasis & Caution missing ending period")

            elif line.startswith('Things to Avoid:'):
                has_avoid = True
                content = line[len('Things to Avoid:'):].strip()
                if self.enable_period_check and content and content != '-' and not content.endswith('.'):
                    errors.append("Things to Avoid missing ending period")

        if not has_definition:
            errors.append("Missing Definition section")
        if not has_emphasis:
            errors.append("Missing Emphasis & Caution section")
        if not has_avoid:
            errors.append("Missing Things to Avoid section")

        is_valid = (has_definition and has_emphasis and has_avoid and len(errors) == 0)
        return is_valid, errors

    def check_things_to_avoid_completeness(self, instruction: str, row_num: int) -> Tuple[bool, List[str]]:
        """
        Check if Things to Avoid section is complete and not just copied back

        Returns:
            (is_complete, warning_messages)
        """
        warnings = []

        if not instruction or pd.isna(instruction):
            return False, ["Instruction is empty"]

        # Extract Things to Avoid section
        avoid_pattern = r'Things to Avoid:\s*(.+?)(?:\n|$)'
        match = re.search(avoid_pattern, instruction, re.DOTALL)

        if not match:
            warnings.append("Could not find Things to Avoid section")
            return False, warnings

        avoid_content = match.group(1).strip()

        # Check if it's just a dash (incomplete)
        if avoid_content == '-':
            warnings.append("Things to Avoid is just a dash (incomplete)")
            return False, warnings

        # Check if it's suspiciously short
        if len(avoid_content) < 20:
            warnings.append(f"Things to Avoid is very short ({len(avoid_content)} chars)")
            return False, warnings

        # Check for common incomplete patterns
        incomplete_patterns = [
            r'^-\s*$',
            r'^\s*$',
            r'^TBD\s*$',
            r'^TODO\s*$',
            r'^N/A\s*$',
        ]

        for pattern in incomplete_patterns:
            if re.match(pattern, avoid_content, re.IGNORECASE):
                warnings.append(f"Things to Avoid appears incomplete: '{avoid_content}'")
                return False, warnings

        return True, []

    def validate_description_instruction_correspondence(
            self,
            description: str,
            instruction: str,
            row_num: int
    ) -> Tuple[bool, List[str]]:
        """
        Validate that Instruction corresponds to Description

        Checks:
        1. Key entities from Description appear in Instruction
        2. Use case names are referenced
        3. Actor names are mentioned

        Returns:
            (is_valid, warning_messages)
        """
        warnings = []

        if not description or not instruction:
            warnings.append("Description or Instruction is empty")
            return False, warnings

        try:
            desc_json = json.loads(description)

            # Extract key entities
            actors = [actor.get('name', '').lower() for actor in desc_json.get('actors', [])]
            use_cases = [uc.get('name', '').lower() for uc in desc_json.get('use_cases', [])]

            instruction_lower = instruction.lower()

            # Check if at least some use cases are mentioned
            use_cases_mentioned = sum(1 for uc in use_cases if uc and uc in instruction_lower)
            if len(use_cases) > 0 and use_cases_mentioned == 0:
                warnings.append("None of the use cases from Description are mentioned in Instruction")

            # Check if at least some actors are mentioned
            actors_mentioned = sum(1 for actor in actors if actor and actor in instruction_lower)
            if len(actors) > 0 and actors_mentioned == 0:
                warnings.append("None of the actors from Description are mentioned in Instruction")

            # Check for relationship types
            relationships = desc_json.get('relationships', [])
            has_include = any(rel.get('type') == 'include' for rel in relationships)
            has_extend = any(rel.get('type') == 'extend' for rel in relationships)

            if has_include and 'include' not in instruction_lower and 'required' not in instruction_lower:
                warnings.append("Description has 'include' relationships but Instruction doesn't mention them")

            if has_extend and 'extend' not in instruction_lower and 'optional' not in instruction_lower and 'conditional' not in instruction_lower:
                warnings.append("Description has 'extend' relationships but Instruction doesn't mention them")

        except json.JSONDecodeError:
            warnings.append("Cannot parse Description JSON for correspondence check")
            return False, warnings
        except Exception as e:
            warnings.append(f"Error checking correspondence: {str(e)}")
            return False, warnings

        # If there are warnings, it's a potential issue but not necessarily invalid
        return len(warnings) == 0, warnings

    def check_error_markers(self, instruction: str, row_num: int) -> Tuple[bool, List[str]]:
        """
        Check for ERROR markers in instruction

        Returns:
            (is_clean, error_messages)
        """
        errors = []

        if not instruction or pd.isna(instruction):
            errors.append("Instruction is empty")
            return False, errors

        # Check for various ERROR formats
        error_patterns = [
            r'ERROR\s*:',
            r'error\s*:',
            r'生成失败',
            r'generation failed',
            r'failed to generate',
        ]

        for pattern in error_patterns:
            if re.search(pattern, instruction, re.IGNORECASE):
                errors.append(f"Contains ERROR marker: matched pattern '{pattern}'")
                return False, errors

        return True, []

    def validate_row(self, row: pd.Series, row_num: int) -> Dict[str, Any]:
        """
        Validate a single row

        Returns:
            Dictionary containing validation results
        """
        result = {
            'row_num': row_num,
            'header': row.get('Header', 'N/A'),
            'is_valid': True,
            'errors': [],
            'warnings': []
        }

        description = str(row.get('Description', ''))
        instruction = str(row.get('Instruction', ''))

        # Check 1: JSON Description validity
        json_valid, json_errors = self.validate_json_description(description, row_num)
        if not json_valid:
            result['is_valid'] = False
            result['errors'].extend([f"[JSON] {err}" for err in json_errors])

        # Check 2: Error markers
        error_clean, error_messages = self.check_error_markers(instruction, row_num)
        if not error_clean:
            result['is_valid'] = False
            result['errors'].extend([f"[ERROR] {err}" for err in error_messages])

        # Check 3: Three-part format
        format_valid, format_errors = self.validate_three_part_format(instruction, row_num)
        if not format_valid:
            result['is_valid'] = False
            result['errors'].extend([f"[FORMAT] {err}" for err in format_errors])

        # Check 4: Things to Avoid completeness
        avoid_complete, avoid_warnings = self.check_things_to_avoid_completeness(instruction, row_num)
        if not avoid_complete:
            result['warnings'].extend([f"[AVOID] {warn}" for warn in avoid_warnings])

        # Check 5: Description-Instruction correspondence
        corr_valid, corr_warnings = self.validate_description_instruction_correspondence(
            description, instruction, row_num
        )
        if not corr_valid:
            result['warnings'].extend([f"[CORRESPONDENCE] {warn}" for warn in corr_warnings])

        return result

    def validate_dataset(self) -> List[Dict[str, Any]]:
        """
        Validate entire dataset

        Returns:
            List of validation results for each row
        """
        print("=" * 80)
        print("UML Dataset Quality Validation".center(80))
        print("=" * 80)
        print(f"Dataset: {os.path.basename(self.dataset_path)}")
        print(f"Period check: {'Enabled' if self.enable_period_check else 'Disabled'}")
        print("=" * 80)
        print()

        df = self.load_dataset()

        print("Starting validation...\n")

        results = []
        for idx, row in df.iterrows():
            row_num = idx + 1
            if row_num % 100 == 0:
                print(f"Progress: {row_num}/{len(df)} rows validated")

            result = self.validate_row(row, row_num)
            results.append(result)

            if not result['is_valid']:
                self.error_count += 1
            if result['warnings']:
                self.warning_count += 1

        self.validation_results = results
        return results

    def generate_report(self, save_path: str = None) -> str:
        """
        Generate detailed validation report

        Args:
            save_path: Optional path to save the report CSV

        Returns:
            Report summary as string
        """
        if not self.validation_results:
            return "No validation results available. Run validate_dataset() first."

        print("\n" + "=" * 80)
        print("Validation Report".center(80))
        print("=" * 80)

        total_rows = len(self.validation_results)
        valid_rows = sum(1 for r in self.validation_results if r['is_valid'])
        invalid_rows = total_rows - valid_rows
        rows_with_warnings = sum(1 for r in self.validation_results if r['warnings'])

        summary = f"""
Total Rows: {total_rows}
Valid Rows: {valid_rows} ({valid_rows / total_rows * 100:.1f}%)
Invalid Rows: {invalid_rows} ({invalid_rows / total_rows * 100:.1f}%)
Rows with Warnings: {rows_with_warnings} ({rows_with_warnings / total_rows * 100:.1f}%)
"""

        print(summary)

        # Print detailed errors
        if invalid_rows > 0:
            print("\n" + "-" * 80)
            print("Detailed Errors:")
            print("-" * 80)

            for result in self.validation_results:
                if not result['is_valid']:
                    print(f"\nRow {result['row_num']} [{result['header'][:40]}...]:")
                    for error in result['errors']:
                        print(f"  ERROR: {error}")
                    for warning in result['warnings']:
                        print(f"  WARNING: {warning}")

        # Print warnings separately
        if rows_with_warnings > 0:
            print("\n" + "-" * 80)
            print("Detailed Warnings:")
            print("-" * 80)

            warning_count = 0
            for result in self.validation_results:
                if result['warnings'] and result['is_valid']:  # Only valid rows with warnings
                    warning_count += 1
                    if warning_count <= 20:  # Limit output
                        print(f"\nRow {result['row_num']} [{result['header'][:40]}...]:")
                        for warning in result['warnings']:
                            print(f"  WARNING: {warning}")

            if warning_count > 20:
                print(f"\n... and {warning_count - 20} more rows with warnings")

        # Save to CSV if requested
        if save_path:
            self.save_report_csv(save_path)
            print(f"\nDetailed report saved to: {save_path}")

        print("=" * 80)

        return summary

    def save_report_csv(self, save_path: str):
        """Save validation report to CSV file"""
        report_data = []

        for result in self.validation_results:
            report_data.append({
                'Row': result['row_num'],
                'Header': result['header'],
                'Valid': result['is_valid'],
                'Errors': ' | '.join(result['errors']),
                'Warnings': ' | '.join(result['warnings'])
            })

        df_report = pd.DataFrame(report_data)
        df_report.to_csv(save_path, index=False, encoding='utf-8-sig')

    def get_error_rows(self) -> List[int]:
        """Get list of row numbers with errors"""
        return [r['row_num'] for r in self.validation_results if not r['is_valid']]

    def get_warning_rows(self) -> List[int]:
        """Get list of row numbers with warnings"""
        return [r['row_num'] for r in self.validation_results if r['warnings']]


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Validate UML Dataset Quality')
    parser.add_argument('--dataset', type=str,
                        default='dataset/uml/uml_dataset_qwen3_v3.csv',
                        help='Path to the dataset CSV file')
    parser.add_argument('--enable-period-check', action='store_true',
                        help='Enable period checking at end of sentences')
    parser.add_argument('--report-output', type=str,
                        default=None,
                        help='Path to save the validation report CSV')

    args = parser.parse_args()

    # Initialize validator
    validator = UMLDatasetValidator(
        dataset_path=args.dataset,
        enable_period_check=args.enable_period_check
    )

    # Run validation
    start_time = datetime.now()
    results = validator.validate_dataset()
    end_time = datetime.now()

    # Generate report
    if args.report_output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.report_output = f'outputs/validation/uml_validation_report_{timestamp}.csv'

    os.makedirs(os.path.dirname(args.report_output), exist_ok=True)

    summary = validator.generate_report(save_path=args.report_output)

    # Print timing
    duration = end_time - start_time
    print(f"\nValidation completed in {duration}")

    # Summary of results
    error_count = validator.error_count
    warning_count = validator.warning_count

    print(f"\n{'=' * 80}")
    print(f"Validation Summary".center(80))
    print(f"{'=' * 80}")
    print(f"Total Errors: {error_count}")
    print(f"Total Warnings: {warning_count}")
    print(f"Validation Report: {args.report_output}")
    print(f"{'=' * 80}")

    if error_count > 0:
        print(f"\nError rows: {validator.get_error_rows()[:20]}")
        if len(validator.get_error_rows()) > 20:
            print(f"... and {len(validator.get_error_rows()) - 20} more rows with errors")
        print(f"\nTo fix errors, run:")
        print(f"python scripts/dataset_preparation/uml_dataset_regenerate.py")

    if warning_count > 0:
        print(f"\nWarning rows: {validator.get_warning_rows()[:20]}")
        if len(validator.get_warning_rows()) > 20:
            print(f"... and {len(validator.get_warning_rows()) - 20} more rows with warnings")

    if error_count == 0 and warning_count == 0:
        print("\nDataset quality is excellent! No issues found.")

    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())