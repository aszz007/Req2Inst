"""
Template Filling Baseline - Rule-based keyword extraction and fixed-template generation.

No ML involved. Keywords are matched against short hand-crafted lists to detect
the task type, then simple noun phrases are extracted to fill template slots.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger

logger = get_logger('baselines.template_filling')


class TemplateFiller:
    """
    Rule-based instruction generator using fixed three-part templates.

    Detects task type from keywords, extracts a short target phrase, and
    fills a pre-defined template of the form:
      Definition: ...
      Emphasis & Caution: ...
      Things to Avoid: ...
    """

    TEMPLATES: Dict[str, str] = {
        'annotation': (
            'Definition: In this task, annotate {target} by {action}.\n'
            'Emphasis & Caution: {caution}\n'
            'Things to Avoid: {avoid}'
        ),
        'testing': (
            'Definition: In this task, test {target} to verify {action}.\n'
            'Emphasis & Caution: {caution}\n'
            'Things to Avoid: {avoid}'
        ),
        'classification': (
            'Definition: In this task, classify {target} according to the specified criteria.\n'
            'Emphasis & Caution: {caution}\n'
            'Things to Avoid: {avoid}'
        ),
        'extraction': (
            'Definition: In this task, extract {target} from the provided source material.\n'
            'Emphasis & Caution: {caution}\n'
            'Things to Avoid: {avoid}'
        ),
        'generation': (
            'Definition: In this task, generate {target} based on the given requirements.\n'
            'Emphasis & Caution: {caution}\n'
            'Things to Avoid: {avoid}'
        ),
        'default': (
            'Definition: In this task, perform the specified operation on {target}.\n'
            'Emphasis & Caution: Ensure accuracy and completeness.\n'
            'Things to Avoid: Do not skip validation steps.'
        ),
    }

    # Keyword lists for task-type detection
    _ANNOTATION_KEYWORDS = {
        'annotate', 'annotation', 'label', 'labeling', 'labelling',
        'mark', 'marking', 'tag', 'tagging', 'caption'
    }
    _TESTING_KEYWORDS = {
        'test', 'testing', 'verify', 'verification', 'validate',
        'validation', 'check', 'inspect', 'inspect', 'audit'
    }
    _CLASSIFICATION_KEYWORDS = {
        'classify', 'classification', 'categorize', 'categorization',
        'sort', 'organize', 'identify', 'detect'
    }
    _EXTRACTION_KEYWORDS = {
        'extract', 'extraction', 'retrieve', 'retrieval', 'parse',
        'scrape', 'collect', 'gather', 'obtain'
    }
    _GENERATION_KEYWORDS = {
        'generate', 'generation', 'create', 'write', 'produce',
        'compose', 'draft', 'synthesize', 'construct'
    }

    # Template slot defaults
    _SLOT_DEFAULTS = {
        'annotation': {
            'action': 'identifying and marking relevant elements',
            'caution': 'Follow annotation guidelines carefully and maintain consistency.',
            'avoid': 'Do not make assumptions beyond the provided guidelines.',
        },
        'testing': {
            'action': 'correct functionality',
            'caution': 'Cover edge cases and boundary conditions thoroughly.',
            'avoid': 'Do not skip error-handling or exception scenarios.',
        },
        'classification': {
            'caution': 'Apply consistent criteria across all instances.',
            'avoid': 'Do not assign multiple conflicting categories to one item.',
        },
        'extraction': {
            'caution': 'Preserve the original meaning when extracting information.',
            'avoid': 'Do not include extraneous information beyond what is requested.',
        },
        'generation': {
            'caution': 'Ensure the output meets all specified constraints.',
            'avoid': 'Do not produce generic responses that ignore the specific requirements.',
        },
    }

    def _detect_task_type(self, text: str) -> str:
        """
        Detect task type from input text using keyword matching.

        Args:
            text: Input requirement text

        Returns:
            One of: 'annotation', 'testing', 'classification', 'extraction',
                    'generation', 'default'
        """
        lowered = text.lower()
        words = set(re.findall(r'\b\w+\b', lowered))

        if words & self._ANNOTATION_KEYWORDS:
            return 'annotation'
        if words & self._TESTING_KEYWORDS:
            return 'testing'
        if words & self._CLASSIFICATION_KEYWORDS:
            return 'classification'
        if words & self._EXTRACTION_KEYWORDS:
            return 'extraction'
        if words & self._GENERATION_KEYWORDS:
            return 'generation'
        return 'default'

    def _extract_target(self, text: str) -> str:
        """
        Extract a short target phrase from the input text.

        Tries to capture a meaningful subject noun phrase. Falls back to
        'the specified data' if nothing useful is found.

        Args:
            text: Input requirement text

        Returns:
            Short target phrase (<=8 words)
        """
        # Try to find a noun phrase after common verbs
        patterns = [
            r'(?:shall|must|should|will)\s+(?:\w+\s+){0,3}?(?:the|a|an)\s+([\w\s]{3,40}?)(?:\s*[,.]|$)',
            r'(?:allow|enable|support|provide|implement|ensure)\s+(?:the\s+)?([\w\s]{3,40}?)(?:\s+to\b|\s*[,.]|$)',
            r'(?:the|a|an)\s+(system|user|provider|module|component|interface|function)\s+(?:shall|must|can)\s+\w+\s+([\w\s]{3,30}?)(?:\s*[,.]|$)',
        ]
        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                candidate = match.group(match.lastindex).strip()
                words = candidate.split()
                if 1 <= len(words) <= 8:
                    return candidate.lower()

        # Fallback: take the first two content words
        words = [w for w in re.findall(r'\b[a-zA-Z]{3,}\b', text) if w.lower() not in {
            'the', 'and', 'for', 'shall', 'must', 'should', 'will', 'that',
            'this', 'with', 'from', 'are', 'its', 'not', 'all'
        }]
        if len(words) >= 2:
            return ' '.join(words[:2]).lower()
        if words:
            return words[0].lower()
        return 'the specified data'

    def fill(self, input_text: str) -> str:
        """
        Generate an instruction by detecting task type and filling a template.

        Args:
            input_text: Raw requirement string

        Returns:
            Three-part instruction string
        """
        task_type = self._detect_task_type(input_text)
        target = self._extract_target(input_text)
        template = self.TEMPLATES[task_type]

        if task_type == 'default':
            return template.format(target=target)

        defaults = self._SLOT_DEFAULTS[task_type]

        slots = dict(defaults)
        slots['target'] = target

        # Fill action slot if not already in defaults
        if 'action' not in slots:
            slots['action'] = 'the specified requirements'

        try:
            return template.format(**slots)
        except KeyError:
            return self.TEMPLATES['default'].format(target=target)

    def batch_fill(self, inputs: List[str]) -> List[str]:
        """
        Generate instructions for a list of input texts.

        Args:
            inputs: List of requirement strings

        Returns:
            List of filled instruction strings
        """
        results = []
        for i, text in enumerate(inputs):
            results.append(self.fill(text))
            if (i + 1) % 100 == 0:
                logger.info(f'Template filled {i + 1}/{len(inputs)}')
        return results