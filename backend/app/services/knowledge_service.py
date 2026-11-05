
import re
from pathlib import Path

from app.config import Config


class KnowledgeService:

    def __init__(self):
        self.knowledge_path = Path(Config.KNOWLEDGE_PATH)

    def get_lesion_information(self, lesion_code):

        if not self.knowledge_path.exists():
            return {
                "available": False,
                "title": None,
                "content": None,
                "message": "Educational information is unavailable."
            }

        markdown = self.knowledge_path.read_text(encoding="utf-8")

        # Match headings such as:
        # ## 1. AK — Actinic Keratosis
        pattern = r"^##\s+\d+\.\s+([A-Z]+)\s+—\s+(.+)$"

        matches = list(re.finditer(pattern, markdown, re.MULTILINE))

        for index, match in enumerate(matches):

            code = match.group(1).strip()
            title = match.group(2).strip()

            if code == lesion_code:

                start = match.end()
                end = (
                    matches[index + 1].start()
                    if index + 1 < len(matches)
                    else len(markdown)
                )

                content = markdown[start:end].strip()

                return {
                    "available": True,
                    "title": title,
                    "content": content,
                    "message": None
                }

        return {
            "available": False,
            "title": None,
            "content": None,
            "message": (
                f"Educational information for {lesion_code} "
                "is not available in the current knowledge file."
            )
        }


knowledge_service = KnowledgeService()
